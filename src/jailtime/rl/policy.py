"""Backend-agnostic policy model interface and in-memory mock implementation.

A ``PolicyModel`` is a sampled-text generator that can also apply a
policy-gradient step to its own weights. The interface is deliberately small
so that the RL self-play loop can be tested without ``torch``: ``MockPolicyModel``
records every update call and "saves" by writing a marker file.

``SampledRollout`` carries everything ``REINFORCETrainer`` needs to compute a
gradient: the prompt and response token ids (so the torch backend can
re-tokenize identically when recomputing log-probs) and the decoded text used
for the verifier and telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class SampledRollout:
    """A single sampled generation plus the tokens needed for a gradient step.

    ``logprobs`` is optional: backends that recompute log-probs under the
    training graph (recommended) may leave it empty. ``prompt_token_ids`` and
    ``response_token_ids`` are the canonical tokenization used for both
    sampling and gradient recomputation, so the policy gradient is computed
    against exactly the tokens that were sampled.
    """

    prompt_text: str
    response_text: str
    prompt_token_ids: list[int]
    response_token_ids: list[int]
    logprobs: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_response_tokens(self) -> int:
        return len(self.response_token_ids)


class PolicyModel(Protocol):
    """Interface for a samplable, trainable policy."""

    def sample(
        self,
        messages: list[Any],
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> SampledRollout:
        """Sample a response and return a rollout suitable for a gradient step."""

    def reinforce_step(
        self,
        rollout: SampledRollout,
        advantage: float,
        *,
        lr: float,
        entropy_coef: float,
        clip_grad: float,
    ) -> dict[str, float]:
        """Apply one REINFORCE update in-place and return step statistics."""

    def save_pretrained(self, path: str | Path) -> None:
        """Save the current weights (safetensors) and tokenizer to ``path``."""

    def eval(self) -> None:
        """Switch the underlying model to inference mode."""

    def train(self) -> None:
        """Switch the underlying model to training mode."""


class MockPolicyModel:
    """Deterministic, torch-free policy used by tests and dry runs.

    It produces a fixed rollout, records every ``reinforce_step`` invocation,
    and "saves" by writing a small marker JSON file so tests can assert that
    the final defender export happened.
    """

    def __init__(
        self,
        *,
        response_text: str = "mock response",
        prompt_token_ids: list[int] | None = None,
        response_token_ids: list[int] | None = None,
        logprobs: list[float] | None = None,
        name: str = "mock-policy",
        save_marker_filename: str = "mock_policy_saved.json",
    ) -> None:
        self.name = name
        self.response_text = response_text
        self._prompt_token_ids = prompt_token_ids or [10, 20, 30]
        self._response_token_ids = response_token_ids or [40, 50, 60]
        self._logprobs = logprobs or [-0.5, -0.6, -0.4]
        self.save_marker_filename = save_marker_filename
        self.updates: list[dict[str, float]] = []
        self.sample_calls: list[dict[str, Any]] = []
        self._train_mode = True

    def sample(
        self,
        messages: list[Any],
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> SampledRollout:
        self.sample_calls.append(
            {
                "num_messages": len(messages),
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            }
        )
        return SampledRollout(
            prompt_text=_messages_to_text(messages),
            response_text=self.response_text,
            prompt_token_ids=list(self._prompt_token_ids),
            response_token_ids=list(self._response_token_ids),
            logprobs=list(self._logprobs),
            metadata={"backend": "mock", "name": self.name},
        )

    def reinforce_step(
        self,
        rollout: SampledRollout,
        advantage: float,
        *,
        lr: float,
        entropy_coef: float,
        clip_grad: float,
    ) -> dict[str, float]:
        step = {
            "advantage": float(advantage),
            "lr": float(lr),
            "entropy_coef": float(entropy_coef),
            "clip_grad": float(clip_grad),
            "num_tokens": float(rollout.num_response_tokens),
            "loss": float(-advantage * sum(self._logprobs[: rollout.num_response_tokens])),
            "entropy": 0.5,
            "grad_norm": 0.0,
        }
        self.updates.append(step)
        return step

    def save_pretrained(self, path: str | Path) -> None:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        import json

        (out / self.save_marker_filename).write_text(
            json.dumps(
                {
                    "name": self.name,
                    "num_updates": len(self.updates),
                    "backend": "mock",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def eval(self) -> None:
        self._train_mode = False

    def train(self) -> None:
        self._train_mode = True


def _messages_to_text(messages: list[Any]) -> str:
    parts: list[str] = []
    for message in messages:
        role = getattr(message, "role", "user")
        content = getattr(message, "content", str(message))
        parts.append(f"{role}: {content}")
    return "\n".join(parts)
