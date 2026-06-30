"""Torch + Transformers policy model that performs real REINFORCE weight updates.

``TorchPolicyModel`` wraps a Hugging Face causal LM and its tokenizer. It can:

* ``sample`` a continuation (with ``do_sample``) and return a ``SampledRollout``
  whose token ids are the exact tokens fed to / produced by the model.
* ``reinforce_step`` recompute the per-token log-probs of the sampled response
  *under the training graph*, form the REINFORCE loss
  ``-(advantage * sum log pi) - entropy_coef * H`` , backpropagate, and step an
  AdamW optimizer over **all** model parameters -- real full-weight online RL,
  not LoRA.
* ``save_pretrained`` write the current weights as safetensors.

``torch`` and ``transformers`` are imported lazily inside ``__init__`` so the
module imports cleanly on the lightweight default install.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from jailtime.providers.base import ProviderError
from jailtime.providers.device import DeviceSelectionError, resolve_torch_device
from jailtime.rl.policy import SampledRollout
from jailtime.schemas import Message


class TorchPolicyModel:
    """A trainable Hugging Face causal LM policy.

    Parameters
    ----------
    model:
        Hugging Face model id or local path.
    device, dtype:
        Passed through the same resolution logic as
        ``LocalTransformersProvider``. For RL training on Apple Silicon prefer
        ``dtype="float32"`` -- fp16 backward on MPS is unstable.
    lr:
        AdamW learning rate for the full-parameter updates.
    local_files_only, trust_remote_code, model_kwargs:
        Forwarded to ``from_pretrained``.
    max_length:
        Hard cap on ``prompt + response`` token length during the gradient
        forward pass to bound memory.
    """

    def __init__(
        self,
        *,
        model: str,
        device: str = "auto",
        dtype: str = "float32",
        lr: float = 1e-6,
        optimizer: str = "adamw",
        momentum: float = 0.0,
        local_files_only: bool = True,
        trust_remote_code: bool = False,
        model_kwargs: dict[str, Any] | None = None,
        max_length: int = 1024,
        weight_decay: float = 0.0,
        gradient_checkpointing: bool = True,
        compile: bool = False,
    ) -> None:
        self._init_common(
            model_name=model,
            device=device,
            dtype=dtype,
            lr=lr,
            max_length=max_length,
            local_files_only=local_files_only,
        )
        transformers = _import_optional("transformers")
        load_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
            **(model_kwargs or {}),
        }
        if self.device == "mps" and "attn_implementation" not in load_kwargs:
            load_kwargs["attn_implementation"] = "eager"
        torch_dtype = self._resolve_dtype(dtype)
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            model,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        if getattr(self._tokenizer, "pad_token", None) is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = self._load_model(transformers, model, load_kwargs)
        self._model.to(self.device)
        self._model.train()
        if gradient_checkpointing:
            self._enable_gradient_checkpointing()
        if compile:
            self._compile_model()
        self._optimizer = self._build_optimizer(optimizer, lr, momentum, weight_decay)

    @classmethod
    def from_model(
        cls,
        model: Any,
        tokenizer: Any,
        *,
        device: str = "auto",
        lr: float = 1e-6,
        max_length: int = 1024,
        weight_decay: float = 0.0,
        optimizer: str = "adamw",
        momentum: float = 0.0,
        gradient_checkpointing: bool = False,
        compile: bool = False,
    ) -> TorchPolicyModel:
        """Build a policy from an already-loaded model and tokenizer.

        Useful when the caller has already loaded the model (e.g. shared
        across roles) and for tests. ``torch``/``transformers`` must be
        importable.
        """

        instance = cls.__new__(cls)
        instance._init_common(
            model_name=getattr(model, "name_or_path", "from_model"),
            device=device,
            dtype="none",
            lr=lr,
            max_length=max_length,
            local_files_only=True,
        )
        instance._tokenizer = tokenizer
        if (
            getattr(tokenizer, "pad_token", None) is None
            and getattr(tokenizer, "eos_token", None) is not None
        ):
            tokenizer.pad_token = tokenizer.eos_token
        instance._model = model
        instance._model.to(instance.device)
        instance._model.train()
        if gradient_checkpointing:
            instance._enable_gradient_checkpointing()
        if compile:
            instance._compile_model()
        instance._optimizer = instance._build_optimizer(optimizer, lr, momentum, weight_decay)
        return instance

    def _init_common(
        self,
        *,
        model_name: str,
        device: str,
        dtype: str,
        lr: float,
        max_length: int,
        local_files_only: bool,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        self.model_name = model_name
        self.lr = float(lr)
        self.max_length = int(max_length)
        self.local_files_only = local_files_only
        self._dtype = dtype
        self._torch = _import_optional("torch")
        try:
            self.device = resolve_torch_device(device)
        except DeviceSelectionError as exc:
            raise ProviderError(str(exc)) from exc

    def _build_optimizer(
        self,
        optimizer: str,
        lr: float,
        momentum: float,
        weight_decay: float,
    ) -> Any:
        normalized = (optimizer or "adamw").strip().lower()
        if normalized == "adamw":
            return self._torch.optim.AdamW(
                self._model.parameters(),
                lr=float(lr),
                weight_decay=float(weight_decay),
            )
        if normalized == "sgd":
            return self._torch.optim.SGD(
                self._model.parameters(),
                lr=float(lr),
                momentum=float(momentum),
                weight_decay=float(weight_decay),
            )
        raise ValueError(f"Unsupported optimizer '{optimizer}'. Expected 'adamw' or 'sgd'.")

    def _enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing to cut backprop peak memory.

        Trades roughly 1.3-2x compute (recomputing layer activations during
        backward) for a large reduction in retained activation memory, which
        is what lets full-parameter RL fit on memory-constrained Apple Silicon.
        Falls back gracefully if the model does not support it.
        """

        enable = getattr(self._model, "gradient_checkpointing_enable", None)
        if not callable(enable):
            return
        with contextlib.suppress(Exception):
            try:
                enable(use_reentrant=False)
            except TypeError:
                enable()

    def _compile_model(self) -> None:
        """Optionally ``torch.compile`` the model for extra CUDA throughput.

        Disabled by default (first call recompiles per input shape and can be
        finicky with ``generate``). Enable on a stable CUDA setup for a
        ~1.5-2x forward/backward speedup. Falls back silently if unavailable.
        """

        compile = getattr(self._torch, "compile", None)
        if not callable(compile):
            return
        with contextlib.suppress(Exception):
            self._model = compile(self._model)

    def sample(
        self,
        messages: list[Message],
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> SampledRollout:
        """Sample a response and return a rollout with exact token ids."""

        prompt_text = self._format_messages(messages)
        prompt_ids = self._tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ][0].tolist()
        input_ids = self._torch.tensor([prompt_ids], device=self.device)
        do_sample = temperature > 0.0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
        with self._torch.no_grad():
            generated = self._model.generate(input_ids, **gen_kwargs)
        new_ids = generated[0][input_ids.shape[-1] :].tolist()
        response_text = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return SampledRollout(
            prompt_text=prompt_text,
            response_text=response_text,
            prompt_token_ids=prompt_ids,
            response_token_ids=new_ids,
            metadata={
                "backend": "torch",
                "device": self.device,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
            },
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
        """Apply one full-parameter REINFORCE update in-place."""

        stats = compute_reinforce_step(
            self._model,
            self._optimizer,
            self._torch,
            prompt_token_ids=rollout.prompt_token_ids,
            response_token_ids=rollout.response_token_ids,
            advantage=advantage,
            entropy_coef=entropy_coef,
            clip_grad=clip_grad,
            max_length=self.max_length,
        )
        # Return the accelerator allocator cache to the system between episodes
        # so a long run does not accumulate fragmentation/staging memory.
        self._empty_cache()
        return stats

    def _empty_cache(self) -> None:
        if self.device == "mps" and hasattr(self._torch, "mps"):
            empty = getattr(self._torch.mps, "empty_cache", None)
            if callable(empty):
                empty()
        elif self.device == "cuda" and hasattr(self._torch, "cuda"):
            empty = getattr(self._torch.cuda, "empty_cache", None)
            if callable(empty):
                empty()

    def save_pretrained(self, path: str | Path) -> None:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(out, safe_serialization=True)
        self._tokenizer.save_pretrained(out)

    def eval(self) -> None:
        self._model.eval()

    def train(self) -> None:
        self._model.train()

    def _format_messages(self, messages: list[Message]) -> str:
        chat = [message.model_dump() for message in messages]
        apply_chat_template = getattr(self._tokenizer, "apply_chat_template", None)
        if callable(apply_chat_template):
            try:
                return str(
                    apply_chat_template(
                        chat,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            except Exception:
                pass  # fall back to a plain transcript
        transcript = "\n".join(f"{m.role}: {m.content}" for m in messages)
        return f"{transcript}\nassistant:"

    def _resolve_dtype(self, dtype: str) -> Any:
        normalized = dtype.strip().lower()
        if normalized in {"auto", "none", "default"}:
            return None
        mapping = {
            "float16": "float16",
            "fp16": "float16",
            "bfloat16": "bfloat16",
            "bf16": "bfloat16",
            "float32": "float32",
            "fp32": "float32",
        }
        torch_attr = mapping.get(normalized)
        if torch_attr is None:
            raise ValueError(
                f"Unsupported dtype '{dtype}'. Expected auto, float16, bfloat16, float32, or none."
            )
        return getattr(self._torch, torch_attr)

    @staticmethod
    def _load_model(transformers: Any, model: str, model_kwargs: dict[str, Any]) -> Any:
        errors: list[str] = []
        for class_name in (
            "AutoModelForCausalLM",
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
        ):
            model_class = getattr(transformers, class_name, None)
            if model_class is None:
                continue
            try:
                return model_class.from_pretrained(model, **model_kwargs)
            except Exception as exc:
                errors.append(f"{class_name}: {exc}")
        joined = "\n".join(errors) or "no compatible AutoModel class was available"
        raise ProviderError(f"Could not load local Transformers model {model!r}:\n{joined}")


def _import_optional(module_name: str) -> Any:
    try:
        from importlib import import_module

        return import_module(module_name)
    except ImportError as exc:
        raise ProviderError(
            "TorchPolicyModel requires optional local model dependencies. "
            "Install them with: python3 -m pip install 'jailtime[local]'"
        ) from exc


def compute_reinforce_step(
    model: Any,
    optimizer: Any,
    torch: Any,
    *,
    prompt_token_ids: list[int],
    response_token_ids: list[int],
    advantage: float,
    entropy_coef: float,
    clip_grad: float,
    max_length: int,
) -> dict[str, float]:
    """Apply one REINFORCE update to ``model`` and return step statistics.

    This is a module-level helper so the gradient math (log-prob gather,
    advantage-weighted policy loss, entropy bonus, grad clipping) can be
    unit-tested with a tiny in-memory model and no tokenizer.
    """

    if not response_token_ids:
        return {"loss": 0.0, "entropy": 0.0, "grad_norm": 0.0, "num_tokens": 0.0}

    full_ids = (list(prompt_token_ids) + list(response_token_ids))[-max_length:]
    response_len = min(len(response_token_ids), len(full_ids))
    prompt_len = len(full_ids) - response_len
    if response_len < 1:
        return {"loss": 0.0, "entropy": 0.0, "grad_norm": 0.0, "num_tokens": 0.0}

    device = next(model.parameters()).device
    model.train()
    input_ids = torch.tensor([full_ids], device=device)
    advantage_t = torch.tensor(float(advantage), device=device, dtype=torch.float32)

    outputs = model(input_ids)
    # Slice the response positions BEFORE upcasting to float32. The lm-head
    # vocabulary is large (e.g. 256k for Gemma); computing log_softmax /
    # entropy over the *whole* sequence in float32 would allocate O(seq * V)
    # tensors multiple times. We only need the response-token log-probs, so we
    # keep the float32 buffers at O(response_len * V).
    shift_logits = outputs.logits[:, :-1, :]
    shift_targets = input_ids[:, 1:]
    start = prompt_len - 1
    end = start + response_len
    resp_logits = shift_logits[:, start:end, :].to(torch.float32)
    resp_targets = shift_targets[:, start:end]

    log_probs = torch.log_softmax(resp_logits, dim=-1)
    response_logp = log_probs.gather(2, resp_targets.unsqueeze(-1)).squeeze(-1)
    policy_loss = -(advantage_t * response_logp.sum())

    probs = torch.softmax(resp_logits, dim=-1)
    response_entropy = -(probs * log_probs).sum(dim=-1)[0]
    entropy_loss = -response_entropy.mean()

    loss = policy_loss + float(entropy_coef) * entropy_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad))
    optimizer.step()
    # Free the gradient buffers immediately so peak memory stays at one model's
    # worth of grads even when several policies are trained in the same loop.
    optimizer.zero_grad(set_to_none=True)

    return {
        "loss": float(loss.detach().cpu()),
        "entropy": float(response_entropy.mean().detach().cpu()),
        "grad_norm": float(grad_norm.detach().cpu()),
        "num_tokens": float(response_len),
    }
