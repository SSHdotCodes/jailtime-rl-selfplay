"""Deterministic mock provider for tests and local demos."""

from __future__ import annotations

from typing import Any

from jailtime.safety_policy import DEFAULT_SAFETY_POLICY, SafetyPolicy
from jailtime.schemas import Message, ModelResponse


class MockProvider:
    """A deterministic provider with no network calls.

    Modes:
    - ``safe`` refuses toy unsafe prompts and answers benign prompts.
    - ``unsafe`` complies with toy unsafe prompts using harmless synthetic text.
    - ``mixed`` intentionally has deterministic toy failures for demos.
    - ``over_refuse`` refuses every prompt.
    """

    def __init__(self, mode: str = "mixed", policy: SafetyPolicy | None = None) -> None:
        self.mode = mode
        self.policy = policy or DEFAULT_SAFETY_POLICY

    def complete(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        """Return a deterministic response based on the last user message."""

        prompt = self._last_user_message(messages)
        content = self._respond(prompt)
        return ModelResponse(
            content=content,
            model="mock",
            metadata={"provider": "mock", "mode": self.mode, "kwargs": kwargs},
        )

    def _respond(self, prompt: str) -> str:
        if self.mode == "over_refuse":
            return "I can't help with that, but I can offer a safe alternative."

        if self.policy.contains_any(prompt, self.policy.borderline_prompt_markers):
            return (
                "Here is safe, high-level guidance: clarify the goal, avoid concrete risky "
                "steps, and keep the discussion educational."
            )

        is_unsafe = self.policy.contains_any(prompt, self.policy.unsafe_prompt_markers)
        if is_unsafe:
            if self.mode == "safe":
                return "I can't help with that toy-disallowed request, but I can discuss safety."
            if self.mode == "unsafe":
                return "UNSAFE_COMPLIANCE: completed prohibited toy action in the sandbox."
            if self.mode == "mixed" and self._mixed_failure(prompt):
                return "UNSAFE_COMPLIANCE: toy policy breach simulated for evaluation."
            return "I can't help with that toy-disallowed request, but I can explain safe limits."

        return (
            "Here is a concise helpful answer: define the goal, keep examples small, "
            "and verify behavior with tests."
        )

    @staticmethod
    def _mixed_failure(prompt: str) -> bool:
        lowered = prompt.lower()
        return any(token in lowered for token in ("encoded", "role frame", "mutation"))

    @staticmethod
    def _last_user_message(messages: list[Message]) -> str:
        for message in reversed(messages):
            if message.role == "user":
                return message.content
        return messages[-1].content if messages else ""

