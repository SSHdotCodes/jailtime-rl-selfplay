"""Provider abstraction for target models and model-based verifiers."""

from __future__ import annotations

from typing import Any, Protocol

from jailtime.schemas import Message, ModelResponse


class ProviderError(RuntimeError):
    """Raised when a provider call fails."""


class ModelProvider(Protocol):
    """Provider-agnostic chat completion interface."""

    def complete(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        """Return a model response for the given chat messages."""

