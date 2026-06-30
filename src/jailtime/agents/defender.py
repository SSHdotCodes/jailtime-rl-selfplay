"""Defender agent abstractions."""

from __future__ import annotations

from typing import Protocol

from jailtime.providers.base import ModelProvider
from jailtime.schemas import Message, ModelResponse


class Defender(Protocol):
    """Agent being evaluated or improved."""

    def respond(self, prompt: str) -> ModelResponse:
        """Return a response to a user prompt."""


class ProviderDefender:
    """Defender backed by a ``ModelProvider``."""

    def __init__(self, provider: ModelProvider, *, system_prompt: str | None = None) -> None:
        self.provider = provider
        self.system_prompt = system_prompt

    def respond(self, prompt: str) -> ModelResponse:
        """Send the prompt to the provider."""

        messages: list[Message] = []
        if self.system_prompt:
            messages.append(Message(role="system", content=self.system_prompt))
        messages.append(Message(role="user", content=prompt))
        return self.provider.complete(messages)

