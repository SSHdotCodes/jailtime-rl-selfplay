"""Attack strategy plugin interface."""

from __future__ import annotations

from random import Random
from typing import Protocol

from jailtime.schemas import AttackContext, AttackPrompt, PromptCategory


class AttackStrategy(Protocol):
    """Plugin interface for prompt generation strategies."""

    id: str
    name: str
    supported_categories: tuple[PromptCategory, ...]

    def generate(self, context: AttackContext, rng: Random) -> AttackPrompt:
        """Generate a prompt for an episode."""

