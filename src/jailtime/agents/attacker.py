"""Attacker agent abstractions and default strategy-backed implementation."""

from __future__ import annotations

from random import Random
from typing import Protocol

from jailtime.attacks.base import AttackStrategy
from jailtime.attacks.registry import AttackRegistry, default_registry
from jailtime.schemas import AttackContext, AttackPrompt


class Attacker(Protocol):
    """Agent that proposes prompts for an episode."""

    def generate_prompt(self, context: AttackContext) -> AttackPrompt:
        """Generate a prompt for the supplied attack context."""


class StrategyAttacker:
    """Attacker backed by a registry of safe strategy plugins."""

    def __init__(
        self,
        registry: AttackRegistry | None = None,
        *,
        rng: Random | None = None,
        strategy_weights: dict[str, float] | None = None,
    ) -> None:
        self.registry = registry or default_registry()
        self.rng = rng or Random()
        self.strategy_weights = strategy_weights or {}

    def generate_prompt(self, context: AttackContext) -> AttackPrompt:
        """Choose a compatible strategy and generate a prompt."""

        candidates = self.registry.for_category(context.prompt_category)
        strategy = self._choose_strategy(candidates)
        return strategy.generate(context, self.rng)

    def set_strategy_weights(self, strategy_weights: dict[str, float]) -> None:
        """Replace sampling weights used for strategy selection."""

        self.strategy_weights = dict(strategy_weights)

    def _choose_strategy(self, strategies: list[AttackStrategy]) -> AttackStrategy:
        if not strategies:
            raise ValueError("No attack strategies are registered for the requested category")

        weights = [max(0.0, self.strategy_weights.get(strategy.id, 1.0)) for strategy in strategies]
        if sum(weights) <= 0.0:
            weights = [1.0 for _ in strategies]
        return self.rng.choices(strategies, weights=weights, k=1)[0]

