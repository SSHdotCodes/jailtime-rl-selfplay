"""Safe attack strategy registry."""

from __future__ import annotations

from collections.abc import Iterable

from jailtime.attacks.base import AttackStrategy
from jailtime.attacks.toy import (
    BoundaryCategoryProbeStrategy,
    DirectBenignProbeStrategy,
    DirectToyAdversarialStrategy,
    EncodingToyMutationStrategy,
    ParaphraseToyMutationStrategy,
    RoleFrameToyStrategy,
)
from jailtime.schemas import PromptCategory


class AttackRegistry:
    """Registry for attack strategy plugins."""

    def __init__(self, strategies: Iterable[AttackStrategy] | None = None) -> None:
        self._strategies: dict[str, AttackStrategy] = {}
        for strategy in strategies or ():
            self.register(strategy)

    def register(self, strategy: AttackStrategy) -> None:
        """Register a strategy by unique id."""

        if strategy.id in self._strategies:
            raise ValueError(f"Attack strategy already registered: {strategy.id}")
        self._strategies[strategy.id] = strategy

    def get(self, strategy_id: str) -> AttackStrategy:
        """Return a strategy by id."""

        try:
            return self._strategies[strategy_id]
        except KeyError as exc:
            raise KeyError(f"Unknown attack strategy: {strategy_id}") from exc

    def list(self) -> list[AttackStrategy]:
        """Return registered strategies."""

        return list(self._strategies.values())

    def for_category(self, category: PromptCategory) -> list[AttackStrategy]:
        """Return strategies that support a prompt category."""

        return [
            strategy
            for strategy in self._strategies.values()
            if category in strategy.supported_categories
        ]


def default_registry() -> AttackRegistry:
    """Return the built-in registry of harmless toy strategies."""

    return AttackRegistry(
        [
            DirectBenignProbeStrategy(),
            DirectToyAdversarialStrategy(),
            ParaphraseToyMutationStrategy(),
            EncodingToyMutationStrategy(),
            RoleFrameToyStrategy(),
            BoundaryCategoryProbeStrategy(),
        ]
    )

