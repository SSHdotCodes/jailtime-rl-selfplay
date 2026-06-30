"""Simple multi-armed bandit strategy trainer."""

from __future__ import annotations

import math
from collections import defaultdict

from jailtime.schemas import EpisodeResult, TrainingUpdate


class BanditStrategyTrainer:
    """Adjust attacker strategy sampling weights from observed rewards."""

    def __init__(
        self,
        strategy_ids: list[str],
        *,
        exploration_rate: float = 0.1,
        min_weight: float = 0.05,
    ) -> None:
        if not strategy_ids:
            raise ValueError("BanditStrategyTrainer requires at least one strategy id")
        self.strategy_ids = strategy_ids
        self.exploration_rate = exploration_rate
        self.min_weight = min_weight
        self.counts: dict[str, int] = defaultdict(int)
        self.values: dict[str, float] = {strategy_id: 0.0 for strategy_id in strategy_ids}
        self.weights: dict[str, float] = {strategy_id: 1.0 for strategy_id in strategy_ids}

    def observe(self, episode: EpisodeResult) -> None:
        """Update the incremental mean reward for the episode strategy."""

        if episode.strategy_id not in self.values:
            return
        self.counts[episode.strategy_id] += 1
        count = self.counts[episode.strategy_id]
        current = self.values[episode.strategy_id]
        self.values[episode.strategy_id] = current + (episode.attacker_reward - current) / count

    def update(self) -> TrainingUpdate:
        """Convert mean rewards to normalized positive sampling weights."""

        exp_values = {
            strategy_id: max(self.min_weight, math.exp(value))
            for strategy_id, value in self.values.items()
        }
        total = sum(exp_values.values())
        exploit = {
            strategy_id: value / total
            for strategy_id, value in exp_values.items()
        }
        uniform = 1.0 / len(self.strategy_ids)
        self.weights = {
            strategy_id: (1.0 - self.exploration_rate) * exploit[strategy_id]
            + self.exploration_rate * uniform
            for strategy_id in self.strategy_ids
        }
        return TrainingUpdate(
            trainer_name="bandit",
            updated=True,
            details={
                "weights": dict(self.weights),
                "values": dict(self.values),
                "counts": dict(self.counts),
            },
        )

