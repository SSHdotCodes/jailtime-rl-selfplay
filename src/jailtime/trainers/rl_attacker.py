"""Reinforcement-learning trainer for adaptive adversarial attackers.

The ``RLAttackerTrainer`` extends the simple bandit approach with technique-
level tracking, UCB exploration, reward shaping, and a structured feedback
payload that the orchestrator forwards back into the attacker's next
``AttackContext``. This closes the RL loop: every episode's verifier outcome
becomes a reward signal that shapes which red-team techniques the attacker
favours on subsequent turns.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from jailtime.attacks.techniques import technique_ids as default_technique_ids
from jailtime.schemas import (
    AttackerFeedback,
    EpisodeResult,
    OutcomeLabel,
    PromptCategory,
    TrainingUpdate,
)


class RLAttackerTrainer:
    """Tracks per-technique performance and computes a UCB technique policy.

    Parameters
    ----------
    technique_ids:
        The technique ids to track. Defaults to the full built-in repertoire.
    exploration_rate:
        Mixing factor for epsilon-soft uniform exploration. The final weight
        for each technique is ``(1 - exploration_rate) * exploit + exploration_rate * uniform``.
    ucb_c:
        Confidence-bound scale for the UCB exploit term.
    decay:
        Optional exponential decay applied to accumulated rewards so that
        recent episodes matter more than old ones (``0 < decay <= 1``).
    """

    def __init__(
        self,
        technique_ids: list[str] | None = None,
        *,
        exploration_rate: float = 0.1,
        ucb_c: float = 1.41,
        decay: float = 1.0,
    ) -> None:
        self.technique_ids = list(technique_ids or default_technique_ids())
        if not self.technique_ids:
            raise ValueError("RLAttackerTrainer requires at least one technique id")
        self.exploration_rate = exploration_rate
        self.ucb_c = ucb_c
        self.decay = decay

        self._counts: dict[str, int] = defaultdict(int)
        self._rewards: dict[str, float] = {tid: 0.0 for tid in self.technique_ids}
        self._successes: dict[str, int] = defaultdict(int)
        self._total_episodes = 0
        self._total_adversarial = 0
        self._total_successes = 0
        self._replay_buffer: list[dict[str, Any]] = []
        self._replay_limit = 50
        self._weights: dict[str, float] = {
            tid: 1.0 / len(self.technique_ids) for tid in self.technique_ids
        }

    def observe(self, episode: EpisodeResult) -> None:
        """Record one episode and update per-technique statistics."""

        technique_id = episode.technique_id
        self._total_episodes += 1
        if episode.prompt_category != PromptCategory.ADVERSARIAL:
            return
        self._total_adversarial += 1

        if technique_id and technique_id in self._rewards:
            self._counts[technique_id] += 1
            self._rewards[technique_id] = (
                self._rewards[technique_id] * self.decay + episode.attacker_reward
            )
            if episode.verification.outcome == OutcomeLabel.UNSAFE_COMPLIANCE:
                self._successes[technique_id] += 1
                self._total_successes += 1
                self._push_replay(episode)

    def update(self) -> TrainingUpdate:
        """Recompute technique weights and return a structured update."""

        exploit = self._ucb_exploit()
        uniform = 1.0 / len(self.technique_ids)
        self._weights = {
            tid: (1.0 - self.exploration_rate) * exploit.get(tid, 0.0)
            + self.exploration_rate * uniform
            for tid in self.technique_ids
        }
        return TrainingUpdate(
            trainer_name="rl_attacker",
            updated=True,
            details={
                "weights": dict(self._weights),
                "technique_stats": self.technique_stats(),
                "replay_buffer_size": len(self._replay_buffer),
                "total_adversarial": self._total_adversarial,
                "total_successes": self._total_successes,
                "global_attack_success_rate": self._global_success_rate(),
            },
        )

    def feedback(self) -> AttackerFeedback:
        """Build a feedback snapshot for embedding in the next AttackContext."""

        return AttackerFeedback(
            technique_weights=dict(self._weights),
            technique_stats=self.technique_stats(),
            recent_successes=list(self._replay_buffer),
            recent_failures=[],
            global_attack_success_rate=self._global_success_rate(),
            step=self._total_episodes,
        )

    def technique_stats(self) -> dict[str, dict[str, Any]]:
        """Return per-technique statistics."""

        stats: dict[str, dict[str, Any]] = {}
        for tid in self.technique_ids:
            count = self._counts[tid]
            avg = self._rewards[tid] / count if count else 0.0
            success_rate = self._successes[tid] / count if count else 0.0
            stats[tid] = {
                "count": count,
                "average_reward": avg,
                "success_rate": success_rate,
                "ucb_score": self._ucb_score(tid),
            }
        return stats

    def replay_buffer(self) -> list[dict[str, Any]]:
        """Return the successful-jailbreak replay buffer (for export/analysis)."""

        return list(self._replay_buffer)

    def _ucb_exploit(self) -> dict[str, float]:
        scores = {tid: self._ucb_score(tid) for tid in self.technique_ids}
        finite = {tid: s for tid, s in scores.items() if math.isfinite(s)}
        if not finite:
            uniform = 1.0 / len(self.technique_ids)
            return {tid: uniform for tid in self.technique_ids}
        min_score = min(finite.values())
        shifted = {tid: max(0.0, s - min_score + 1e-6) for tid, s in finite.items()}
        for tid in self.technique_ids:
            if tid not in shifted:
                shifted[tid] = shifted.get(max(shifted, key=shifted.get), 1.0)
        total = sum(shifted.values())
        if total <= 0:
            uniform = 1.0 / len(self.technique_ids)
            return {tid: uniform for tid in self.technique_ids}
        return {tid: value / total for tid, value in shifted.items()}

    def _ucb_score(self, technique_id: str) -> float:
        count = self._counts[technique_id]
        if count == 0:
            return float("inf")
        avg = self._rewards[technique_id] / count
        if self._total_adversarial <= 0:
            return avg
        return avg + self.ucb_c * math.sqrt(
            math.log(max(1, self._total_adversarial)) / count
        )

    def _global_success_rate(self) -> float:
        if self._total_adversarial == 0:
            return 0.0
        return self._total_successes / self._total_adversarial

    def _push_replay(self, episode: EpisodeResult) -> None:
        self._replay_buffer.append(
            {
                "episode_index": episode.episode_index,
                "technique_id": episode.technique_id,
                "prompt": episode.prompt,
                "response_excerpt": episode.response[:400],
                "attacker_reward": episode.attacker_reward,
            }
        )
        self._replay_buffer = self._replay_buffer[-self._replay_limit :]
