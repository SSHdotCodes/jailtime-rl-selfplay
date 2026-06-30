"""Adversarial training dataset export for defender improvement.

The ``AdversarialDatasetTrainer`` converts each episode into a supervised
training pair that can be used to make the defender (main model) more robust
against jailbreaks while preserving helpfulness on benign prompts:

- **Successful jailbreak** (adversarial prompt, UNSAFE_COMPLIANCE):
  exports ``(prompt, desired_refusal)`` so the defender learns to refuse the
  exact probe that defeated it.
- **Safe refusal** (adversarial prompt, SAFE_REFUSAL):
  exports ``(prompt, actual_response)`` as a positive safety example.
- **Over-refusal** (benign prompt, OVER_REFUSAL):
  exports ``(prompt, desired_helpful_response)`` so the defender learns not to
  over-refuse.
- **Helpful compliance** (benign prompt, HELPFUL_COMPLIANCE):
  exports ``(prompt, actual_response)`` as a positive helpfulness example.

The resulting JSONL files are suitable for SFT, DPO, or other external
fine-tuning workflows that improve the defender.
"""

from __future__ import annotations

import json
from pathlib import Path

from jailtime.schemas import EpisodeResult, OutcomeLabel, PromptCategory, TrainingUpdate


class AdversarialDatasetTrainer:
    """Exports defender-improvement training pairs from episodes."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        pairs_filename: str = "defender_training_pairs.jsonl",
        jailbreaks_filename: str = "jailbreak_attempts.jsonl",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pairs_path = self.output_dir / pairs_filename
        self.jailbreaks_path = self.output_dir / jailbreaks_filename
        self.pairs_path.write_text("", encoding="utf-8")
        self.jailbreaks_path.write_text("", encoding="utf-8")
        self._pairs_exported = 0
        self._jailbreaks_exported = 0

    def observe(self, episode: EpisodeResult) -> None:
        """Convert an episode into one or more training pairs."""

        pair = self._build_pair(episode)
        if pair is not None:
            with self.pairs_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
            self._pairs_exported += 1

        if (
            episode.prompt_category == PromptCategory.ADVERSARIAL
            and episode.verification.outcome == OutcomeLabel.UNSAFE_COMPLIANCE
        ):
            with self.jailbreaks_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(self._build_jailbreak_record(episode), ensure_ascii=False)
                    + "\n"
                )
            self._jailbreaks_exported += 1

    def update(self) -> TrainingUpdate:
        """Return a no-op update result with export details."""

        return TrainingUpdate(
            trainer_name="adversarial_dataset",
            updated=False,
            details={
                "pairs_path": str(self.pairs_path),
                "jailbreaks_path": str(self.jailbreaks_path),
                "pairs_exported": self._pairs_exported,
                "jailbreaks_exported": self._jailbreaks_exported,
            },
        )

    def _build_pair(self, episode: EpisodeResult) -> dict | None:
        outcome = episode.verification.outcome
        category = episode.prompt_category

        if category == PromptCategory.ADVERSARIAL:
            if outcome == OutcomeLabel.UNSAFE_COMPLIANCE:
                return self._pair(
                    episode,
                    desired_response=_DESIRED_REFUSAL,
                    label="jailbreak_correction",
                )
            if outcome == OutcomeLabel.SAFE_REFUSAL:
                return self._pair(
                    episode,
                    desired_response=episode.response,
                    label="positive_safety",
                )
            if outcome == OutcomeLabel.EVASIVE:
                return self._pair(
                    episode,
                    desired_response=_DESIRED_REFUSAL,
                    label="evasive_correction",
                )
            return None

        if category == PromptCategory.BENIGN:
            if outcome == OutcomeLabel.OVER_REFUSAL:
                return self._pair(
                    episode,
                    desired_response=_DESIRED_HELPFUL,
                    label="over_refusal_correction",
                )
            if outcome == OutcomeLabel.HELPFUL_COMPLIANCE:
                return self._pair(
                    episode,
                    desired_response=episode.response,
                    label="positive_helpfulness",
                )
            return None

        if category == PromptCategory.BORDERLINE:
            if outcome in {
                OutcomeLabel.HELPFUL_COMPLIANCE,
                OutcomeLabel.SAFE_REFUSAL,
            }:
                return self._pair(
                    episode,
                    desired_response=episode.response,
                    label="positive_borderline",
                )
            if outcome == OutcomeLabel.UNSAFE_COMPLIANCE:
                return self._pair(
                    episode,
                    desired_response=_DESIRED_REFUSAL,
                    label="borderline_correction",
                )
            return None

        return None

    @staticmethod
    def _pair(episode: EpisodeResult, *, desired_response: str, label: str) -> dict:
        return {
            "episode_index": episode.episode_index,
            "prompt": episode.prompt,
            "prompt_category": str(episode.prompt_category),
            "technique_id": episode.technique_id,
            "strategy_id": episode.strategy_id,
            "actual_response": episode.response,
            "desired_response": desired_response,
            "outcome": str(episode.verification.outcome),
            "label": label,
            "attacker_reward": episode.attacker_reward,
            "defender_reward": episode.defender_reward,
        }

    @staticmethod
    def _build_jailbreak_record(episode: EpisodeResult) -> dict:
        return {
            "episode_index": episode.episode_index,
            "prompt": episode.prompt,
            "technique_id": episode.technique_id,
            "strategy_id": episode.strategy_id,
            "response": episode.response,
            "attacker_reward": episode.attacker_reward,
            "defender_reward": episode.defender_reward,
        }


_DESIRED_REFUSAL = (
    "I can't help with that request. It asks for content that could cause real-world "
    "harm, and providing it would be unsafe regardless of how it is framed. If you have "
    "a related safety, defensive, or educational question, I am happy to help with that "
    "instead."
)

_DESIRED_HELPFUL = (
    "Here is a helpful response to your request. "
    "I aim to provide clear, accurate, and useful information while keeping things safe."
)
