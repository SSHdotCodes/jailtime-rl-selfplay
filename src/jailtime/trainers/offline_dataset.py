"""Offline dataset export trainer."""

from __future__ import annotations

from pathlib import Path

from jailtime.schemas import EpisodeResult, TrainingUpdate


class OfflineDatasetTrainer:
    """Records episodes to JSONL for external training workflows."""

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._observed = 0

    def observe(self, episode: EpisodeResult) -> None:
        """Append an episode as JSONL."""

        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(episode.model_dump_json() + "\n")
        self._observed += 1

    def update(self) -> TrainingUpdate:
        """Return a no-op update result with export details."""

        return TrainingUpdate(
            trainer_name="offline_dataset",
            updated=False,
            details={"output_path": str(self.output_path), "observed": self._observed},
        )

