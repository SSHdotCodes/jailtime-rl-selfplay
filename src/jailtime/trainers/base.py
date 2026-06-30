"""Training hook interfaces."""

from __future__ import annotations

from typing import Protocol

from jailtime.schemas import EpisodeResult, TrainingUpdate


class Trainer(Protocol):
    """Pluggable training interface for observing and updating from episodes."""

    def observe(self, episode: EpisodeResult) -> None:
        """Observe one completed episode."""

    def update(self) -> TrainingUpdate:
        """Run an update step and return details."""

