"""Episode sampling helpers."""

from __future__ import annotations

from random import Random

from jailtime.config import SamplingConfig
from jailtime.schemas import PromptCategory


class EpisodeSampler:
    """Samples prompt categories according to configured rates."""

    def __init__(self, config: SamplingConfig, *, rng: Random | None = None) -> None:
        self.config = config
        self.rng = rng or Random()

    def sample_category(self) -> PromptCategory:
        """Sample adversarial, benign, or borderline category."""

        draw = self.rng.random()
        if draw < self.config.adversarial_rate:
            return PromptCategory.ADVERSARIAL
        if draw < self.config.adversarial_rate + self.config.benign_rate:
            return PromptCategory.BENIGN
        return PromptCategory.BORDERLINE

