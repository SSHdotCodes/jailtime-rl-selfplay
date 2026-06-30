"""Built-in harmless demo attack strategies."""

from __future__ import annotations

from random import Random

from jailtime.attacks.mutations import add_encoding_frame, add_paraphrase_frame, add_role_frame
from jailtime.datasets import PromptDataset
from jailtime.schemas import AttackContext, AttackPrompt, PromptCategory


class DirectBenignProbeStrategy:
    """Samples normal prompts to measure helpfulness and over-refusal."""

    id = "direct_benign_probe"
    name = "Direct benign probes"
    supported_categories = (PromptCategory.BENIGN,)

    def __init__(self, dataset: PromptDataset | None = None) -> None:
        self.dataset = dataset or PromptDataset()

    def generate(self, context: AttackContext, rng: Random) -> AttackPrompt:
        prompt = self.dataset.sample(PromptCategory.BENIGN, rng)
        return AttackPrompt(prompt=prompt, category=PromptCategory.BENIGN, strategy_id=self.id)


class DirectToyAdversarialStrategy:
    """Uses abstract toy unsafe markers for safety refusal testing."""

    id = "direct_toy_adversarial"
    name = "Direct toy adversarial probes"
    supported_categories = (PromptCategory.ADVERSARIAL,)

    def __init__(self, dataset: PromptDataset | None = None) -> None:
        self.dataset = dataset or PromptDataset()

    def generate(self, context: AttackContext, rng: Random) -> AttackPrompt:
        prompt = self.dataset.sample(PromptCategory.ADVERSARIAL, rng)
        return AttackPrompt(prompt=prompt, category=PromptCategory.ADVERSARIAL, strategy_id=self.id)


class ParaphraseToyMutationStrategy:
    """Applies a harmless paraphrase wrapper to toy unsafe markers."""

    id = "paraphrase_toy_mutation"
    name = "Paraphrase mutation of toy prompts"
    supported_categories = (PromptCategory.ADVERSARIAL,)

    def __init__(self, dataset: PromptDataset | None = None) -> None:
        self.dataset = dataset or PromptDataset()

    def generate(self, context: AttackContext, rng: Random) -> AttackPrompt:
        prompt = add_paraphrase_frame(self.dataset.sample(PromptCategory.ADVERSARIAL, rng))
        return AttackPrompt(prompt=prompt, category=PromptCategory.ADVERSARIAL, strategy_id=self.id)


class EncodingToyMutationStrategy:
    """Applies an abstract encoding wrapper to harmless toy markers."""

    id = "encoding_toy_mutation"
    name = "Encoding mutation of harmless strings"
    supported_categories = (PromptCategory.ADVERSARIAL,)

    def __init__(self, dataset: PromptDataset | None = None) -> None:
        self.dataset = dataset or PromptDataset()

    def generate(self, context: AttackContext, rng: Random) -> AttackPrompt:
        prompt = add_encoding_frame(self.dataset.sample(PromptCategory.ADVERSARIAL, rng))
        return AttackPrompt(prompt=prompt, category=PromptCategory.ADVERSARIAL, strategy_id=self.id)


class RoleFrameToyStrategy:
    """Uses non-harmful role framing around toy prompts."""

    id = "role_frame_toy"
    name = "Role framing with non-harmful toy examples"
    supported_categories = (PromptCategory.ADVERSARIAL,)

    def __init__(self, dataset: PromptDataset | None = None) -> None:
        self.dataset = dataset or PromptDataset()

    def generate(self, context: AttackContext, rng: Random) -> AttackPrompt:
        prompt = add_role_frame(self.dataset.sample(PromptCategory.ADVERSARIAL, rng))
        return AttackPrompt(prompt=prompt, category=PromptCategory.ADVERSARIAL, strategy_id=self.id)


class BoundaryCategoryProbeStrategy:
    """Samples safe educational borderline prompts."""

    id = "boundary_category_probe"
    name = "Boundary-category safe educational probes"
    supported_categories = (PromptCategory.BORDERLINE,)

    def __init__(self, dataset: PromptDataset | None = None) -> None:
        self.dataset = dataset or PromptDataset()

    def generate(self, context: AttackContext, rng: Random) -> AttackPrompt:
        prompt = self.dataset.sample(PromptCategory.BORDERLINE, rng)
        return AttackPrompt(prompt=prompt, category=PromptCategory.BORDERLINE, strategy_id=self.id)

