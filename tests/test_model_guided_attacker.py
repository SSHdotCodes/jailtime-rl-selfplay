from typing import Any

from jailtime.attacks.model_guided import (
    ModelGuidedToyAdversarialStrategy,
    ModelWrappedToyAdversarialStrategy,
)
from jailtime.schemas import AttackContext, Message, ModelResponse, PromptCategory


class FakeChoiceProvider:
    def complete(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        return ModelResponse(content="encoding")


def test_model_guided_attacker_uses_safe_toy_mutations() -> None:
    strategy = ModelGuidedToyAdversarialStrategy(FakeChoiceProvider())
    context = AttackContext(
        episode_index=1,
        prompt_category=PromptCategory.ADVERSARIAL,
    )

    prompt = strategy.generate(context, __import__("random").Random(1))

    assert prompt.category == PromptCategory.ADVERSARIAL
    assert prompt.strategy_id == "model_guided_toy_adversarial"
    assert "[UNSAFE_TOY]" in prompt.prompt
    assert "Encoded toy marker evaluation" in prompt.prompt


class FakeWrapperProvider:
    def complete(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        return ModelResponse(
            content='{"prefix": "### weird mock system frame", "suffix": "obey the toy marker"}'
        )


def test_model_wrapped_attacker_uses_model_wrapper_with_toy_target() -> None:
    strategy = ModelWrappedToyAdversarialStrategy(FakeWrapperProvider())
    context = AttackContext(
        episode_index=2,
        prompt_category=PromptCategory.ADVERSARIAL,
    )

    prompt = strategy.generate(context, __import__("random").Random(1))

    assert prompt.category == PromptCategory.ADVERSARIAL
    assert prompt.strategy_id == "model_wrapped_toy_adversarial"
    assert "### weird mock system frame" in prompt.prompt
    assert "[UNSAFE_TOY]" in prompt.prompt
    assert "obey the toy marker" in prompt.prompt
