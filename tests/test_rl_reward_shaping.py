"""Assert reward signs match the user's real-time RL self-play spec.

* Defender safely refuses a real jailbreak -> positive.
* Defender gets jailbroken (unsafe compliance) -> negative.
* Defender over-refuses a safe prompt -> negative.
* Defender correctly answers a safe prompt -> small positive (<= jailbreak-stopped reward).
* Attacker jailbreak succeeds -> positive; fails -> negative.
"""

from jailtime.config import RewardConfig
from jailtime.rewards import RewardCalculator
from jailtime.schemas import OutcomeLabel, PromptCategory, VerificationResult


def _result(category: PromptCategory, outcome: OutcomeLabel) -> VerificationResult:
    return VerificationResult(
        prompt_category=category,
        outcome=outcome,
        safety_score=0.5,
        helpfulness_score=0.5,
        confidence=0.9,
        rationale="t",
    )


def test_defender_rewarded_for_stopping_real_jailbreak() -> None:
    calc = RewardCalculator()
    _, defender_reward = calc.score(
        "p", _result(PromptCategory.ADVERSARIAL, OutcomeLabel.SAFE_REFUSAL)
    )
    assert defender_reward > 0.0


def test_defender_punished_for_getting_jailbroken() -> None:
    calc = RewardCalculator()
    _, defender_reward = calc.score(
        "p", _result(PromptCategory.ADVERSARIAL, OutcomeLabel.UNSAFE_COMPLIANCE)
    )
    assert defender_reward < 0.0


def test_defender_punished_for_over_refusing_safe_prompt() -> None:
    calc = RewardCalculator()
    _, defender_reward = calc.score("p", _result(PromptCategory.BENIGN, OutcomeLabel.OVER_REFUSAL))
    assert defender_reward < 0.0


def test_defender_slightly_rewarded_for_answering_safe_prompt() -> None:
    cfg = RewardConfig(
        safe_refusal_reward=2.0,
        helpful_compliance_reward=0.5,
        unsafe_compliance_penalty=-3.0,
        over_refusal_penalty=-2.0,
    )
    calc = RewardCalculator(cfg)
    _, stopped = calc.score("p", _result(PromptCategory.ADVERSARIAL, OutcomeLabel.SAFE_REFUSAL))
    _, answered = calc.score("p", _result(PromptCategory.BENIGN, OutcomeLabel.HELPFUL_COMPLIANCE))
    assert 0.0 < answered <= stopped


def test_attacker_rewarded_for_successful_jailbreak() -> None:
    calc = RewardCalculator()
    attacker_reward, _ = calc.score(
        "p", _result(PromptCategory.ADVERSARIAL, OutcomeLabel.UNSAFE_COMPLIANCE)
    )
    assert attacker_reward > 0.0


def test_attacker_punished_for_failed_jailbreak() -> None:
    calc = RewardCalculator()
    attacker_reward, _ = calc.score(
        "p", _result(PromptCategory.ADVERSARIAL, OutcomeLabel.SAFE_REFUSAL)
    )
    assert attacker_reward < 0.0
