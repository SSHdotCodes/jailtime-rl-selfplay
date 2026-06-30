from jailtime.rl.agents import RLModelVerifier
from jailtime.rl.policy import MockPolicyModel
from jailtime.rl.verifier_calibration import (
    VerifierCalibrationTrainer,
    default_calibration_probes,
)
from jailtime.schemas import OutcomeLabel, PromptCategory


class JSONMockPolicy(MockPolicyModel):
    """Mock policy that returns a fixed verifier JSON judgment."""

    def __init__(self, outcome: OutcomeLabel, category: PromptCategory) -> None:
        super().__init__(response_text="x")
        self._outcome = outcome
        self._category = category

    def sample(self, messages, *, max_new_tokens, temperature):
        import json

        from jailtime.rl.policy import SampledRollout

        payload = {
            "prompt_category": self._category.value,
            "outcome": self._outcome.value,
            "safety_score": 1.0,
            "helpfulness_score": 0.5,
            "confidence": 0.9,
            "rationale": "mock",
        }
        return SampledRollout(
            prompt_text="",
            response_text=json.dumps(payload),
            prompt_token_ids=[1, 2],
            response_token_ids=[3, 4],
            metadata={"backend": "mock"},
        )


def test_calibration_rewards_correct_judgment() -> None:
    probes = default_calibration_probes()
    first = probes[0]
    policy = JSONMockPolicy(first.expected_outcome, first.expected_category)
    verifier = RLModelVerifier(policy)
    trainer = VerifierCalibrationTrainer(verifier, probes, lr=1e-4)

    stats = trainer.calibrate()
    # At least the first probe matched its expected label.
    assert stats["accuracy"] > 0.0
    assert stats["num_probes"] == len(probes)
    assert policy.updates  # REINFORCE steps were applied


def test_calibration_penalty_for_wrong_judgment() -> None:
    probes = default_calibration_probes()
    policy = JSONMockPolicy(OutcomeLabel.AMBIGUOUS, PromptCategory.UNKNOWN)
    verifier = RLModelVerifier(policy)
    trainer = VerifierCalibrationTrainer(verifier, probes, lr=1e-4)

    stats = trainer.calibrate()
    # No probe expects AMBIGUOUS+UNKNOWN, so accuracy is 0 and rewards are negative.
    assert stats["accuracy"] == 0.0
    assert stats["mean_reward"] < 0.0
    assert len(policy.updates) == len(probes)


def test_calibration_requires_probes() -> None:
    import pytest

    verifier = RLModelVerifier(MockPolicyModel())
    with pytest.raises(ValueError):
        VerifierCalibrationTrainer(verifier, [])
