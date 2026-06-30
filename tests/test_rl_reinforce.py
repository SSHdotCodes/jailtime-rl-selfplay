from jailtime.rl.policy import MockPolicyModel, SampledRollout
from jailtime.rl.reinforce import REINFORCETrainer


def _rollout(n_tokens: int = 3) -> SampledRollout:
    return SampledRollout(
        prompt_text="p",
        response_text="r",
        prompt_token_ids=[1, 2],
        response_token_ids=list(range(100, 100 + n_tokens)),
    )


def test_reinforce_step_calls_policy_with_centered_advantage() -> None:
    policy = MockPolicyModel()
    trainer = REINFORCETrainer(policy, lr=1e-4, entropy_coef=0.0, baseline_decay=0.9, name="t")

    trainer.step(_rollout(), reward=2.0)
    # baseline starts at 0, so advantage == reward on the first step
    assert policy.updates[0]["advantage"] == 2.0
    assert policy.updates[0]["lr"] == 1e-4
    assert policy.updates[0]["num_tokens"] == 3.0


def test_reinforce_baseline_ema_centers_future_advantages() -> None:
    policy = MockPolicyModel()
    trainer = REINFORCETrainer(policy, lr=1e-4, entropy_coef=0.0, baseline_decay=0.5)

    trainer.step(_rollout(), reward=4.0)  # advantage=4-0=4, baseline -> 0.5*0 + 0.5*4 = 2.0
    trainer.step(_rollout(), reward=4.0)  # advantage=4-2=2, baseline -> 0.5*2 + 0.5*4 = 3.0
    assert trainer._baseline == 3.0
    assert policy.updates[0]["advantage"] == 4.0
    assert policy.updates[1]["advantage"] == 2.0


def test_reinforce_skips_empty_rollout_without_calling_policy() -> None:
    policy = MockPolicyModel()
    trainer = REINFORCETrainer(policy, lr=1e-4, entropy_coef=0.0)

    stats = trainer.step(_rollout(n_tokens=0), reward=1.0)
    assert stats["skipped"] == 1.0
    assert policy.updates == []
    assert trainer.stats()["skipped"] == 1


def test_reinforce_clips_extreme_rewards() -> None:
    policy = MockPolicyModel()
    trainer = REINFORCETrainer(policy, lr=1e-4, entropy_coef=0.0, reward_clip=5.0)

    stats = trainer.step(_rollout(), reward=1000.0)
    assert stats["reward"] == 5.0
    assert stats["advantage"] == 5.0


def test_reinforce_negative_reward_propagates_negative_advantage() -> None:
    policy = MockPolicyModel()
    trainer = REINFORCETrainer(policy, lr=1e-4, entropy_coef=0.0)

    stats = trainer.step(_rollout(), reward=-3.0)
    assert stats["advantage"] == -3.0
    assert policy.updates[0]["advantage"] == -3.0


def test_reinforce_stats_accumulate_across_steps() -> None:
    policy = MockPolicyModel()
    trainer = REINFORCETrainer(policy, lr=1e-4, entropy_coef=0.0, name="defender")

    trainer.step(_rollout(), reward=1.0)
    trainer.step(_rollout(), reward=-1.0)

    stats = trainer.stats()
    assert stats["name"] == "defender"
    assert stats["steps"] == 2
    assert stats["mean_reward"] == 0.0


def test_reinforce_rejects_invalid_hyperparameters() -> None:
    policy = MockPolicyModel()
    import pytest

    with pytest.raises(ValueError):
        REINFORCETrainer(policy, lr=0.0)
    with pytest.raises(ValueError):
        REINFORCETrainer(policy, baseline_decay=1.0)
