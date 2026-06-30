from jailtime.config import JailtimeConfig, ProviderConfig, RewardConfig
from jailtime.rl.policy import MockPolicyModel
from jailtime.rl.selfplay import RLSelfPlayOrchestrator


def _config(tmp_path) -> JailtimeConfig:
    config = JailtimeConfig()
    config.run.output_dir = str(tmp_path)
    config.run.name = "rl-selfplay-test"
    config.run.seed = 7
    config.run.episodes = 5
    config.sampling.adversarial_rate = 0.6
    config.sampling.benign_rate = 0.4
    config.sampling.borderline_rate = 0.0
    config.providers.defender = ProviderConfig(type="mock")
    config.providers.verifier = ProviderConfig(type="rule_based")
    config.trainer.type = "rl_selfplay"
    config.rl.enabled = True
    config.rl.checkpoint_every = 0  # disable mid-run checkpoints in tests
    config.rl.save_final_defender = True
    config.rl.final_defender_dir = "defender_final"
    config.rl.verifier_calibration.enabled = True
    config.rl.verifier_calibration.every = 3
    config.rewards = RewardConfig(
        safe_refusal_reward=2.0,
        unsafe_compliance_penalty=-3.0,
        over_refusal_penalty=-2.0,
        helpful_compliance_reward=0.5,
        successful_attack_reward=1.0,
        failed_attack_penalty=-0.5,
    )
    return config


def test_rl_selfplay_runs_and_updates_weights_each_episode(tmp_path) -> None:
    config = _config(tmp_path)
    attacker_policy = MockPolicyModel(name="attacker", response_text="mock adversarial probe")
    defender_policy = MockPolicyModel(name="defender", response_text="mock response")
    verifier_policy = MockPolicyModel(name="verifier", response_text="mock verifier")

    orchestrator = RLSelfPlayOrchestrator(
        config,
        attacker_policy=attacker_policy,
        defender_policy=defender_policy,
        verifier_policy=verifier_policy,
    )

    report = orchestrator.run(num_episodes=5)

    assert len(report.episodes) == 5
    # Every episode produced a real REINFORCE update on both policies.
    assert len(attacker_policy.updates) == 5
    assert len(defender_policy.updates) == 5
    # Final defender safetensors-equivalent was saved.
    final_dir = tmp_path / "latest" / "defender_final"
    assert final_dir.exists()
    assert (final_dir / "mock_policy_saved.json").exists()


def test_rl_selfplay_summary_contains_rl_stats(tmp_path) -> None:
    config = _config(tmp_path)
    orchestrator = RLSelfPlayOrchestrator(
        config,
        attacker_policy=MockPolicyModel(response_text="probe"),
        defender_policy=MockPolicyModel(response_text="response"),
        verifier_policy=MockPolicyModel(response_text="verifier"),
    )
    report = orchestrator.run(num_episodes=4)

    assert "rl_attacker_stats" in report.summary
    assert "rl_defender_stats" in report.summary
    assert "rl_verifier_stats" in report.summary
    assert report.summary["rl_defender_stats"]["steps"] == 4
    # Verifier calibration ran at least once (every=3, 4 episodes).
    assert report.summary["rl_verifier_stats"]["rounds"] >= 1


def test_rl_selfplay_requires_local_transformers_provider(tmp_path) -> None:
    config = _config(tmp_path)
    # No injected policies -> orchestrator tries to build torch policies from
    # provider config. Defender provider is "mock", which is not allowed.
    import pytest

    with pytest.raises(ValueError, match="local_transformers"):
        RLSelfPlayOrchestrator(config)


def test_rl_selfplay_supports_hybrid_verifier_via_base_construction(tmp_path) -> None:
    config = _config(tmp_path)
    config.providers.verifier = ProviderConfig(
        type="hybrid",
        params={"provider": {"type": "mock", "params": {"mode": "safe"}}},
    )
    config.rl.verifier_calibration.enabled = False

    orchestrator = RLSelfPlayOrchestrator(
        config,
        attacker_policy=MockPolicyModel(response_text="probe"),
        defender_policy=MockPolicyModel(response_text="response"),
    )
    # Base Orchestrator built the hybrid verifier (no AttributeError on the
    # provider cache) and the loop still runs end-to-end.
    report = orchestrator.run(num_episodes=3)
    assert len(report.episodes) == 3
    assert "rl_defender_stats" in report.summary
