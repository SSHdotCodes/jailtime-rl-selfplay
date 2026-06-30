from typer.testing import CliRunner

from jailtime.cli import app


def test_cli_demo_command_exits_successfully(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(app, ["demo", "--episodes", "3"])

    assert result.exit_code == 0
    assert "Run artifacts" in result.output
    assert (tmp_path / "runs" / "latest" / "summary.json").exists()


def test_validate_config_catches_invalid_sampling_rates(tmp_path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text(
        """
sampling:
  adversarial_rate: 0.8
  benign_rate: 0.8
  borderline_rate: 0.1
""",
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(app, ["validate-config", str(config)])

    assert result.exit_code != 0
    assert "sampling rates must sum to 1.0" in result.output


_MIN_RL_SELPLAY_YAML = """
run:
  name: "cli-rl-test"
  episodes: 3
  output_dir: "./runs"
providers:
  defender:
    type: "mock"
  verifier:
    type: "rule_based"
trainer:
  type: "rl_selfplay"
rl:
  enabled: true
  save_final_defender: true
  final_defender_dir: "defender_final"
  verifier_calibration:
    enabled: false
"""


def _patch_selfplay_with_fake(monkeypatch, tmp_path) -> None:
    from jailtime.rl import selfplay as selfplay_mod

    class FakeReport:
        summary = {
            "episodes": 3,
            "rl_defender_stats": {
                "steps": 3,
                "mean_reward": 0.1,
                "mean_abs_advantage": 0.5,
            },
        }
        output_dir = str(tmp_path / "runs" / "latest")

    class FakeOrchestrator:
        def __init__(self, config) -> None:
            self.config = config

        def run(self, num_episodes: int | None = None) -> FakeReport:
            return FakeReport()

    monkeypatch.setattr(selfplay_mod, "RLSelfPlayOrchestrator", FakeOrchestrator)


def test_cli_selfplay_command_prints_hardened_defender_path(tmp_path, monkeypatch) -> None:
    _patch_selfplay_with_fake(monkeypatch, tmp_path)
    config = tmp_path / "rl.yaml"
    config.write_text(_MIN_RL_SELPLAY_YAML, encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(app, ["selfplay", "--config", str(config), "--episodes", "3"])

    assert result.exit_code == 0
    assert "Hardened defender weights" in result.output
    assert "Defender RL steps" in result.output


def test_cli_run_dispatches_to_selfplay_for_rl_selfplay_config(tmp_path, monkeypatch) -> None:
    _patch_selfplay_with_fake(monkeypatch, tmp_path)
    config = tmp_path / "rl.yaml"
    config.write_text(_MIN_RL_SELPLAY_YAML, encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(app, ["run", "--config", str(config), "--episodes", "3"])

    assert result.exit_code == 0
    assert "Hardened defender weights" in result.output

