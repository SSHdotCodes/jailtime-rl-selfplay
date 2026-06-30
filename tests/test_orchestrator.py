from jailtime.config import JailtimeConfig
from jailtime.orchestrator import Orchestrator


def test_orchestrator_runs_five_demo_episodes(tmp_path) -> None:
    config = JailtimeConfig()
    config.run.output_dir = str(tmp_path)
    config.run.seed = 123

    report = Orchestrator(config).run(num_episodes=5)

    assert len(report.episodes) == 5
    assert report.summary["episodes"] == 5
    assert (tmp_path / "latest" / "episodes.jsonl").exists()
    assert (tmp_path / "latest" / "summary.json").exists()
    assert (tmp_path / "latest" / "report.md").exists()

