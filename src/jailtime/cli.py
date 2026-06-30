"""Typer CLI for jailtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from jailtime.config import JailtimeConfig, load_config
from jailtime.orchestrator import Orchestrator
from jailtime.providers.device import probe_torch_devices
from jailtime.telemetry import load_summary

app = typer.Typer(help="Defensive AI safety research loop for authorized testing.")
console = Console()


@app.command()
def run(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to a jailtime YAML config."),
    ] = Path("examples/config.yaml"),
    episodes: Annotated[
        int | None,
        typer.Option("--episodes", "-n", help="Override configured episode count."),
    ] = None,
) -> None:
    """Run a configured jailtime evaluation."""

    jailtime_config = _load_config_or_exit(config)
    if jailtime_config.trainer.type == "rl_selfplay":
        _run_rl_selfplay(jailtime_config, episodes)
        return
    report = Orchestrator(jailtime_config).run(num_episodes=episodes)
    _print_summary(report.summary, output_dir=report.output_dir)


@app.command()
def selfplay(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to a jailtime RL self-play YAML config."),
    ] = Path("examples/config_rl_selfplay.yaml"),
    episodes: Annotated[
        int | None,
        typer.Option("--episodes", "-n", help="Override configured episode count."),
    ] = None,
) -> None:
    """Run real-time RL self-play and save the hardened defender safetensors."""

    jailtime_config = _load_config_or_exit(config)
    _run_rl_selfplay(jailtime_config, episodes)


def _run_rl_selfplay(jailtime_config: JailtimeConfig, episodes: int | None) -> None:
    from jailtime.rl.selfplay import RLSelfPlayOrchestrator

    if not jailtime_config.rl.enabled:
        jailtime_config.rl.enabled = True
    jailtime_config.trainer.type = "rl_selfplay"
    report = RLSelfPlayOrchestrator(jailtime_config).run(num_episodes=episodes)
    _print_summary(report.summary, output_dir=report.output_dir)
    defender_dir = Path(report.output_dir) / jailtime_config.rl.final_defender_dir
    console.print(f"[green]Hardened defender weights:[/] {defender_dir}")
    stats = report.summary.get("rl_defender_stats", {})
    if stats:
        console.print(
            f"[cyan]Defender RL steps:[/] {stats.get('steps')} "
            f"mean_reward={stats.get('mean_reward', 0):.3f} "
            f"mean_abs_advantage={stats.get('mean_abs_advantage', 0):.3f}"
        )


@app.command()
def demo(
    episodes: Annotated[
        int,
        typer.Option("--episodes", "-n", min=1, help="Number of local mock episodes."),
    ] = 10,
) -> None:
    """Run a fully local mock demo."""

    config = JailtimeConfig()
    config.run.name = "demo-run"
    config.run.episodes = episodes
    config.providers.defender.type = "mock"
    config.providers.defender.params = {"mode": "mixed"}
    config.providers.verifier.type = "rule_based"
    report = Orchestrator(config).run(num_episodes=episodes)
    _print_summary(report.summary, output_dir=report.output_dir)


@app.command("report")
def report_command(
    run_dir: Annotated[Path, typer.Argument(help="Run directory containing summary.json.")],
) -> None:
    """Print a summary for an existing run directory."""

    summary = load_summary(run_dir)
    _print_summary(summary, output_dir=str(run_dir))
    report_path = run_dir / "report.md"
    if report_path.exists():
        console.print(f"[green]Markdown report:[/] {report_path}")


@app.command("validate-config")
def validate_config(
    config: Annotated[
        Path,
        typer.Argument(help="Path to a jailtime YAML config."),
    ],
) -> None:
    """Validate a jailtime YAML config."""

    loaded = _load_config_or_exit(config)
    console.print(f"[green]Valid config:[/] {config}")
    console.print(json.dumps(loaded.model_dump(mode="json"), indent=2, sort_keys=True))


@app.command("devices")
def devices() -> None:
    """Show local PyTorch accelerator availability."""

    status = probe_torch_devices()
    table = Table(title="jailtime devices")
    table.add_column("Capability", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("torch installed", str(status.torch_installed))
    table.add_row("mps built", str(status.mps_built))
    table.add_row("mps available", str(status.mps_available))
    table.add_row("cuda available", str(status.cuda_available))
    table.add_row("selected device", status.selected_device or "none")
    console.print(table)
    console.print(status.reason)


def _print_summary(summary: dict[str, object], *, output_dir: str) -> None:
    table = Table(title="jailtime summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    for key in [
        "episodes",
        "attack_success_rate",
        "safe_refusal_rate",
        "unsafe_compliance_rate",
        "benign_helpfulness_rate",
        "over_refusal_rate",
        "ambiguous_verification_rate",
        "average_attacker_reward",
        "average_defender_reward",
    ]:
        value = summary.get(key, 0)
        if isinstance(value, float):
            table.add_row(key, f"{value:.3f}")
        else:
            table.add_row(key, str(value))
    console.print(table)
    console.print(f"[green]Run artifacts:[/] {output_dir}")


def _load_config_or_exit(config: Path) -> JailtimeConfig:
    try:
        return load_config(config)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Invalid config:[/] {exc}")
        raise typer.Exit(code=1) from exc


def main() -> None:
    """CLI entry point."""

    app()


if __name__ == "__main__":
    main()
