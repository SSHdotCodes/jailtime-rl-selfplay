#!/usr/bin/env python3
"""Run a visible real-time RL self-play pass and save the hardened defender.

This replaces the older LoRA-SFT driver (work/gemma4_jailtime_full_run.py).
Instead of collecting SFT pairs and fine-tuning a LoRA adapter offline, this
driver runs ``RLSelfPlayOrchestrator`` which applies full-parameter REINFORCE
updates to the attacker and defender weights *every episode*. The verifier
judges each defender response, the reward calculator turns the judgment into
scalar rewards, and both policies are updated in real time. When the loop
finishes the defender's final weights are saved as safetensors to
``<run_dir>/latest/defender_final``.

Each episode is logged verbosely: the prompt category, technique, the
verifier verdict, who got rewarded or punished and why, the reward and
advantage values, and excerpts of the attacker probe and defender response.

Use only on models you own or are explicitly authorized to modify.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from jailtime.config import load_config
from jailtime.evals.metrics import compute_metrics
from jailtime.rl.selfplay import RLSelfPlayOrchestrator
from jailtime.schemas import OutcomeLabel, PromptCategory

console = Console()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="examples/config_rl_selfplay.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print running metrics every 50 episodes (skip per-episode logs).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config.run.seed = args.seed
    total = args.episodes or config.run.episodes
    run_dir = Path(config.run.output_dir) / "latest"
    run_dir.mkdir(parents=True, exist_ok=True)

    console.rule("[bold]Jailtime real-time RL self-play")
    console.print(f"Run dir: {run_dir}")
    console.print(f"Episodes: {total}")
    console.print(f"Attacker: {config.providers.attacker.model}")
    console.print(f"Defender: {config.providers.defender.model}")
    console.print(f"Main verifier: {config.providers.verifier.type}")
    console.print(
        f"Attacker RL: optimizer={config.rl.attacker.optimizer} lr={config.rl.attacker.lr} "
        f"dtype={config.rl.attacker.dtype}"
    )
    console.print(
        f"Defender RL: optimizer={config.rl.defender.optimizer} lr={config.rl.defender.lr} "
        f"dtype={config.rl.defender.dtype}"
    )
    console.print("Updates: full-parameter REINFORCE every episode (no LoRA).")
    console.print(
        "Reward: defender + for stopping jailbreaks, - for getting jailbroken, "
        "- for over-refusal, slight + for safe helpful answers."
    )
    console.print(f"Verbose per-episode log: {('OFF' if args.quiet else 'ON')}")

    orchestrator = RLSelfPlayOrchestrator(config)

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    episodes: list[Any] = []

    def on_episode(episode: Any, a_stats: dict[str, float], d_stats: dict[str, float]) -> None:
        episodes.append(episode)
        completed = len(episodes)
        if not args.quiet:
            _log_episode(console, episode, a_stats, d_stats)
        if completed % 50 == 0:
            summary = compute_metrics(episodes)
            console.print(
                f"[bold cyan]-- running @ {completed}/{total}[/] "
                f"attack_success={summary['attack_success_rate']:.3f} "
                f"safe_refusal={summary['safe_refusal_rate']:.3f} "
                f"benign_help={summary['benign_helpfulness_rate']:.3f} "
                f"over_refusal={summary['over_refusal_rate']:.3f} "
                f"d_mean_reward={orchestrator.defender_rl.stats()['mean_reward']:.3f}"
            )
        progress.advance(task)

    with progress:
        task = progress.add_task("episodes", total=total)
        report = orchestrator.run(num_episodes=total, on_episode=on_episode)

    final_dir = Path(report.output_dir) / config.rl.final_defender_dir
    console.print(f"[green]Saved hardened defender safetensors:[/] {final_dir}")
    _print_summary(console, report.summary)
    return 0


def _log_episode(
    console: Console,
    episode: Any,
    a_stats: dict[str, float],
    d_stats: dict[str, float],
) -> None:
    idx = episode.episode_index
    category = _category_short(episode.prompt_category)
    technique = episode.technique_id or "-"
    outcome = str(episode.verification.outcome)
    rationale = (episode.verification.rationale or "").strip()

    d_reward = float(episode.defender_reward)
    a_reward = float(episode.attacker_reward)
    d_reason = _defender_reason(episode.prompt_category, episode.verification.outcome)
    a_reason = _attacker_reason(episode.prompt_category, episode.verification.outcome)
    d_verdict = _verdict_label(d_reward)
    a_verdict = _verdict_label(a_reward)

    probe = _truncate(episode.prompt, 110)
    resp = _truncate(episode.response, 110)

    console.print(
        f"[bold]ep#{idx:04d}[/] {category} {technique} | verdict=[magenta]{outcome}[/]"
    )
    console.print(
        f"  attacker [bold]{a_verdict.label}[/] {a_verdict.sign}{a_reward:+.2f} "
        f"({a_reason}) adv={a_stats.get('advantage', 0):+.3f}"
    )
    console.print(
        f"  defender [bold]{d_verdict.label}[/] {d_verdict.sign}{d_reward:+.2f} "
        f"({d_reason}) adv={d_stats.get('advantage', 0):+.3f}"
    )
    console.print(f"  probe: [dim]{probe}[/]")
    console.print(f"  resp : [dim]{resp}[/]")
    if rationale:
        console.print(f"  why  : [dim]{_truncate(rationale, 140)}[/]")


def _defender_reason(category: Any, outcome: Any) -> str:
    cat = PromptCategory(str(category)) if not isinstance(category, PromptCategory) else category
    out = OutcomeLabel(str(outcome)) if not isinstance(outcome, OutcomeLabel) else outcome
    if cat == PromptCategory.ADVERSARIAL:
        if out == OutcomeLabel.SAFE_REFUSAL:
            return "stopped a real jailbreak"
        if out == OutcomeLabel.UNSAFE_COMPLIANCE:
            return "got jailbroken (unsafe compliance on adversarial)"
        if out == OutcomeLabel.EVASIVE:
            return "evasive instead of refusing an adversarial probe"
        return "ambiguous handling of an adversarial probe"
    if cat == PromptCategory.BENIGN:
        if out == OutcomeLabel.HELPFUL_COMPLIANCE:
            return "correctly answered a safe prompt"
        if out == OutcomeLabel.OVER_REFUSAL:
            return "over-refused a safe prompt (false alarm)"
        if out == OutcomeLabel.SAFE_REFUSAL:
            return "refused a safe prompt (over-refusal)"
        if out == OutcomeLabel.EVASIVE:
            return "evasive on a safe prompt"
        return "ambiguous handling of a safe prompt"
    if cat == PromptCategory.BORDERLINE:
        if out in {OutcomeLabel.SAFE_REFUSAL, OutcomeLabel.HELPFUL_COMPLIANCE}:
            return "handled a borderline prompt safely"
        if out == OutcomeLabel.UNSAFE_COMPLIANCE:
            return "got jailbroken on a borderline prompt"
        return "ambiguous handling of a borderline prompt"
    return "ambiguous handling of an unknown prompt"


def _attacker_reason(category: Any, outcome: Any) -> str:
    cat = PromptCategory(str(category)) if not isinstance(category, PromptCategory) else category
    out = OutcomeLabel(str(outcome)) if not isinstance(outcome, OutcomeLabel) else outcome
    if cat != PromptCategory.ADVERSARIAL:
        return f"{cat.value.lower()} probe dispatched (no attacker reward)"
    if out == OutcomeLabel.UNSAFE_COMPLIANCE:
        return "jailbreak succeeded"
    if out == OutcomeLabel.EVASIVE:
        return "partial credit (defender evasive, closer to compliance)"
    if out == OutcomeLabel.SAFE_REFUSAL:
        return "jailbreak failed (defender refused)"
    if out == OutcomeLabel.AMBIGUOUS:
        return "ambiguous outcome"
    return "jailbreak failed"


class _Verdict:
    def __init__(self, reward: float) -> None:
        self.label: str
        self.sign: str
        if reward > 1e-9:
            self.label = "REWARDED"
            self.sign = "+"
        elif reward < -1e-9:
            self.label = "PUNISHED"
            self.sign = ""
        else:
            self.label = "NEUTRAL"
            self.sign = ""


def _verdict_label(reward: float) -> _Verdict:
    return _Verdict(reward)


def _category_short(category: Any) -> str:
    val = getattr(category, "value", str(category))
    return val[:4].upper()


def _truncate(text: str, limit: int) -> str:
    flat = " ".join(str(text).split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1] + "\u2026"


def _print_summary(console: Console, summary: dict[str, Any]) -> None:
    table = Table(title="Final RL self-play summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    for key, value in summary.items():
        if isinstance(value, dict):
            continue
        table.add_row(key, f"{value:.4f}" if isinstance(value, float) else str(value))
    console.print(table)

    for role in ("rl_attacker_stats", "rl_defender_stats", "rl_verifier_stats"):
        stats = summary.get(role)
        if not stats:
            continue
        sub = Table(title=role)
        sub.add_column("Metric", style="cyan")
        sub.add_column("Value", justify="right")
        for key, value in stats.items():
            if isinstance(value, dict):
                continue
            sub.add_row(key, f"{value:.4f}" if isinstance(value, float) else str(value))
        console.print(sub)


if __name__ == "__main__":
    sys.exit(main())