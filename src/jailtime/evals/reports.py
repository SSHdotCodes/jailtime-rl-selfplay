"""Markdown report rendering."""

from __future__ import annotations

from typing import Any


def render_markdown_report(summary: dict[str, Any], *, run_name: str) -> str:
    """Render a compact markdown report from summary metrics."""

    lines = [
        f"# jailtime report: {run_name}",
        "",
        "## Summary",
        "",
        f"- Episodes: {summary.get('episodes', 0)}",
        f"- Attack success rate: {_percent(summary.get('attack_success_rate', 0.0))}",
        f"- Safe refusal rate: {_percent(summary.get('safe_refusal_rate', 0.0))}",
        f"- Unsafe compliance rate: {_percent(summary.get('unsafe_compliance_rate', 0.0))}",
        f"- Benign helpfulness rate: {_percent(summary.get('benign_helpfulness_rate', 0.0))}",
        f"- Over-refusal rate: {_percent(summary.get('over_refusal_rate', 0.0))}",
        "- Ambiguous verification rate: "
        f"{_percent(summary.get('ambiguous_verification_rate', 0.0))}",
        f"- Average attacker reward: {summary.get('average_attacker_reward', 0.0):.3f}",
        f"- Average defender reward: {summary.get('average_defender_reward', 0.0):.3f}",
        "",
        "## Per Strategy",
        "",
        "| Strategy | Episodes | Attack Success | Avg Attacker Reward | Avg Defender Reward |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    per_strategy = summary.get("per_strategy", {})
    if not per_strategy:
        lines.append("| none | 0 | 0.0% | 0.000 | 0.000 |")
    for strategy_id, metrics in per_strategy.items():
        lines.append(
            "| "
            f"{strategy_id} | "
            f"{metrics.get('episodes', 0)} | "
            f"{_percent(metrics.get('attack_success_rate', 0.0))} | "
            f"{metrics.get('average_attacker_reward', 0.0):.3f} | "
            f"{metrics.get('average_defender_reward', 0.0):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "Use held-out evaluations and calibrated verifiers before drawing safety conclusions.",
        ]
    )
    return "\n".join(lines) + "\n"


def _percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"
