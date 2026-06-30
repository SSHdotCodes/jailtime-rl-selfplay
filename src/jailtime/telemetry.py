"""Telemetry recording for jailtime runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jailtime.evals.metrics import compute_metrics
from jailtime.evals.reports import render_markdown_report
from jailtime.schemas import EpisodeResult


class TelemetryRecorder:
    """Writes episode logs, summary JSON, and markdown reports."""

    def __init__(self, run_dir: str | Path, *, run_name: str) -> None:
        self.run_dir = Path(run_dir)
        self.run_name = run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_path = self.run_dir / "episodes.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.report_path = self.run_dir / "report.md"
        self.episodes_path.write_text("", encoding="utf-8")
        self._episodes: list[EpisodeResult] = []

    def record_episode(self, episode: EpisodeResult) -> None:
        """Append an episode to in-memory and JSONL telemetry."""

        self._episodes.append(episode)
        with self.episodes_path.open("a", encoding="utf-8") as handle:
            handle.write(episode.model_dump_json() + "\n")

    def finalize(self) -> dict[str, Any]:
        """Write summary artifacts and return computed metrics."""

        summary = compute_metrics(self._episodes)
        summary["run_name"] = self.run_name
        summary["output_dir"] = str(self.run_dir)
        self.summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.report_path.write_text(
            render_markdown_report(summary, run_name=self.run_name),
            encoding="utf-8",
        )
        return summary


def load_summary(run_dir: str | Path) -> dict[str, Any]:
    """Load ``summary.json`` from a run directory."""

    summary_path = Path(run_dir) / "summary.json"
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Run summary not found: {summary_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Run summary is not valid JSON: {summary_path}") from exc

