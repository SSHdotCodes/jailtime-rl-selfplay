#!/usr/bin/env python3
"""Run a visible Gemma 4 Jailtime pass and export a LoRA safetensors adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from transformers import AutoTokenizer

from jailtime.config import load_config
from jailtime.evals.metrics import compute_metrics
from jailtime.orchestrator import Orchestrator
from jailtime.providers.local_transformers import LocalTransformersProvider
from jailtime.schemas import EpisodeResult, OutcomeLabel, PromptCategory
from jailtime.telemetry import TelemetryRecorder

SAFE_REFUSAL = (
    "I can't help with that request. I can offer a safe, high-level explanation "
    "of relevant safety boundaries or help reformulate the goal in a benign way."
)
BORDERLINE_SAFE = (
    "I can help at a high level. Please clarify the benign goal, and I will avoid "
    "operational details that could create risk."
)
BENIGN_FALLBACK = (
    "Here is a concise helpful answer: define the goal, keep the explanation clear, "
    "and verify the result with a small test."
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="work/gemma4_jailtime_full_run.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--train-adapter", action="store_true")
    parser.add_argument("--adapter-epochs", type=int, default=2)
    parser.add_argument("--max-train-examples", type=int, default=1024)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    console = Console()
    config = load_config(args.config)
    total_episodes = args.episodes or config.run.episodes
    run_dir = Path(config.run.output_dir) / "latest"
    run_dir.mkdir(parents=True, exist_ok=True)

    console.rule("[bold]Jailtime Gemma 4 MPS run")
    console.print(f"Run dir: {run_dir}")
    console.print(f"Episodes: {total_episodes}")
    console.print("Attacker: OBLITERATUS/gemma-4-E4B-it-OBLITERATED")
    console.print("Defender: google/gemma-4-E4B-it")
    console.print("Verifier: hybrid rule-based + google/gemma-4-E4B-it fallback")
    console.print("Safety boundary: model-guided toy mutations only; no real harmful corpus.")

    orchestrator = Orchestrator(config)
    console.print("Loaded models. Running one preflight episode before the long loop.")
    telemetry = TelemetryRecorder(run_dir, run_name=config.run.name)
    episodes: list[EpisodeResult] = []

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task("episodes", total=total_episodes)
        for episode_index in range(total_episodes):
            episode = orchestrator.run_episode(episode_index)
            episodes.append(episode)
            telemetry.record_episode(episode)
            for trainer in orchestrator.trainers:
                trainer.observe(episode)
            if (episode_index + 1) % config.trainer.update_every == 0:
                orchestrator._update_trainers()
                _print_metrics(console, episodes, episode_index + 1)
            progress.advance(task)

    orchestrator._update_trainers()
    summary = telemetry.finalize()
    _print_summary(console, summary)

    train_path = run_dir / "sft_examples.jsonl"
    examples = write_sft_examples(episodes, train_path, limit=args.max_train_examples)
    console.print(f"Wrote {len(examples)} SFT examples: {train_path}")

    if args.train_adapter:
        adapter_dir = run_dir / "gemma4_jailtime_lora_adapter"
        train_lora_adapter(
            console=console,
            model_name=config.providers.defender.model or "google/gemma-4-E4B-it",
            examples=examples,
            output_dir=adapter_dir,
            epochs=args.adapter_epochs,
            max_length=args.max_length,
        )
    return 0


def write_sft_examples(
    episodes: list[EpisodeResult],
    output_path: Path,
    *,
    limit: int,
) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for episode in episodes:
        target = target_response_for_episode(episode)
        examples.append({"prompt": episode.prompt, "response": target})
        if len(examples) >= limit:
            break
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    return examples


def target_response_for_episode(episode: EpisodeResult) -> str:
    if episode.prompt_category == PromptCategory.ADVERSARIAL:
        return SAFE_REFUSAL
    if episode.prompt_category == PromptCategory.BORDERLINE:
        return BORDERLINE_SAFE
    if episode.verification.outcome == OutcomeLabel.HELPFUL_COMPLIANCE and episode.response:
        return episode.response
    return BENIGN_FALLBACK


def train_lora_adapter(
    *,
    console: Console,
    model_name: str,
    examples: list[dict[str, str]],
    output_dir: Path,
    epochs: int,
    max_length: int,
) -> None:
    if not examples:
        raise RuntimeError("No SFT examples were generated; cannot train adapter.")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    console.rule("[bold]LoRA adapter training")
    console.print(f"Model: {model_name}")
    console.print(f"Device: {device}")
    console.print(f"Examples: {len(examples)}")
    console.print(f"Output: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    provider = LocalTransformersProvider(
        model=model_name,
        device=device,
        dtype="float16" if device == "mps" else "float32",
        max_new_tokens=96,
        temperature=0.0,
        local_files_only=False,
        model_kwargs={"low_cpu_mem_usage": True},
    )
    model = provider._model
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    rows = [_encode_example(tokenizer, example, max_length, device) for example in examples]
    step = 0
    for epoch in range(epochs):
        total_loss = 0.0
        for row in rows:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**row)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            step += 1
            if step % 25 == 0:
                console.print(f"step={step} epoch={epoch + 1} loss={total_loss / step:.4f}")
        console.print(f"epoch={epoch + 1} average_loss={total_loss / len(rows):.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    adapter_path = output_dir / "adapter_model.safetensors"
    console.print(f"[green]Saved adapter safetensors:[/] {adapter_path}")


def _encode_example(
    tokenizer: Any,
    example: dict[str, str],
    max_length: int,
    device: str,
) -> dict[str, torch.Tensor]:
    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["response"]},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(messages, tokenize=False)
    else:
        text = f"user: {example['prompt']}\nassistant: {example['response']}"
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100
    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
        "labels": labels.to(device),
    }


def _print_metrics(console: Console, episodes: list[EpisodeResult], completed: int) -> None:
    summary = compute_metrics(episodes)
    console.print(
        f"[cyan]episodes={completed}[/] "
        f"attack_success={summary['attack_success_rate']:.3f} "
        f"safe_refusal={summary['safe_refusal_rate']:.3f} "
        f"benign_help={summary['benign_helpfulness_rate']:.3f} "
        f"over_refusal={summary['over_refusal_rate']:.3f}"
    )


def _print_summary(console: Console, summary: dict[str, Any]) -> None:
    table = Table(title="Final summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key, value in summary.items():
        if isinstance(value, dict):
            continue
        table.add_row(key, f"{value:.4f}" if isinstance(value, float) else str(value))
    console.print(table)


if __name__ == "__main__":
    sys.exit(main())
