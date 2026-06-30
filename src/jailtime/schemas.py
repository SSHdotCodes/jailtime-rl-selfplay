"""Public data contracts used across jailtime."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PromptCategory(StrEnum):
    """Expected prompt category for sampling, rewards, and verification."""

    ADVERSARIAL = "ADVERSARIAL"
    BENIGN = "BENIGN"
    BORDERLINE = "BORDERLINE"
    UNKNOWN = "UNKNOWN"


class OutcomeLabel(StrEnum):
    """Verifier outcome labels used for reward assignment and metrics."""

    SAFE_REFUSAL = "SAFE_REFUSAL"
    UNSAFE_COMPLIANCE = "UNSAFE_COMPLIANCE"
    HELPFUL_COMPLIANCE = "HELPFUL_COMPLIANCE"
    OVER_REFUSAL = "OVER_REFUSAL"
    EVASIVE = "EVASIVE"
    AMBIGUOUS = "AMBIGUOUS"


class Message(BaseModel):
    """A provider-agnostic chat message."""

    role: str
    content: str


class ModelResponse(BaseModel):
    """A model response with optional provider metadata."""

    content: str
    model: str | None = None
    raw: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AttackerFeedback(BaseModel):
    """RL feedback passed from the trainer back to the attacker each step."""

    technique_weights: dict[str, float] = Field(default_factory=dict)
    technique_stats: dict[str, dict[str, Any]] = Field(default_factory=dict)
    recent_successes: list[dict[str, Any]] = Field(default_factory=list)
    recent_failures: list[dict[str, Any]] = Field(default_factory=list)
    global_attack_success_rate: float = 0.0
    step: int = 0


class AttackContext(BaseModel):
    """Context passed to attacker strategies when generating prompts."""

    episode_index: int
    prompt_category: PromptCategory
    run_name: str = "jailtime-run"
    strategy_weights: dict[str, float] = Field(default_factory=dict)
    technique_weights: dict[str, float] = Field(default_factory=dict)
    feedback: AttackerFeedback | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AttackPrompt(BaseModel):
    """A prompt proposed by an attacker or environment strategy."""

    prompt: str
    category: PromptCategory
    strategy_id: str
    technique_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerificationResult(BaseModel):
    """Structured verifier output with safety and helpfulness judgments."""

    prompt_category: PromptCategory
    outcome: OutcomeLabel
    safety_score: float = Field(ge=0.0, le=1.0)
    helpfulness_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class EpisodeResult(BaseModel):
    """Complete structured result for a single adversarial loop episode."""

    model_config = ConfigDict(use_enum_values=False)

    episode_index: int
    prompt: str
    prompt_category: PromptCategory
    strategy_id: str
    technique_id: str | None = None
    response: str
    verification: VerificationResult
    attacker_reward: float
    defender_reward: float
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrainingUpdate(BaseModel):
    """Result returned by a trainer after an update step."""

    trainer_name: str
    updated: bool
    details: dict[str, Any] = Field(default_factory=dict)


class RunReport(BaseModel):
    """Structured return value for orchestrator runs."""

    run_id: str
    episodes: list[EpisodeResult]
    summary: dict[str, Any]
    output_dir: str
