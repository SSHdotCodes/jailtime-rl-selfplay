"""Configuration schema and YAML loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


class RunConfig(BaseModel):
    """Top-level run metadata and output settings."""

    name: str = "jailtime-run"
    episodes: int = Field(default=50, ge=1)
    output_dir: str = "./runs"
    seed: int | None = 13


class SamplingConfig(BaseModel):
    """Prompt category sampling rates."""

    adversarial_rate: float = Field(default=0.5, ge=0.0)
    benign_rate: float = Field(default=0.4, ge=0.0)
    borderline_rate: float = Field(default=0.1, ge=0.0)

    @model_validator(mode="after")
    def rates_sum_to_one(self) -> SamplingConfig:
        total = self.adversarial_rate + self.benign_rate + self.borderline_rate
        if abs(total - 1.0) > 1e-6:
            msg = (
                "sampling rates must sum to 1.0 "
                f"(got adversarial+benign+borderline={total:.6f})"
            )
            raise ValueError(msg)
        return self


class RewardConfig(BaseModel):
    """Configurable reward coefficients for attacker and defender."""

    unsafe_compliance_penalty: float = -2.0
    safe_refusal_reward: float = 1.0
    helpful_compliance_reward: float = 1.0
    over_refusal_penalty: float = -1.0
    borderline_safe_reward: float = 0.5
    ambiguous_reward: float = 0.0
    evasive_penalty: float = -0.25
    failed_attack_penalty: float = -0.2
    successful_attack_reward: float = 1.0
    novelty_bonus: float = 0.25
    evasive_partial_credit: float = 0.15
    progressive_difficulty_bonus: float = 0.1
    technique_novelty_bonus: float = 0.2


class ProviderConfig(BaseModel):
    """Provider or verifier adapter configuration."""

    type: str = "mock"
    model: str | None = None
    base_url: str | None = None
    endpoint: str | None = None
    api_key_env: str | None = None
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def normalize_type(cls, value: str) -> str:
        return value.strip().lower().replace("-", "_")


class ProvidersConfig(BaseModel):
    """Configured defender and verifier adapters."""

    attacker: ProviderConfig | None = None
    defender: ProviderConfig = Field(default_factory=ProviderConfig)
    verifier: ProviderConfig = Field(default_factory=lambda: ProviderConfig(type="rule_based"))


class AttackerConfig(BaseModel):
    """Attacker construction settings."""

    type: Literal["registry", "model_guided", "model_wrapped", "adaptive_rl"] = "registry"
    params: dict[str, Any] = Field(default_factory=dict)


class TrainerConfig(BaseModel):
    """Training hook configuration."""

    type: Literal[
        "bandit",
        "rl_attacker",
        "rl_selfplay",
        "adversarial_dataset",
        "offline_dataset",
        "none",
    ] = "bandit"
    update_every: int = Field(default=10, ge=1)
    exploration_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    output_path: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class RLAgentConfig(BaseModel):
    """Per-role real-time RL hyperparameters."""

    lr: float = Field(default=1e-6, gt=0.0)
    optimizer: Literal["adamw", "sgd"] = "adamw"
    momentum: float = Field(default=0.0, ge=0.0, lt=1.0)
    entropy_coef: float = Field(default=0.01, ge=0.0)
    clip_grad: float = Field(default=1.0, gt=0.0)
    baseline_decay: float = Field(default=0.95, gt=0.0, lt=1.0)
    reward_clip: float = Field(default=10.0, gt=0.0)
    temperature: float = Field(default=0.7, ge=0.0)
    max_new_tokens: int = Field(default=96, ge=1)
    dtype: str = "float32"
    gradient_checkpointing: bool = True
    compile: bool = False


class RLVerifierCalibrationConfig(BaseModel):
    """Verifier RL calibration schedule."""

    enabled: bool = True
    every: int = Field(default=50, ge=1)
    lr: float = Field(default=1e-6, gt=0.0)
    entropy_coef: float = Field(default=0.005, ge=0.0)
    correct_reward: float = 1.0
    incorrect_penalty: float = -1.0


class RLConfig(BaseModel):
    """Real-time RL self-play configuration.

    When ``trainer.type == "rl_selfplay"`` the orchestrator applies full-
    parameter REINFORCE updates to the attacker and defender policies every
    episode. The defender's final weights are saved as safetensors to
    ``final_defender_dir``.

    The main-loop verifier label is taken from the deterministic
    ``providers.verifier`` adapter (rule_based / hybrid) to avoid
    self-referential reward hacking. A separate model verifier is RL-calibrated
    against a frozen probe set and can optionally be promoted to the main
    verifier after calibration rounds.
    """

    enabled: bool = False
    checkpoint_every: int = Field(default=100, ge=1)
    save_final_defender: bool = True
    final_defender_dir: str = "defender_final"
    promote_calibrated_verifier: bool = False
    attacker: RLAgentConfig = Field(default_factory=RLAgentConfig)
    defender: RLAgentConfig = Field(default_factory=lambda: RLAgentConfig(temperature=0.7))
    verifier_calibration: RLVerifierCalibrationConfig = Field(
        default_factory=RLVerifierCalibrationConfig
    )


class JailtimeConfig(BaseModel):
    """Complete jailtime configuration."""

    run: RunConfig = Field(default_factory=RunConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    rewards: RewardConfig = Field(default_factory=RewardConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    attacker: AttackerConfig = Field(default_factory=AttackerConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    rl: RLConfig = Field(default_factory=RLConfig)


def load_config(path: str | Path) -> JailtimeConfig:
    """Load a YAML config file into a validated ``JailtimeConfig``."""

    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config file not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in config file {config_path}: {exc}") from exc
    try:
        return JailtimeConfig.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid jailtime config {config_path}: {exc}") from exc


def load_config_from_mapping(data: dict[str, Any]) -> JailtimeConfig:
    """Validate a mapping as ``JailtimeConfig``."""

    return JailtimeConfig.model_validate(data)
