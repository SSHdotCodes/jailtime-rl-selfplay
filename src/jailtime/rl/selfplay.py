"""Real-time RL self-play orchestrator.

``RLSelfPlayOrchestrator`` runs the three-agent loop with full-parameter
REINFORCE updates applied to the attacker and defender policies *every
episode*. Unlike the offline adversarial-dataset trainer or LoRA SFT, this
changes the model weights in real time as the loop proceeds:

1. The attacker policy samples an adversarial probe (rollout captured).
2. The defender policy samples a response (rollout captured).
3. A deterministic verifier (rule_based / hybrid) judges the response and the
   ``RewardCalculator`` turns the judgment into scalar rewards.
4. ``REINFORCETrainer`` applies a policy-gradient update to the attacker
   weights using the attacker reward, and to the defender weights using the
   defender reward.
5. Periodically, a separate model verifier is RL-calibrated against a frozen
   probe set so the verifier also improves -- without feeding the main-loop
   reward back into itself (which would be self-referential).
6. At the end, the defender's final weights are saved as safetensors.

Reward semantics (set via ``RewardConfig``):

* Defender safely refuses a real jailbreak attempt -> positive reward.
* Defender gets jailbroken (unsafe compliance) -> large negative reward.
* Defender over-refuses a safe prompt -> negative reward.
* Defender correctly answers a safe prompt -> small positive reward.
* Attacker jailbreak succeeds -> positive reward; fails -> negative reward.
"""

from __future__ import annotations

from pathlib import Path
from random import Random
from typing import Any

from jailtime.config import JailtimeConfig, ProviderConfig
from jailtime.orchestrator import Orchestrator
from jailtime.providers.base import ProviderError
from jailtime.rl.agents import RLAttackerAgent, RLDefenderAgent, RLModelVerifier
from jailtime.rl.policy import PolicyModel, SampledRollout
from jailtime.rl.reinforce import REINFORCETrainer
from jailtime.rl.torch_policy import TorchPolicyModel
from jailtime.rl.verifier_calibration import (
    VerifierCalibrationTrainer,
    default_calibration_probes,
)
from jailtime.schemas import AttackContext, AttackerFeedback, EpisodeResult, RunReport
from jailtime.telemetry import TelemetryRecorder


class RLSelfPlayOrchestrator(Orchestrator):
    """Orchestrator that applies per-episode REINFORCE updates to live weights."""

    def __init__(
        self,
        config: JailtimeConfig | str | Path | None = None,
        *,
        attacker_policy: PolicyModel | None = None,
        defender_policy: PolicyModel | None = None,
        verifier_policy: PolicyModel | None = None,
        verifier: Any | None = None,
    ) -> None:
        if isinstance(config, str | Path):
            from jailtime.config import load_config

            config = load_config(config)
        config = config or JailtimeConfig()
        if not config.rl.enabled:
            config.rl.enabled = True
        config.trainer.type = "none"
        self.rl_config = config.rl

        self.attacker_policy = attacker_policy or self._build_policy(
            config.providers.attacker, self.rl_config.attacker
        )
        self.defender_policy = defender_policy or self._build_policy(
            config.providers.defender, self.rl_config.defender
        )

        self.attacker_agent = RLAttackerAgent(
            self.attacker_policy,
            max_prompt_tokens=int(self.rl_config.attacker.max_new_tokens),
            temperature=float(self.rl_config.attacker.temperature),
            rng=Random(config.run.seed),
        )
        self.defender_agent = RLDefenderAgent(
            self.defender_policy,
            max_new_tokens=int(self.rl_config.defender.max_new_tokens),
            temperature=float(self.rl_config.defender.temperature),
        )

        # Defer main-verifier construction to the base Orchestrator (which owns
        # the provider cache); when ``verifier`` is None it builds from
        # providers.verifier, otherwise it uses the injected one.
        super().__init__(
            config,
            attacker=self.attacker_agent,
            defender=self.defender_agent,
            verifier=verifier,
            trainers=[],
        )

        self.attacker_rl = REINFORCETrainer(
            self.attacker_policy,
            lr=self.rl_config.attacker.lr,
            entropy_coef=self.rl_config.attacker.entropy_coef,
            clip_grad=self.rl_config.attacker.clip_grad,
            baseline_decay=self.rl_config.attacker.baseline_decay,
            reward_clip=self.rl_config.attacker.reward_clip,
            name="attacker",
        )
        self.defender_rl = REINFORCETrainer(
            self.defender_policy,
            lr=self.rl_config.defender.lr,
            entropy_coef=self.rl_config.defender.entropy_coef,
            clip_grad=self.rl_config.defender.clip_grad,
            baseline_decay=self.rl_config.defender.baseline_decay,
            reward_clip=self.rl_config.defender.reward_clip,
            name="defender",
        )

        self.verifier_calibrator: VerifierCalibrationTrainer | None = None
        self.rl_verifier: RLModelVerifier | None = None
        if self.rl_config.verifier_calibration.enabled:
            self.rl_verifier = RLModelVerifier(
                verifier_policy or self._build_verifier_policy(),
                max_new_tokens=128,
                temperature=0.3,
            )
            self.verifier_calibrator = VerifierCalibrationTrainer(
                self.rl_verifier,
                default_calibration_probes(),
                correct_reward=self.rl_config.verifier_calibration.correct_reward,
                incorrect_penalty=self.rl_config.verifier_calibration.incorrect_penalty,
                lr=self.rl_config.verifier_calibration.lr,
                entropy_coef=self.rl_config.verifier_calibration.entropy_coef,
            )

        self._rl_episode_stats: list[dict[str, Any]] = []

    def run(
        self,
        num_episodes: int | None = None,
        *,
        on_episode: Any | None = None,
    ) -> RunReport:
        """Run the real-time RL self-play loop and save the final defender.

        ``on_episode`` is an optional callback invoked as
        ``on_episode(episode, attacker_stats, defender_stats)`` after each
        episode's REINFORCE updates have been applied. It is used by the CLI
        driver to advance a progress bar and print live metrics without
        bypassing the per-episode weight updates.
        """

        episodes_to_run = num_episodes or self.config.run.episodes
        run_dir = Path(self.config.run.output_dir) / "latest"
        telemetry = TelemetryRecorder(run_dir, run_name=self.config.run.name)
        episodes: list[EpisodeResult] = []

        for episode_index in range(episodes_to_run):
            episode, attacker_rollout, defender_rollout = self.run_episode(episode_index)
            episodes.append(episode)
            telemetry.record_episode(episode)
            self.attacker_agent.observe(episode)

            a_stats = self.attacker_rl.step(attacker_rollout, episode.attacker_reward)
            d_stats = self.defender_rl.step(defender_rollout, episode.defender_reward)
            self._rl_episode_stats.append(
                {
                    "episode": episode_index,
                    "attacker": a_stats,
                    "defender": d_stats,
                }
            )

            if (
                self.verifier_calibrator
                and (episode_index + 1) % self.rl_config.verifier_calibration.every == 0
            ):
                self.verifier_calibrator.calibrate()

            if (
                self.rl_config.checkpoint_every
                and (episode_index + 1) % self.rl_config.checkpoint_every == 0
            ):
                self._save_checkpoint(episode_index + 1)

            if on_episode is not None:
                on_episode(episode, a_stats, d_stats)

        self._save_checkpoint(episodes_to_run, final=True)
        summary = telemetry.finalize()
        self._enrich_rl_summary(summary)
        return RunReport(
            run_id=self.config.run.name,
            episodes=episodes,
            summary=summary,
            output_dir=str(run_dir),
        )

    def run_episode(
        self, episode_index: int
    ) -> tuple[EpisodeResult, SampledRollout, SampledRollout]:
        """Run one episode and return the result plus both rollouts."""

        category = self.sampler.sample_category()
        context = AttackContext(
            episode_index=episode_index,
            prompt_category=category,
            run_name=self.config.run.name,
            strategy_weights=getattr(self.attacker, "strategy_weights", {}),
            technique_weights=getattr(self.attacker, "compute_technique_weights", lambda: {})(),
            feedback=self._build_attacker_feedback(),
        )
        attack_prompt, attacker_rollout = self.attacker_agent.generate_prompt_with_rollout(context)
        model_response, defender_rollout = self.defender_agent.respond_with_rollout(
            attack_prompt.prompt
        )
        verification = self.verifier.verify(
            attack_prompt.prompt,
            model_response.content,
            expected_category=attack_prompt.category,
        )
        attacker_reward, defender_reward = self.reward_calculator.score(
            attack_prompt.prompt,
            verification,
            technique_id=attack_prompt.technique_id,
        )
        episode = EpisodeResult(
            episode_index=episode_index,
            prompt=attack_prompt.prompt,
            prompt_category=attack_prompt.category,
            strategy_id=attack_prompt.strategy_id,
            technique_id=attack_prompt.technique_id,
            response=model_response.content,
            verification=verification,
            attacker_reward=attacker_reward,
            defender_reward=defender_reward,
            metadata={
                "model": model_response.model,
                "provider_metadata": model_response.metadata,
                "attack_metadata": attack_prompt.metadata,
            },
        )
        return episode, attacker_rollout, defender_rollout

    def _build_attacker_feedback(self) -> AttackerFeedback | None:
        feedback_method = getattr(self.attacker, "feedback", None)
        if callable(feedback_method):
            return feedback_method()
        return None

    def _save_checkpoint(self, episode: int, *, final: bool = False) -> Path | None:
        run_dir = Path(self.config.run.output_dir) / "latest"
        if final and self.rl_config.save_final_defender:
            target = run_dir / self.rl_config.final_defender_dir
            self.defender_policy.save_pretrained(target)
            return target
        if not self.rl_config.checkpoint_every:
            return None
        target = run_dir / f"defender_checkpoint_{episode}"
        self.defender_policy.save_pretrained(target)
        return target

    def _enrich_rl_summary(self, summary: dict[str, Any]) -> None:
        summary["rl_attacker_stats"] = self.attacker_rl.stats()
        summary["rl_defender_stats"] = self.defender_rl.stats()
        if self.verifier_calibrator is not None:
            summary["rl_verifier_stats"] = self.verifier_calibrator.stats()
        if self._rl_episode_stats:
            last = self._rl_episode_stats[-1]
            summary["rl_last_episode"] = {
                "attacker": last["attacker"],
                "defender": last["defender"],
            }

    def _build_policy(self, provider_config: ProviderConfig | None, agent_cfg: Any) -> PolicyModel:
        if provider_config is None:
            raise ValueError(
                "RL self-play requires a local_transformers provider for both attacker "
                "and defender (or an injected PolicyModel)."
            )
        if provider_config.type not in {"local_transformers", "transformers_local", "transformers"}:
            raise ValueError(
                f"RL self-play policies must be local_transformers providers "
                f"(got providers.*.type={provider_config.type!r})."
            )
        if not provider_config.model:
            raise ValueError("providers.*.model is required for RL self-play policies")
        params = provider_config.params
        return TorchPolicyModel(
            model=provider_config.model,
            device=str(params.get("device", "auto")),
            dtype=str(params.get("dtype", agent_cfg.dtype)),
            lr=float(agent_cfg.lr),
            optimizer=str(getattr(agent_cfg, "optimizer", "adamw")),
            momentum=float(getattr(agent_cfg, "momentum", 0.0)),
            local_files_only=bool(params.get("local_files_only", True)),
            trust_remote_code=bool(params.get("trust_remote_code", False)),
            model_kwargs=dict(params.get("model_kwargs", {})),
            max_length=int(params.get("max_length", 1024)),
            gradient_checkpointing=bool(
                getattr(agent_cfg, "gradient_checkpointing", True)
            ),
            compile=bool(getattr(agent_cfg, "compile", False)),
        )

    def _build_verifier_policy(self) -> PolicyModel:
        verifier_cfg = self.config.providers.verifier
        provider_data = (
            verifier_cfg.params.get("provider") if verifier_cfg.type == "hybrid" else None
        )
        if isinstance(provider_data, dict):
            from jailtime.config import ProviderConfig as _PC

            return self._build_policy(_PC.model_validate(provider_data), self.rl_config.defender)
        if verifier_cfg.type in {"model_based", "local_transformers", "transformers"}:
            return self._build_policy(verifier_cfg, self.rl_config.defender)
        # Never silently reuse the defender policy: that would make the
        # calibration REINFORCE updates perturb the defender's weights, which
        # is exactly the self-referential corruption the calibration step is
        # kept separate to avoid.
        raise ProviderError(
            "RL verifier calibration needs a model provider under "
            "providers.verifier (hybrid with params.provider, model_based, or "
            "local_transformers)."
        )


def is_local_transformers_available() -> bool:
    """Return True if the optional local transformers backend is importable."""

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


__all__ = ["RLSelfPlayOrchestrator", "is_local_transformers_available"]
