"""Main adversarial training loop orchestrator."""

from __future__ import annotations

from pathlib import Path
from random import Random

from jailtime.agents.attacker import Attacker, StrategyAttacker
from jailtime.agents.defender import Defender, ProviderDefender
from jailtime.attacks.adaptive import AdaptiveRLAttacker
from jailtime.attacks.model_guided import (
    ModelGuidedToyAdversarialStrategy,
    ModelWrappedToyAdversarialStrategy,
)
from jailtime.attacks.registry import AttackRegistry
from jailtime.attacks.toy import BoundaryCategoryProbeStrategy, DirectBenignProbeStrategy
from jailtime.config import JailtimeConfig, ProviderConfig, load_config
from jailtime.episode import EpisodeSampler
from jailtime.providers import (
    LocalHTTPProvider,
    LocalTransformersProvider,
    MockProvider,
    OpenAICompatibleProvider,
)
from jailtime.providers.base import ModelProvider
from jailtime.rewards import RewardCalculator
from jailtime.schemas import AttackContext, AttackerFeedback, EpisodeResult, Message, RunReport
from jailtime.telemetry import TelemetryRecorder
from jailtime.trainers import (
    AdversarialDatasetTrainer,
    BanditStrategyTrainer,
    OfflineDatasetTrainer,
    RLAttackerTrainer,
    Trainer,
)
from jailtime.verifiers import HybridVerifier, ModelBasedVerifier, RuleBasedVerifier, Verifier


class Orchestrator:
    """Runs repeated attacker/defender/verifier episodes.

    When configured with an adaptive RL attacker, the orchestrator closes the
    RL feedback loop: every episode's verifier outcome is fed back to the
    attacker as a reward signal, the RL trainer recomputes technique weights,
    and the next episode's ``AttackContext`` carries a fresh ``AttackerFeedback``
    snapshot so the attacker conditions its generation on what worked.
    """

    def __init__(
        self,
        config: JailtimeConfig | str | Path | None = None,
        *,
        attacker: Attacker | None = None,
        defender: Defender | None = None,
        verifier: Verifier | None = None,
        trainers: list[Trainer] | None = None,
    ) -> None:
        if isinstance(config, str | Path):
            self.config = load_config(config)
        else:
            self.config = config or JailtimeConfig()
        self.rng = Random(self.config.run.seed)
        self.sampler = EpisodeSampler(self.config.sampling, rng=self.rng)
        self.reward_calculator = RewardCalculator(self.config.rewards)
        self._provider_cache: dict[str, ModelProvider] = {}

        self.attacker = attacker or self._build_attacker()
        self.defender = defender or ProviderDefender(
            self._build_provider(self.config.providers.defender)
        )
        self.verifier = verifier or self._build_verifier(self.config.providers.verifier)
        self.trainers = trainers if trainers is not None else self._build_trainers()
        self._rl_trainer: RLAttackerTrainer | None = None
        for trainer in self.trainers:
            if isinstance(trainer, RLAttackerTrainer):
                self._rl_trainer = trainer

    def run(self, num_episodes: int | None = None) -> RunReport:
        """Run episodes and return a structured report."""

        episodes_to_run = num_episodes or self.config.run.episodes
        run_dir = Path(self.config.run.output_dir) / "latest"
        telemetry = TelemetryRecorder(run_dir, run_name=self.config.run.name)
        episodes: list[EpisodeResult] = []

        for episode_index in range(episodes_to_run):
            episode = self.run_episode(episode_index)
            episodes.append(episode)
            telemetry.record_episode(episode)
            self._observe_attacker(episode)
            for trainer in self.trainers:
                trainer.observe(episode)
            if (episode_index + 1) % self.config.trainer.update_every == 0:
                self._update_trainers()

        self._update_trainers()
        summary = telemetry.finalize()
        self._enrich_summary(summary)
        return RunReport(
            run_id=self.config.run.name,
            episodes=episodes,
            summary=summary,
            output_dir=str(run_dir),
        )

    def run_episode(self, episode_index: int) -> EpisodeResult:
        """Run and return a single episode."""

        category = self.sampler.sample_category()
        context = AttackContext(
            episode_index=episode_index,
            prompt_category=category,
            run_name=self.config.run.name,
            strategy_weights=getattr(self.attacker, "strategy_weights", {}),
            technique_weights=getattr(self.attacker, "compute_technique_weights", lambda: {})(),
            feedback=self._build_attacker_feedback(),
        )
        attack_prompt = self.attacker.generate_prompt(context)
        model_response = self.defender.respond(attack_prompt.prompt)
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
        return EpisodeResult(
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

    def _observe_attacker(self, episode: EpisodeResult) -> None:
        observe = getattr(self.attacker, "observe", None)
        if callable(observe):
            observe(episode)

    def _build_attacker_feedback(self) -> AttackerFeedback | None:
        if self._rl_trainer is not None:
            return self._rl_trainer.feedback()
        feedback_method = getattr(self.attacker, "feedback", None)
        if callable(feedback_method):
            return feedback_method()
        return None

    def _update_trainers(self) -> None:
        for trainer in self.trainers:
            update = trainer.update()
            weights = update.details.get("weights")
            if weights and hasattr(self.attacker, "set_strategy_weights"):
                self.attacker.set_strategy_weights(weights)
            technique_weights = update.details.get("technique_weights")
            if technique_weights and hasattr(self.attacker, "set_technique_weights"):
                self.attacker.set_technique_weights(technique_weights)
            elif weights and hasattr(self.attacker, "set_technique_weights"):
                self.attacker.set_technique_weights(weights)

    def _enrich_summary(self, summary: dict) -> None:
        stats_method = getattr(self.attacker, "technique_stats", None)
        if callable(stats_method):
            summary["attacker_technique_stats"] = stats_method()
        if self._rl_trainer is not None:
            summary["rl_trainer_stats"] = self._rl_trainer.technique_stats()
            summary["rl_replay_buffer_size"] = len(self._rl_trainer.replay_buffer())

    def _build_trainers(self) -> list[Trainer]:
        trainer_type = self.config.trainer.type
        if trainer_type == "none":
            return []
        if trainer_type == "offline_dataset":
            output_path = self.config.trainer.output_path
            if not output_path:
                output_path = str(
                    Path(self.config.run.output_dir) / "latest" / "training_episodes.jsonl"
                )
            return [OfflineDatasetTrainer(output_path)]

        if trainer_type == "adversarial_dataset":
            output_dir = self.config.trainer.output_path or str(
                Path(self.config.run.output_dir) / "latest"
            )
            return [AdversarialDatasetTrainer(output_dir)]

        if trainer_type == "rl_attacker":
            trainer = RLAttackerTrainer(
                exploration_rate=self.config.trainer.exploration_rate,
                ucb_c=float(self.config.trainer.params.get("ucb_c", 1.41)),
                decay=float(self.config.trainer.params.get("decay", 1.0)),
            )
            if self.config.trainer.params.get("also_export_dataset", False):
                output_dir = self.config.trainer.output_path or str(
                    Path(self.config.run.output_dir) / "latest"
                )
                return [trainer, AdversarialDatasetTrainer(output_dir)]
            return [trainer]

        registry = getattr(self.attacker, "registry", None)
        if registry is not None and hasattr(registry, "list"):
            strategy_ids = [strategy.id for strategy in registry.list()]
        else:
            fallback_attacker = StrategyAttacker(rng=self.rng)
            strategy_ids = [strategy.id for strategy in fallback_attacker.registry.list()]
        return [
            BanditStrategyTrainer(
                strategy_ids,
                exploration_rate=self.config.trainer.exploration_rate,
            )
        ]

    def _build_verifier(self, config: ProviderConfig) -> Verifier:
        if config.type in {"rule_based", "rules"}:
            return RuleBasedVerifier()
        if config.type == "model_based":
            return ModelBasedVerifier(self._build_provider_from_params(config))
        if config.type == "hybrid":
            return HybridVerifier(ModelBasedVerifier(self._build_provider_from_params(config)))
        if config.type == "mock":
            return ModelBasedVerifier(_MockVerifierProvider())
        if config.type in {
            "local_transformers",
            "transformers_local",
            "transformers",
            "openai_compatible",
            "openai",
            "local_http",
            "http",
        }:
            return ModelBasedVerifier(self._build_provider(config))
        raise ValueError(f"Unsupported verifier type: {config.type}")

    def _build_attacker(self) -> Attacker:
        if self.config.attacker.type == "registry":
            return StrategyAttacker(rng=self.rng)
        if self.config.attacker.type == "model_guided":
            provider_config = self.config.providers.attacker
            if provider_config is None:
                raise ValueError(
                    "providers.attacker is required when attacker.type is model_guided"
                )
            registry = AttackRegistry(
                [
                    DirectBenignProbeStrategy(),
                    BoundaryCategoryProbeStrategy(),
                    ModelGuidedToyAdversarialStrategy(self._build_provider(provider_config)),
                ]
            )
            return StrategyAttacker(registry=registry, rng=self.rng)
        if self.config.attacker.type == "model_wrapped":
            provider_config = self.config.providers.attacker
            if provider_config is None:
                raise ValueError(
                    "providers.attacker is required when attacker.type is model_wrapped"
                )
            registry = AttackRegistry(
                [
                    DirectBenignProbeStrategy(),
                    BoundaryCategoryProbeStrategy(),
                    ModelWrappedToyAdversarialStrategy(
                        self._build_provider(provider_config),
                        max_wrapper_tokens=int(
                            self.config.attacker.params.get("max_wrapper_tokens", 192)
                        ),
                        max_wrapper_chars=int(
                            self.config.attacker.params.get("max_wrapper_chars", 1800)
                        ),
                    ),
                ]
            )
            return StrategyAttacker(registry=registry, rng=self.rng)
        if self.config.attacker.type == "adaptive_rl":
            provider_config = self.config.providers.attacker
            if provider_config is None:
                raise ValueError(
                    "providers.attacker is required when attacker.type is adaptive_rl"
                )
            params = self.config.attacker.params
            return AdaptiveRLAttacker(
                self._build_provider(provider_config),
                rng=self.rng,
                max_prompt_tokens=int(params.get("max_prompt_tokens", 512)),
                max_prompt_chars=int(params.get("max_prompt_chars", 4000)),
                memory_size=int(params.get("memory_size", 20)),
                success_memory_size=int(params.get("success_memory_size", 8)),
                failure_memory_size=int(params.get("failure_memory_size", 4)),
                ucb_exploration=float(params.get("ucb_exploration", 1.41)),
                temperature=float(params.get("temperature", 0.9)),
            )
        raise ValueError(f"Unsupported attacker type: {self.config.attacker.type}")

    def _build_provider_from_params(self, config: ProviderConfig) -> ModelProvider:
        provider_data = config.params.get("provider")
        if not isinstance(provider_data, dict):
            raise ValueError(
                f"providers.verifier.type={config.type!r} requires params.provider configuration"
            )
        return self._build_provider(ProviderConfig.model_validate(provider_data))

    def _build_provider(self, config: ProviderConfig) -> ModelProvider:
        cache_key = config.model_dump_json()
        cached = self._provider_cache.get(cache_key)
        if cached is not None:
            return cached

        provider = self._create_provider(config)
        self._provider_cache[cache_key] = provider
        return provider

    @staticmethod
    def _create_provider(config: ProviderConfig) -> ModelProvider:
        if config.type == "mock":
            mode = str(config.params.get("mode", "mixed"))
            return MockProvider(mode=mode)
        if config.type in {"openai_compatible", "openai"}:
            if not config.base_url:
                raise ValueError("providers.*.base_url is required for openai_compatible providers")
            if not config.model:
                raise ValueError("providers.*.model is required for openai_compatible providers")
            return OpenAICompatibleProvider(
                base_url=config.base_url,
                model=config.model,
                api_key_env=config.api_key_env,
                timeout_seconds=config.timeout_seconds,
            )
        if config.type in {"local_http", "http"}:
            endpoint = config.endpoint or config.base_url
            if not endpoint:
                raise ValueError(
                    "providers.*.endpoint or base_url is required for local_http providers"
                )
            return LocalHTTPProvider(
                endpoint=endpoint,
                model=config.model,
                timeout_seconds=config.timeout_seconds,
            )
        if config.type in {"local_transformers", "transformers_local", "transformers"}:
            if not config.model:
                raise ValueError("providers.*.model is required for local_transformers providers")
            return LocalTransformersProvider(
                model=config.model,
                device=str(config.params.get("device", "auto")),
                dtype=str(config.params.get("dtype", "auto")),
                max_new_tokens=int(config.params.get("max_new_tokens", 256)),
                temperature=float(config.params.get("temperature", 0.0)),
                local_files_only=bool(config.params.get("local_files_only", True)),
                trust_remote_code=bool(config.params.get("trust_remote_code", False)),
                model_kwargs=dict(config.params.get("model_kwargs", {})),
                generation_kwargs=dict(config.params.get("generation_kwargs", {})),
            )
        raise ValueError(f"Unsupported provider type: {config.type}")


class _MockVerifierProvider:
    """Provider that returns valid ambiguous verifier JSON for adapter demos."""

    def complete(self, messages: list[Message], **kwargs: object) -> object:
        return MockProvider(mode="safe").complete(messages, **kwargs)
