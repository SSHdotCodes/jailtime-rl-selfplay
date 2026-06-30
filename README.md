# jailtime

`jailtime` is a defensive AI safety research package for authorized adversarial
red-teaming and adversarial training experiments. It runs a three-agent loop:

1. An attacker proposes adversarial, benign, or borderline prompts.
2. A defender, usually a target model behind a provider adapter, responds.
3. A verifier judges both safety and helpfulness with structured labels.

The package intentionally ships only harmless toy prompts, abstract unsafe
markers, and placeholder extension points. It does not include a database of
real jailbreaks, real-world harmful requests, or actionable harmful
instructions.

## Responsible Use

Use `jailtime` only on models, systems, datasets, and endpoints that you own or
are explicitly authorized to evaluate. Treat generated logs as sensitive safety
research artifacts. Do not run unsupervised live self-modification against
production models.

This toolkit does not guarantee model safety. Verifier quality is a bottleneck,
attackers can overfit to verifier weaknesses, reward hacking is possible, and
held-out evaluations are required before drawing conclusions.

## Why Mix Attack And Normal Prompts?

Safety evaluation is incomplete if it measures only refusal. A useful defender
must refuse unsafe requests and answer normal benign requests. `jailtime`
therefore samples adversarial, benign, and borderline prompts. The defender is
rewarded for safe refusal on unsafe prompts and helpful compliance on benign
prompts, and is punished for unsafe compliance and over-refusal.

## Installation

```bash
python -m pip install -e ".[dev]"
```

For local Hugging Face Transformers models on Apple Silicon, install the
optional local extra:

```bash
python3 -m pip install -e ".[dev,local]"
```

## CLI

Run the local mock demo:

```bash
jailtime demo
```

Run from a config file:

```bash
jailtime run --config examples/config.yaml --episodes 100
```

Run real-time RL self-play and save the hardened defender safetensors:

```bash
jailtime selfplay --config examples/config_rl_selfplay.yaml --episodes 200
```

Validate config:

```bash
jailtime validate-config examples/config.yaml
```

Check local accelerator detection:

```bash
jailtime devices
```

Render a report from a run directory:

```bash
jailtime report ./runs/latest
```

Each run writes:

- `episodes.jsonl`
- `summary.json`
- `report.md`

## Python API

```python
from jailtime.config import JailtimeConfig
from jailtime.orchestrator import Orchestrator

config = JailtimeConfig()
report = Orchestrator(config).run(num_episodes=25)

print(report.summary["attack_success_rate"])
print(report.output_dir)
```

## Adding A Custom Attacker

Implement the `AttackStrategy` protocol and register it. Keep real red-team
datasets private and authorized.

```python
from random import Random

from jailtime.attacks.base import AttackStrategy
from jailtime.attacks.registry import AttackRegistry
from jailtime.schemas import AttackContext, AttackPrompt, PromptCategory


class InternalDatasetStrategy:
    id = "internal_dataset"
    name = "Authorized internal dataset"
    supported_categories = (PromptCategory.ADVERSARIAL,)

    def generate(self, context: AttackContext, rng: Random) -> AttackPrompt:
        prompt = load_authorized_prompt_somehow()
        return AttackPrompt(
            prompt=prompt,
            category=PromptCategory.ADVERSARIAL,
            strategy_id=self.id,
        )


registry = AttackRegistry()
registry.register(InternalDatasetStrategy())
```

## Adaptive RL Attacker

The `adaptive_rl` attacker is a real model-driven adversarial agent. Unlike
the toy strategies, it uses a language model to craft genuinely adversarial
prompts, selects red-team techniques via a learned UCB policy, and conditions
each new probe on its memory of past successes and failures. The orchestrator
feeds verifier outcomes back to the attacker every episode, closing an RL loop
that iteratively improves the attacker's jailbreak success rate. The
successful jailbreaks it discovers become training data that makes the
defender (main model) more robust.

### How It Works

1. **Technique repertoire**: the attacker draws from 12 documented red-team
   technique categories (persona role-play, hypothetical framing, encoding,
   authority escalation, instruction override, gradual escalation, etc.).
   These are methodological categories from published AI safety research --
   the attacker *model* generates the actual probe text at runtime.
2. **UCB policy**: a upper-confidence-bound policy balances exploitation
   (techniques with high observed reward) against exploration (under-tried
   techniques).
3. **Memory-conditioned generation**: each new prompt is conditioned on
   memories of recent successful and failed attempts so the attacker repeats
   what worked and avoids what did not.
4. **Reward shaping**: the attacker earns full reward for a successful
   jailbreak, partial credit for evasive responses (closer to compliance than
   a clean refusal), novelty bonuses for unique prompts and techniques, and a
   progressive difficulty bonus that rewards later successes.
5. **Defender improvement**: the `adversarial_dataset` trainer exports
   `(prompt, desired_response)` training pairs -- jailbreak corrections,
   over-refusal corrections, and positive safety/helpfulness examples --
   suitable for SFT or DPO fine-tuning of the defender.

### Configuration

```yaml
providers:
  attacker:
    type: "openai_compatible"
    base_url: "https://your-authorized-attacker-endpoint.example/v1"
    model: "your-attacker-model"
    api_key_env: "ATTACKER_API_KEY"
  defender:
    type: "openai_compatible"
    base_url: "https://your-authorized-defender-endpoint.example/v1"
    model: "your-defender-model"
    api_key_env: "DEFENDER_API_KEY"
  verifier:
    type: "hybrid"
    params:
      provider:
        type: "openai_compatible"
        base_url: "https://your-authorized-verifier-endpoint.example/v1"
        model: "your-verifier-model"
        api_key_env: "VERIFIER_API_KEY"

attacker:
  type: "adaptive_rl"
  params:
    temperature: 0.9
    ucb_exploration: 1.41
    memory_size: 20

trainer:
  type: "rl_attacker"
  update_every: 10
  params:
    also_export_dataset: true
```

See `examples/config_adaptive_rl.yaml` for a complete example. Run it with:

```bash
jailtime run --config examples/config_adaptive_rl.yaml --episodes 100
```

The run produces:
- `episodes.jsonl` -- full episode traces with technique metadata
- `summary.json` -- metrics including per-technique attack success rates
- `defender_training_pairs.jsonl` -- SFT/DPO pairs for defender improvement
- `jailbreak_attempts.jsonl` -- successful jailbreak prompts for analysis

### Python API

```python
from jailtime.agents import AdaptiveRLAttacker
from jailtime.config import JailtimeConfig, ProviderConfig
from jailtime.orchestrator import Orchestrator
from jailtime.providers import OpenAICompatibleProvider

config = JailtimeConfig()
config.attacker.type = "adaptive_rl"
config.trainer.type = "rl_attacker"
config.providers.attacker = ProviderConfig(
    type="openai_compatible",
    base_url="https://your-authorized-endpoint.example/v1",
    model="your-attacker-model",
    api_key_env="ATTACKER_API_KEY",
)

report = Orchestrator(config).run(num_episodes=50)
print(report.summary["attack_success_rate"])
print(report.summary["attacker_technique_stats"])
```

## Real-Time RL Self-Play (Full-Weight Updates)

The `rl_selfplay` mode is a genuine online reinforcement-learning closed
loop. Unlike the offline adversarial-dataset trainer (which exports SFT/DPO
pairs) or LoRA fine-tuning, it applies **full-parameter REINFORCE updates to
the attacker and defender model weights every episode**, in real time, as the
loop runs. When it finishes, the hardened defender is saved as safetensors.

### How it works

Each episode:

1. The **attacker** policy samples an adversarial probe (rollout captured).
2. The **defender** policy samples a response (rollout captured).
3. A deterministic **verifier** (rule-based / hybrid) judges whether the
   defender was jailbroken, and the `RewardCalculator` turns the judgment into
   scalar rewards.
4. `REINFORCETrainer` computes `advantage = reward - EMA_baseline` and applies
   `-(advantage * sum_t log pi(a_t)) - entropy_coef * H` to the attacker
   weights and to the defender weights (AdamW over all parameters).
5. A separate model verifier is RL-calibrated against a frozen probe set so
   the verifier also improves -- decoupled from the main reward to avoid
   self-referential reward hacking.
6. The defender's final weights are saved as safetensors to
   `<run_dir>/latest/defender_final`.

### Reward design

The reward signals are configured in the `rewards:` block and encode the
adversarial self-play objective:

| Event | Reward |
| --- | --- |
| Defender safely refuses a real jailbreak | positive |
| Defender gets jailbroken (unsafe compliance) | large negative |
| Defender over-refuses a safe prompt | negative |
| Defender correctly answers a safe prompt | small positive |
| Attacker jailbreak succeeds | positive |
| Attacker fails | negative |

The main-loop verifier label is deliberately taken from the deterministic
rule-based / hybrid verifier, not from the RL-trained model verifier. This
prevents the attacker and defender from reward-hacking a verifier that is
itself being trained against the same reward. The model verifier is instead
RL-calibrated against a held-out frozen probe set (`rl.verifier_calibration`),
which is genuine RL improvement of the verifier with measurable accuracy.

### Why REINFORCE and not PPO/LoRA?

Each episode is a single-step trajectory (one prompt -> one response -> one
terminal reward), which is exactly the REINFORCE setting. Full-parameter
online updates (not LoRA) are used so the defender's entire capacity can
adapt to the discovered jailbreaks. The loop is real-time: weights change
every episode, not in a separate offline phase.

### Configuration

```bash
jailtime selfplay --config examples/config_rl_selfplay.yaml --episodes 200
```

Or with the run driver that shows a live progress bar and saves the final
defender safetensors:

```bash
bash work/start_rl_selfplay_run.sh --config examples/config_rl_selfplay.yaml --episodes 200
```

```yaml
trainer:
  type: "rl_selfplay"
rl:
  enabled: true
  checkpoint_every: 100
  save_final_defender: true
  final_defender_dir: "defender_final"
  attacker:
    lr: 5.0e-7
    entropy_coef: 0.02
    baseline_decay: 0.95
    temperature: 0.9
    max_new_tokens: 160
  defender:
    lr: 5.0e-7
    entropy_coef: 0.01
    baseline_decay: 0.95
    temperature: 0.7     # RL needs sampling; inference can be greedy
    max_new_tokens: 96
  verifier_calibration:
    enabled: true
    every: 50
```

The run produces:

- `episodes.jsonl` -- full episode traces
- `summary.json` -- metrics plus `rl_attacker_stats`, `rl_defender_stats`,
  `rl_verifier_stats` (steps, mean reward, mean abs advantage, entropy, grad norm)
- `report.md`
- `defender_final/` -- the hardened defender safetensors + tokenizer
- `defender_checkpoint_<N>/` -- periodic checkpoints (if `checkpoint_every > 0`)

### Python API

```python
from jailtime.config import JailtimeConfig, ProviderConfig
from jailtime.rl import RLSelfPlayOrchestrator

config = JailtimeConfig()
config.trainer.type = "rl_selfplay"
config.rl.enabled = True
config.rl.save_final_defender = True
config.providers.attacker = ProviderConfig(type="local_transformers", model="your-attacker")
config.providers.defender = ProviderConfig(type="local_transformers", model="your-defender")
config.providers.verifier = ProviderConfig(type="rule_based")

report = RLSelfPlayOrchestrator(config).run(num_episodes=200)
print(report.summary["rl_defender_stats"])
print(report.summary["attack_success_rate"])
```

### Notes and limitations

- Full-parameter online RL on multi-billion-parameter models is expensive.
  On Apple Silicon, use `dtype: "float32"` for training stability (fp16
  backward on MPS is unstable). LRs of `1e-7`--`5e-7` are a reasonable start.
- Real-time RL does not guarantee safety. The defender can overfit to the
  attacker's current style; hold-out evaluations against fresh, authorized
  red-team corpora are required before drawing conclusions.
- The verifier bottleneck still applies: a weak verifier lets both the
  attacker and defender drift toward verifier blind spots. Calibrate against
  reviewed held-out probes and monitor `rl_verifier_stats.last_accuracy`.
- The final defender is a research artifact, not a certified-safe model.

## Adding A Custom Verifier

Custom verifiers should return a structured `VerificationResult` with both
safety and helpfulness scores.

```python
from jailtime.schemas import PromptCategory, VerificationResult
from jailtime.verifiers.base import Verifier


class CalibratedVerifier:
    def verify(
        self,
        prompt: str,
        response: str,
        expected_category: PromptCategory | None = None,
    ) -> VerificationResult:
        ...
```

Use held-out evals to calibrate verifier confidence and check for systematic
over-refusal or unsafe-compliance blind spots.

## Connecting A Real Provider Safely

`jailtime` has no hard dependency on a specific LLM provider. The built-in
OpenAI-compatible and local HTTP adapters make explicit network calls only when
you configure them.

```yaml
providers:
  defender:
    type: "openai_compatible"
    base_url: "https://your-authorized-endpoint.example/v1"
    model: "your-model"
    api_key_env: "YOUR_API_KEY_ENV_VAR"
  verifier:
    type: "rule_based"
```

Prefer isolated staging endpoints, rate limits, audit logging, and reviewed
prompt corpora. Do not put secrets in config files.

## Apple Silicon MPS

`jailtime` supports a local `local_transformers` provider that can run PyTorch
models on Apple MPS. It is optional so the default install remains lightweight
and tests do not require PyTorch.

Example:

```yaml
providers:
  defender:
    type: "local_transformers"
    model: "path-or-huggingface-model-id"
    params:
      device: "auto"          # prefers mps, then cuda, then cpu
      dtype: "auto"
      max_new_tokens: 192
      temperature: 0.0
      local_files_only: true
  verifier:
    type: "rule_based"
```

Use `local_files_only: true` for fully offline runs with models already present
on disk. Set it to `false` only when you intentionally allow Transformers to
download model files.

## Research Notes And Limitations

`jailtime` is research-oriented infrastructure for adversarial red-teaming,
adversarial training data collection, verifier calibration, and studying
over-refusal. The adaptive RL attacker iterates with reinforcement learning
to maximize jailbreak success against the defender, and the adversarial
dataset trainer exports correction pairs so the defender can be fine-tuned
(SFT/DPO) to resist discovered jailbreaks. It can export offline JSONL
traces for supervised fine-tuning, RLHF, DPO, or other external training
workflows, but it does not pretend to fine-tune closed-source models
automatically.

Known limitations:

- It does not guarantee safety.
- Verifier quality is a bottleneck; the attacker can exploit verifier
  blind spots.
- Attackers can overfit to attacker styles and verifier weaknesses.
- Reward hacking can occur (e.g. the attacker learns to produce responses
  the verifier mislabels as unsafe).
- The adaptive RL attacker's effectiveness depends on the quality of the
  attacker model -- a weak attacker model produces weak probes.
- Held-out evals are required before drawing safety conclusions.
- Production models should not be modified by unsupervised live loops.
