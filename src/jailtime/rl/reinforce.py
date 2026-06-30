"""Role-agnostic REINFORCE trainer with an exponential moving-average baseline.

Each episode of the self-play loop is a single-step trajectory: a policy
samples a sequence of tokens and receives one scalar terminal reward from the
verifier. That is exactly the setting for REINFORCE:

    advantage = reward - baseline
    loss      = -(advantage * sum_t log pi(a_t | s)) - entropy_coef * H(pi)

``REINFORCETrainer`` owns the running baseline (an EMA of observed rewards) so
the advantage is centered, applies ``reward_clip`` for stability, and delegates
the actual gradient/backward step to the ``PolicyModel``. It is role-agnostic:
one instance trains the attacker, another trains the defender, another trains
the verifier.
"""

from __future__ import annotations

from typing import Any

from jailtime.rl.policy import PolicyModel, SampledRollout


class REINFORCETrainer:
    """Online REINFORCE trainer for a single ``PolicyModel``.

    Parameters
    ----------
    policy:
        The policy to update. Its ``reinforce_step`` is called once per
        ``step``.
    lr, entropy_coef, clip_grad:
        Forwarded to ``policy.reinforce_step`` on every update.
    baseline_decay:
        EMA weight for the reward baseline (``0 < decay < 1``). Higher values
        keep a slower-moving baseline.
    reward_clip:
        Rewards are clipped to ``[-reward_clip, reward_clip]`` before being
        used, which protects the policy gradient from rare extreme verifier
        scores.
    name:
        Optional label included in stats for telemetry.
    """

    def __init__(
        self,
        policy: PolicyModel,
        *,
        lr: float = 1e-6,
        entropy_coef: float = 0.01,
        clip_grad: float = 1.0,
        baseline_decay: float = 0.95,
        reward_clip: float = 10.0,
        name: str = "",
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0.0 < baseline_decay < 1.0:
            raise ValueError("baseline_decay must be in (0, 1)")
        self.policy = policy
        self.lr = float(lr)
        self.entropy_coef = float(entropy_coef)
        self.clip_grad = float(clip_grad)
        self.baseline_decay = float(baseline_decay)
        self.reward_clip = float(reward_clip)
        self.name = name

        self._baseline = 0.0
        self._steps = 0
        self._skipped = 0
        self._reward_sum = 0.0
        self._advantage_sum = 0.0
        self._loss_sum = 0.0
        self._entropy_sum = 0.0
        self._grad_norm_sum = 0.0
        self._recent_rewards: list[float] = []

    def step(self, rollout: SampledRollout, reward: float) -> dict[str, float]:
        """Apply one REINFORCE update for ``(rollout, reward)``.

        Returns the merged step statistics. If the rollout has no response
        tokens (e.g. the model produced an empty generation) the update is
        skipped and the returned dict carries ``"skipped": 1.0``.
        """

        clipped = max(-self.reward_clip, min(self.reward_clip, float(reward)))
        self._reward_sum += clipped
        self._recent_rewards.append(clipped)
        self._recent_rewards = self._recent_rewards[-100:]

        if rollout.num_response_tokens == 0:
            self._skipped += 1
            return {
                "reward": clipped,
                "advantage": 0.0,
                "baseline": self._baseline,
                "skipped": 1.0,
                "loss": 0.0,
                "entropy": 0.0,
                "grad_norm": 0.0,
            }

        advantage = clipped - self._baseline
        self._update_baseline(clipped)
        stats = self.policy.reinforce_step(
            rollout,
            advantage,
            lr=self.lr,
            entropy_coef=self.entropy_coef,
            clip_grad=self.clip_grad,
        )
        self._steps += 1
        self._advantage_sum += abs(advantage)
        self._loss_sum += float(stats.get("loss", 0.0))
        self._entropy_sum += float(stats.get("entropy", 0.0))
        self._grad_norm_sum += float(stats.get("grad_norm", 0.0))
        return {
            "reward": clipped,
            "advantage": advantage,
            "baseline": self._baseline,
            "skipped": 0.0,
            **stats,
        }

    def stats(self) -> dict[str, Any]:
        """Return cumulative training statistics for telemetry."""

        steps = self._steps
        return {
            "name": self.name,
            "steps": steps,
            "skipped": self._skipped,
            "baseline": self._baseline,
            "mean_reward": self._reward_sum / max(1, steps + self._skipped),
            "mean_abs_advantage": self._advantage_sum / max(1, steps),
            "mean_loss": self._loss_sum / max(1, steps),
            "mean_entropy": self._entropy_sum / max(1, steps),
            "mean_grad_norm": self._grad_norm_sum / max(1, steps),
            "recent_mean_reward": (
                sum(self._recent_rewards) / len(self._recent_rewards)
                if self._recent_rewards
                else 0.0
            ),
            "lr": self.lr,
            "entropy_coef": self.entropy_coef,
        }

    def _update_baseline(self, reward: float) -> None:
        self._baseline = self.baseline_decay * self._baseline + (1.0 - self.baseline_decay) * reward
