"""Real-time reinforcement-learning self-play for adversarial safety training.

This subpackage implements the online RL closed loop:

* An **attacker** policy model samples adversarial probes.
* A **defender** policy model samples responses.
* A **verifier** judges the defender's response and produces a scalar reward
  for both policies.
* ``REINFORCETrainer`` turns each episode's reward into a real-time,
  full-parameter policy-gradient update on the model weights -- no LoRA, no
  offline SFT step. The defender's final weights are saved as safetensors.

The policy interface (``PolicyModel``) is backend-agnostic so the loop can be
unit-tested with ``MockPolicyModel``. The torch-backed implementation
(``TorchPolicyModel``) only imports ``torch``/``transformers`` when
instantiated, keeping the default install lightweight.

Use this only on models you own or are explicitly authorized to modify.
"""

from jailtime.rl.agents import RLAttackerAgent, RLDefenderAgent, RLModelVerifier
from jailtime.rl.policy import MockPolicyModel, PolicyModel, SampledRollout
from jailtime.rl.reinforce import REINFORCETrainer
from jailtime.rl.selfplay import RLSelfPlayOrchestrator
from jailtime.rl.verifier_calibration import (
    CalibrationProbe,
    VerifierCalibrationTrainer,
    default_calibration_probes,
)

__all__ = [
    "CalibrationProbe",
    "MockPolicyModel",
    "PolicyModel",
    "REINFORCETrainer",
    "RLAttackerAgent",
    "RLDefenderAgent",
    "RLModelVerifier",
    "RLSelfPlayOrchestrator",
    "SampledRollout",
    "VerifierCalibrationTrainer",
    "default_calibration_probes",
]
