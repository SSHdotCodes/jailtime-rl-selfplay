"""Hybrid verifier that combines rules with a model verifier."""

from __future__ import annotations

from jailtime.schemas import PromptCategory, VerificationResult
from jailtime.verifiers.base import Verifier
from jailtime.verifiers.rule_based import RuleBasedVerifier


class HybridVerifier:
    """Use rule-based verification first, then defer to a model when uncertain."""

    def __init__(
        self,
        model_verifier: Verifier,
        *,
        rule_verifier: RuleBasedVerifier | None = None,
        rule_confidence_threshold: float = 0.8,
    ) -> None:
        self.rule_verifier = rule_verifier or RuleBasedVerifier()
        self.model_verifier = model_verifier
        self.rule_confidence_threshold = rule_confidence_threshold

    def verify(
        self,
        prompt: str,
        response: str,
        expected_category: PromptCategory | None = None,
    ) -> VerificationResult:
        """Return rule result when confident, otherwise model result."""

        rule_result = self.rule_verifier.verify(prompt, response, expected_category)
        if rule_result.confidence >= self.rule_confidence_threshold:
            return rule_result
        model_result = self.model_verifier.verify(prompt, response, expected_category)
        model_result.metadata["hybrid_rule_result"] = rule_result.model_dump(mode="json")
        return model_result

