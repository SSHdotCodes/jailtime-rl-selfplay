"""Verifier protocol."""

from __future__ import annotations

from typing import Protocol

from jailtime.schemas import PromptCategory, VerificationResult


class Verifier(Protocol):
    """Provider-independent verifier interface."""

    def verify(
        self,
        prompt: str,
        response: str,
        expected_category: PromptCategory | None = None,
    ) -> VerificationResult:
        """Judge a prompt/response pair."""

