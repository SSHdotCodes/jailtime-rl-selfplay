"""Verifier agent protocol."""

from __future__ import annotations

from typing import Protocol

from jailtime.schemas import PromptCategory, VerificationResult


class Verifier(Protocol):
    """Agent that judges safety and helpfulness of a model response."""

    def verify(
        self,
        prompt: str,
        response: str,
        expected_category: PromptCategory | None = None,
    ) -> VerificationResult:
        """Return a structured verification result."""

