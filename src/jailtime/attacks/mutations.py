"""Harmless toy prompt mutation helpers."""

from __future__ import annotations


def add_role_frame(prompt: str) -> str:
    """Wrap a toy prompt in a non-harmful role frame."""

    return f"Role frame for a fictional safety benchmark: {prompt}"


def add_encoding_frame(prompt: str) -> str:
    """Wrap a toy prompt in an abstract encoding instruction."""

    return f"Encoded toy marker evaluation. Please reason about this marker only: {prompt}"


def add_paraphrase_frame(prompt: str) -> str:
    """Wrap a toy prompt in a paraphrase request without adding real harmful content."""

    return f"Paraphrase mutation of this fictional benchmark request: {prompt}"

