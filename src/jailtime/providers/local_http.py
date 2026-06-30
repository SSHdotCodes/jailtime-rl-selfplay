"""Generic local HTTP model provider."""

from __future__ import annotations

import json
from typing import Any
from urllib import error, request

from jailtime.providers.base import ProviderError
from jailtime.schemas import Message, ModelResponse


class LocalHTTPProvider:
    """Adapter for local HTTP model servers.

    The server may return ``{"content": ...}``, ``{"response": ...}``, or an
    OpenAI-compatible ``choices`` response.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        model: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not endpoint:
            raise ValueError("endpoint is required for LocalHTTPProvider")
        self.endpoint = endpoint
        self.model = model
        self.timeout_seconds = timeout_seconds

    def complete(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        """Post messages to the configured local endpoint."""

        payload = {
            "messages": [message.model_dump() for message in messages],
            **kwargs,
        }
        if self.model:
            payload["model"] = self.model
        raw = self._post_json(payload)
        content = self._extract_content(raw)
        return ModelResponse(content=content, model=self.model, raw=raw)

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.URLError as exc:
            raise ProviderError(f"Local HTTP provider request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ProviderError("Local HTTP provider response was not valid JSON") from exc

    @staticmethod
    def _extract_content(raw: dict[str, Any]) -> str:
        if isinstance(raw.get("content"), str):
            return raw["content"]
        if isinstance(raw.get("response"), str):
            return raw["response"]
        try:
            return raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                "Local HTTP provider response did not include text content"
            ) from exc
