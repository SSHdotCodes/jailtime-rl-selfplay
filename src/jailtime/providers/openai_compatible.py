"""OpenAI-compatible HTTP provider with no SDK dependency."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, request

from jailtime.providers.base import ProviderError
from jailtime.schemas import Message, ModelResponse


class OpenAICompatibleProvider:
    """Minimal adapter for OpenAI-compatible ``/chat/completions`` endpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key_env: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required for OpenAICompatibleProvider")
        if not model:
            raise ValueError("model is required for OpenAICompatibleProvider")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds

    def complete(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        """Call the configured OpenAI-compatible endpoint."""

        payload = {
            "model": self.model,
            "messages": [message.model_dump() for message in messages],
            **kwargs,
        }
        response_data = self._post_json(f"{self.base_url}/chat/completions", payload)
        try:
            content = response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                "OpenAI-compatible response missing choices[0].message.content"
            ) from exc
        return ModelResponse(content=content, model=self.model, raw=response_data)

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key_env:
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise ProviderError(f"Environment variable is not set: {self.api_key_env}")
            headers["Authorization"] = f"Bearer {api_key}"
        http_request = request.Request(url, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.URLError as exc:
            raise ProviderError(f"Provider request failed for {url}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ProviderError(f"Provider response was not valid JSON for {url}") from exc
