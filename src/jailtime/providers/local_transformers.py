"""Local Hugging Face Transformers provider with optional Apple MPS acceleration."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from jailtime.providers.base import ProviderError
from jailtime.providers.device import DeviceSelectionError, resolve_torch_device
from jailtime.schemas import Message, ModelResponse


class LocalTransformersProvider:
    """Run a local causal language model through PyTorch/Transformers.

    This provider is optional and only imports ``torch`` and ``transformers``
    when instantiated. Use ``device="auto"`` or ``device="mps"`` to run on
    Apple Silicon MPS when PyTorch reports it is available.
    """

    def __init__(
        self,
        *,
        model: str,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        local_files_only: bool = True,
        trust_remote_code: bool = False,
        model_kwargs: dict[str, Any] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if not model:
            raise ValueError("model is required for LocalTransformersProvider")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")

        self.model_name = model
        self.requested_device = device
        self.device = self._resolve_device(device)
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code
        self.model_kwargs = model_kwargs or {}
        self.generation_kwargs = generation_kwargs or {}

        self._torch = _import_optional("torch")
        transformers = _import_optional("transformers")
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            model,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        model_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
            **self.model_kwargs,
        }
        if self.device == "mps" and "attn_implementation" not in model_kwargs:
            model_kwargs["attn_implementation"] = "eager"
        torch_dtype = self._resolve_dtype(dtype)
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        self._model = self._load_model(transformers, model, model_kwargs)
        self._model.to(self.device)
        self._model.eval()

    def complete(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        """Generate a completion for chat messages."""

        prompt_text = self._format_messages(messages)
        inputs = self._tokenizer(prompt_text, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        generation_kwargs = {
            "max_new_tokens": kwargs.pop("max_new_tokens", self.max_new_tokens),
            "do_sample": self.temperature > 0.0,
            **self.generation_kwargs,
            **kwargs,
        }
        if self.temperature > 0.0:
            generation_kwargs["temperature"] = kwargs.pop("temperature", self.temperature)

        with self._torch.inference_mode():
            generated = self._model.generate(**inputs, **generation_kwargs)

        input_length = inputs["input_ids"].shape[-1]
        new_tokens = generated[0][input_length:]
        content = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return ModelResponse(
            content=content,
            model=self.model_name,
            metadata={
                "provider": "local_transformers",
                "device": self.device,
                "dtype": self.dtype,
                "local_files_only": self.local_files_only,
            },
        )

    def _format_messages(self, messages: list[Message]) -> str:
        chat = [message.model_dump() for message in messages]
        apply_chat_template = getattr(self._tokenizer, "apply_chat_template", None)
        if callable(apply_chat_template):
            return str(
                apply_chat_template(
                    chat,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        transcript = "\n".join(f"{message.role}: {message.content}" for message in messages)
        return f"{transcript}\nassistant:"

    def _resolve_dtype(self, dtype: str) -> Any:
        normalized = dtype.strip().lower()
        if normalized == "auto":
            return "auto"
        if normalized in {"none", "default"}:
            return None
        mapping = {
            "float16": "float16",
            "fp16": "float16",
            "bfloat16": "bfloat16",
            "bf16": "bfloat16",
            "float32": "float32",
            "fp32": "float32",
        }
        torch_attr = mapping.get(normalized)
        if torch_attr is None:
            raise ValueError(
                f"Unsupported dtype '{dtype}'. Expected auto, float16, bfloat16, float32, or none."
            )
        return getattr(self._torch, torch_attr)

    @staticmethod
    def _load_model(transformers: Any, model: str, model_kwargs: dict[str, Any]) -> Any:
        errors: list[str] = []
        for class_name in (
            "AutoModelForCausalLM",
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
        ):
            model_class = getattr(transformers, class_name, None)
            if model_class is None:
                continue
            try:
                return model_class.from_pretrained(model, **model_kwargs)
            except Exception as exc:
                errors.append(f"{class_name}: {exc}")
        joined = "\n".join(errors) or "no compatible AutoModel class was available"
        raise ProviderError(f"Could not load local Transformers model {model!r}:\n{joined}")

    @staticmethod
    def _resolve_device(device: str) -> str:
        try:
            return resolve_torch_device(device)
        except DeviceSelectionError as exc:
            raise ProviderError(str(exc)) from exc


def _import_optional(module_name: str) -> Any:
    try:
        return import_module(module_name)
    except ImportError as exc:
        raise ProviderError(
            "LocalTransformersProvider requires optional local model dependencies. "
            "Install them with: python3 -m pip install 'jailtime[local]'"
        ) from exc
