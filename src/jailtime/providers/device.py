"""Torch device detection helpers, including Apple MPS support."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType


class DeviceSelectionError(RuntimeError):
    """Raised when a requested accelerator is not available."""


@dataclass(frozen=True)
class TorchDeviceStatus:
    """Detected PyTorch accelerator status."""

    torch_installed: bool
    mps_built: bool
    mps_available: bool
    cuda_available: bool
    selected_device: str | None
    reason: str


def probe_torch_devices(torch_module: ModuleType | object | None = None) -> TorchDeviceStatus:
    """Return PyTorch device status without raising when torch is absent."""

    torch = torch_module or _try_import_torch()
    if torch is None:
        return TorchDeviceStatus(
            torch_installed=False,
            mps_built=False,
            mps_available=False,
            cuda_available=False,
            selected_device=None,
            reason=(
                "PyTorch is not installed. Install with: "
                "python3 -m pip install 'jailtime[local]'"
            ),
        )

    mps_built = _mps_is_built(torch)
    mps_available = _mps_is_available(torch)
    cuda_available = _cuda_is_available(torch)
    selected = "mps" if mps_available else "cuda" if cuda_available else "cpu"
    reason = _selection_reason(selected, mps_built, mps_available, cuda_available)
    return TorchDeviceStatus(
        torch_installed=True,
        mps_built=mps_built,
        mps_available=mps_available,
        cuda_available=cuda_available,
        selected_device=selected,
        reason=reason,
    )


def resolve_torch_device(
    requested: str = "auto",
    torch_module: ModuleType | object | None = None,
) -> str:
    """Resolve a requested torch device.

    ``auto`` prefers ``mps`` when available, then ``cuda``, then ``cpu``.
    """

    torch = torch_module or _try_import_torch()
    if torch is None:
        raise DeviceSelectionError(
            "PyTorch is not installed. Install local acceleration support with: "
            "python3 -m pip install 'jailtime[local]'"
        )

    normalized = requested.strip().lower()
    if normalized == "auto":
        status = probe_torch_devices(torch)
        if not status.selected_device:
            raise DeviceSelectionError(status.reason)
        return status.selected_device
    if normalized == "mps":
        if not _mps_is_built(torch):
            raise DeviceSelectionError(
                "This PyTorch build does not include MPS support. Install a recent macOS "
                "PyTorch build with: python3 -m pip install --upgrade torch"
            )
        if not _mps_is_available(torch):
            raise DeviceSelectionError(
                "MPS was requested, but it is not available on this machine. "
                "Use device: auto or device: cpu."
            )
        return "mps"
    if normalized == "cuda":
        if not _cuda_is_available(torch):
            raise DeviceSelectionError(
                "CUDA was requested, but torch.cuda.is_available() is false."
            )
        return "cuda"
    if normalized == "cpu":
        return "cpu"
    raise DeviceSelectionError(
        f"Unsupported torch device '{requested}'. Expected one of: auto, mps, cuda, cpu."
    )


def _try_import_torch() -> ModuleType | None:
    try:
        return import_module("torch")
    except ImportError:
        return None


def _mps_is_built(torch: ModuleType | object) -> bool:
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    is_built = getattr(mps, "is_built", None)
    return bool(callable(is_built) and is_built())


def _mps_is_available(torch: ModuleType | object) -> bool:
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    is_available = getattr(mps, "is_available", None)
    return bool(callable(is_available) and is_available())


def _cuda_is_available(torch: ModuleType | object) -> bool:
    cuda = getattr(torch, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    return bool(callable(is_available) and is_available())


def _selection_reason(
    selected: str,
    mps_built: bool,
    mps_available: bool,
    cuda_available: bool,
) -> str:
    if selected == "mps":
        return "Using Apple MPS acceleration."
    if selected == "cuda":
        return "Using CUDA acceleration."
    if not mps_built:
        return "Using CPU because this PyTorch build does not include MPS support."
    if not mps_available:
        return "Using CPU because MPS is built but not available on this machine."
    if not cuda_available:
        return "Using CPU because no accelerator is available."
    return "Using CPU."
