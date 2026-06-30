import pytest

from jailtime.providers.device import (
    DeviceSelectionError,
    probe_torch_devices,
    resolve_torch_device,
)


class FakeMPSAvailable:
    @staticmethod
    def is_built() -> bool:
        return True

    @staticmethod
    def is_available() -> bool:
        return True


class FakeMPSBuiltUnavailable:
    @staticmethod
    def is_built() -> bool:
        return True

    @staticmethod
    def is_available() -> bool:
        return False


class FakeCudaUnavailable:
    @staticmethod
    def is_available() -> bool:
        return False


class FakeBackendsMPSAvailable:
    mps = FakeMPSAvailable()


class FakeBackendsMPSUnavailable:
    mps = FakeMPSBuiltUnavailable()


class FakeTorchMPSAvailable:
    backends = FakeBackendsMPSAvailable()
    cuda = FakeCudaUnavailable()


class FakeTorchMPSUnavailable:
    backends = FakeBackendsMPSUnavailable()
    cuda = FakeCudaUnavailable()


def test_auto_device_prefers_mps_when_available() -> None:
    assert resolve_torch_device("auto", torch_module=FakeTorchMPSAvailable()) == "mps"


def test_probe_reports_mps_status() -> None:
    status = probe_torch_devices(torch_module=FakeTorchMPSAvailable())

    assert status.torch_installed is True
    assert status.mps_built is True
    assert status.mps_available is True
    assert status.selected_device == "mps"


def test_requested_mps_requires_availability() -> None:
    with pytest.raises(DeviceSelectionError, match="MPS was requested"):
        resolve_torch_device("mps", torch_module=FakeTorchMPSUnavailable())
