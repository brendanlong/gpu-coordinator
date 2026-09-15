from __future__ import annotations

import pytest

from gpuc.host import gpus
from tests.conftest import FAKE_GPUS, fake_smi


def test_list_gpus_parses_index_and_uuid() -> None:
    listed = gpus.list_gpus(fake_smi())
    assert [g.index for g in listed] == [0, 1]
    assert [g.uuid for g in listed] == FAKE_GPUS


def test_driver_version() -> None:
    assert gpus.driver_version(fake_smi()) == "580.173.02"


def test_assert_uuids_present_passes_and_fails_clearly() -> None:
    gpus.assert_uuids_present([FAKE_GPUS[0]], fake_smi())
    gpus.assert_uuids_present([], fake_smi())
    with pytest.raises(gpus.GpuError) as excinfo:
        gpus.assert_uuids_present(["GPU-stale"], fake_smi())
    assert "GPU-stale" in str(excinfo.value)
    assert FAKE_GPUS[0] in str(excinfo.value)


def test_sample_utilization_filters_to_requested_uuids() -> None:
    smi = fake_smi(utilization={FAKE_GPUS[0]: 91.0, FAKE_GPUS[1]: 3.0})
    assert gpus.sample_utilization([FAKE_GPUS[0]], smi) == {FAKE_GPUS[0]: 91.0}
    assert gpus.mean_utilization(FAKE_GPUS, smi) == pytest.approx(47.0)
    assert gpus.mean_utilization([], smi) == 0.0


def test_missing_nvidia_smi_raises_gpu_error() -> None:
    def missing(args: list[str]) -> str:
        raise FileNotFoundError("nvidia-smi")

    with pytest.raises(FileNotFoundError):
        gpus.list_gpus(missing)


def test_real_runner_reports_a_clear_error_when_binary_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "/nonexistent")
    with pytest.raises(gpus.GpuError) as excinfo:
        gpus.run_nvidia_smi(["--query-gpu=uuid", "--format=csv,noheader"])
    assert "nvidia-smi not found" in str(excinfo.value)
