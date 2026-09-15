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


def garbage_smi(value: str):  # type: ignore[no-untyped-def]
    def run(args: list[str]) -> str:
        fields = next(a for a in args if a.startswith("--query-gpu=")).split("=", 1)[1].split(",")
        return (
            ", ".join(value if f == "utilization.gpu" or f == "index" else "GPU-x" for f in fields)
            + "\n"
        )

    return run


@pytest.mark.parametrize("value", ["[N/A]", "[Not Supported]", "", "n/a"])
def test_unparsable_utilization_is_a_gpu_error(value: str) -> None:
    with pytest.raises(gpus.GpuError) as excinfo:
        gpus.sample_utilization(["GPU-x"], garbage_smi(value))
    assert "not a number" in str(excinfo.value)


@pytest.mark.parametrize("value", ["[N/A]", "garbage"])
def test_an_unparsable_index_is_a_gpu_error(value: str) -> None:
    with pytest.raises(gpus.GpuError, match="not a number"):
        gpus.list_gpus(garbage_smi(value))


def test_a_short_row_is_a_gpu_error_not_a_crash() -> None:
    with pytest.raises(gpus.GpuError, match="expected 2"):
        gpus.list_gpus(lambda args: "0\n")


def test_mean_utilization_propagates_the_parse_error() -> None:
    with pytest.raises(gpus.GpuError):
        gpus.mean_utilization(["GPU-x"], garbage_smi("[N/A]"))


def test_no_samples_for_real_gpus_is_an_error_not_zero_percent() -> None:
    """0% is a claim about an idle card; "nvidia-smi said nothing" is not.

    Returning 0.0 here fed the low-util watchdog a floor-breaking sample every
    tick and killed jobs that were perfectly busy.
    """
    with pytest.raises(gpus.GpuError, match="failed sample"):
        gpus.mean_utilization(["GPU-x"], lambda args: "")


def test_uuid_ownership_resolves_to_itself_without_asking_nvidia_smi() -> None:
    """Every pod, and most boxes, own UUIDs: an exec per dispatch pass to look
    up the identity mapping would be pure cost."""

    def refuse(_args: list[str]) -> str:
        raise AssertionError("nvidia-smi should not be run for UUID ownership")

    assert gpus.resolve_owned(["GPU-a", "GPU-b"], refuse) == (["GPU-a", "GPU-b"], [])
    assert gpus.resolve_owned([], refuse) == ([], [])


def test_owned_indices_resolve_to_the_uuids_the_driver_reports_now() -> None:
    smi = fake_smi(["GPU-zero", "GPU-one", "GPU-two"])
    assert gpus.resolve_owned(["1", "2"], smi) == (["GPU-one", "GPU-two"], [])
    # A mix, and the same card named twice, is still each card once.
    assert gpus.resolve_owned(["0", "GPU-zero", "GPU-two"], smi) == (
        ["GPU-zero", "GPU-two"],
        [],
    )


def test_an_owned_entry_the_host_cannot_see_is_reported_unavailable() -> None:
    """A renumbered shared box must cost the cards that moved, not the rest."""
    smi = fake_smi(["GPU-zero", "GPU-one"])
    assert gpus.resolve_owned(["0", "7", "GPU-gone"], smi) == (["GPU-zero"], ["7", "GPU-gone"])
    assert gpus.index_uuids(smi) == {"0": "GPU-zero", "1": "GPU-one"}
    assert "0=GPU-zero" in gpus.describe_table(smi)
