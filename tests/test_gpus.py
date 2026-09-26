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


def test_the_table_carries_index_uuid_name_and_memory() -> None:
    (first, _second) = gpus.list_gpus(fake_smi())
    assert first == gpus.Gpu(0, FAKE_GPUS[0], "Fake A40", 46068)
    with_units = gpus.parse_table("0, GPU-a, NVIDIA A40, 46068 MiB\n1, GPU-b, NVIDIA A40, 46068\n")
    assert with_units == [
        gpus.Gpu(0, "GPU-a", "NVIDIA A40", 46068),
        gpus.Gpu(1, "GPU-b", "NVIDIA A40", 46068),
    ]
    assert gpus.parse_table("nvidia-smi: command not found\n\n") == []
    with pytest.raises(gpus.GpuError, match="no card rows"):
        gpus.list_gpus(lambda args: "No devices were found\n")


def test_resolve_names_each_card_once_by_index_or_uuid() -> None:
    table = gpus.list_gpus(fake_smi())
    assert gpus.resolve([FAKE_GPUS[0]], table).owned == [FAKE_GPUS[0]]
    assert gpus.resolve([], table) == gpus.Resolution([], [], [], [], [])
    assert gpus.resolve(["1"], table).owned == [FAKE_GPUS[1]]
    assert gpus.resolve(["0", FAKE_GPUS[1]], table).owned == FAKE_GPUS


def test_an_absent_uuid_is_missing_exactly_as_an_absent_index_is() -> None:
    """A card the driver stopped reporting is never handed out to fail inside
    a job: the rule is the same for both spellings, live and offline."""
    table = gpus.list_gpus(fake_smi())
    cards = gpus.resolve(["0", "7", "GPU-stale"], table)
    assert cards.owned == [FAKE_GPUS[0]]
    assert cards.missing == ["7", "GPU-stale"]
    assert f"0={FAKE_GPUS[0]}" in gpus.describe_table(table)


def test_an_index_and_its_own_uuid_are_one_card_named_twice() -> None:
    cards = gpus.resolve(["0", FAKE_GPUS[0]], gpus.list_gpus(fake_smi()))
    assert cards.owned == [FAKE_GPUS[0]]
    assert cards.duplicates == [FAKE_GPUS[0]]


def test_a_card_both_owned_and_shared_is_a_duplicate_and_owning_wins() -> None:
    table = gpus.list_gpus(fake_smi())
    cards = gpus.resolve(["0"], table, shared=[FAKE_GPUS[0], "1", "1"])
    assert cards.owned == [FAKE_GPUS[0]]
    assert cards.shared == [FAKE_GPUS[1]]
    assert cards.duplicates == [FAKE_GPUS[0], "1"]
    assert gpus.resolve(["0"], table, shared=["9"]).shared_missing == ["9"]


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
    with pytest.raises(gpus.GpuError, match="no card rows"):
        gpus.list_gpus(garbage_smi(value))


def test_a_short_row_is_a_gpu_error_not_a_crash() -> None:
    with pytest.raises(gpus.GpuError, match="no card rows"):
        gpus.list_gpus(lambda args: "0\n")


def test_mean_utilization_propagates_the_parse_error() -> None:
    with pytest.raises(gpus.GpuError):
        gpus.mean_utilization(["GPU-x"], garbage_smi("[N/A]"))


def test_no_samples_for_real_gpus_is_an_error_not_zero_percent() -> None:
    """0% is a claim about an idle card; "nvidia-smi said nothing" is not.

    Returning 0.0 here showed jobs that were perfectly busy as idle.
    """
    with pytest.raises(gpus.GpuError, match="failed sample"):
        gpus.mean_utilization(["GPU-x"], lambda args: "")


def test_a_renumbered_box_costs_the_cards_that_moved_not_the_rest() -> None:
    table = gpus.parse_table("0, GPU-zero, , \n1, GPU-one, , \n")
    cards = gpus.resolve(["0", "7", "GPU-gone"], table)
    assert (cards.owned, cards.missing) == (["GPU-zero"], ["7", "GPU-gone"])


# -- shared GPUs: is anybody else on this card? --------------------------------


def test_a_card_with_no_memory_and_no_work_is_unused() -> None:
    unused, in_use = gpus.unused_gpus(FAKE_GPUS, fake_smi())
    assert unused == FAKE_GPUS
    assert in_use == {}


def test_memory_alone_is_enough_to_call_a_card_in_use() -> None:
    """Somebody's CUDA context holds hundreds of MiB between steps, so the
    card at 0% util is still theirs. Memory is the half that decides."""
    smi = fake_smi(memory_used={FAKE_GPUS[0]: 512.0})
    unused, in_use = gpus.unused_gpus(FAKE_GPUS, smi)
    assert unused == [FAKE_GPUS[1]]
    assert "512 MiB" in in_use[FAKE_GPUS[0]]
    assert "0% util" in in_use[FAKE_GPUS[0]]


def test_utilization_alone_is_enough_too() -> None:
    smi = fake_smi(utilization={FAKE_GPUS[1]: 37.0})
    unused, in_use = gpus.unused_gpus(FAKE_GPUS, smi)
    assert unused == [FAKE_GPUS[0]]
    assert "37% util" in in_use[FAKE_GPUS[1]]


def test_a_reading_that_cannot_be_read_counts_as_in_use() -> None:
    """This decides whether to run on somebody else's GPU, so every way of not
    knowing has to count against: `[N/A]`, a card nvidia-smi skipped, and
    nvidia-smi failing outright."""

    def unsupported(args: list[str]) -> str:
        return f"{FAKE_GPUS[0]}, [N/A], [Not Supported]\n"

    unused, in_use = gpus.unused_gpus(FAKE_GPUS, unsupported)
    assert unused == []
    assert in_use[FAKE_GPUS[0]] == "? MiB, ?% util"
    assert "nothing about it" in in_use[FAKE_GPUS[1]]

    def broken(args: list[str]) -> str:
        raise gpus.GpuError("nvidia-smi exited 9")

    unused, in_use = gpus.unused_gpus(FAKE_GPUS, broken)
    assert unused == []
    assert all("nvidia-smi exited 9" in why for why in in_use.values())


def test_asking_about_no_cards_is_not_an_error() -> None:
    assert gpus.unused_gpus([], fake_smi()) == ([], {})


def test_a_snapshot_lists_every_card_with_its_reading_in_one_call() -> None:
    calls: list[list[str]] = []
    inner = fake_smi(utilization={FAKE_GPUS[0]: 97.0}, memory_used={FAKE_GPUS[0]: 21504.0})

    def smi(args: list[str]) -> str:
        calls.append(args)
        return inner(args)

    table, usage = gpus.snapshot(smi)
    assert len(calls) == 1
    assert table == gpus.list_gpus(fake_smi())
    assert usage[FAKE_GPUS[0]] == gpus.Usage(FAKE_GPUS[0], 21504.0, 97.0)
    assert usage[FAKE_GPUS[1]].unused


def test_a_snapshot_row_that_will_not_parse_costs_only_its_own_card() -> None:
    def odd(args: list[str]) -> str:
        return f"0, {FAKE_GPUS[0]}, A40, 46068, [N/A], 12\n1, {FAKE_GPUS[1]}, A40, 46068, 3\n"

    table, usage = gpus.snapshot(odd)
    assert [gpu.uuid for gpu in table] == FAKE_GPUS
    assert usage == {FAKE_GPUS[0]: gpus.Usage(FAKE_GPUS[0], None, 12.0)}
    with pytest.raises(gpus.GpuError, match="no card rows"):
        gpus.snapshot(lambda args: "No devices were found\n")
