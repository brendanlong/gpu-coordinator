"""What a host's GPUs actually are, in the registry and in every listing."""

from __future__ import annotations

from pathlib import Path

from gpuc.control.cli import main
from gpuc.control.config import HostEntry, load_registry, registry_transaction
from gpuc.control.gpuinfo import GpuInfo, parse_smi, rows, summarize, vram_text
from gpuc.control.status import HostView, JobView, render

SMI_OUTPUT = """\
0, GPU-80646905-50a9-afc1-4375-43ca475b15e4, NVIDIA A40, 46068
1, GPU-83123e65-fe58-7831-6b21-1814b07c25f7, NVIDIA A40, 46068
"""
A40 = "GPU-80646905-50a9-afc1-4375-43ca475b15e4"
A40_TWO = "GPU-83123e65-fe58-7831-6b21-1814b07c25f7"


def test_smi_rows_parse_with_and_without_units() -> None:
    parsed = parse_smi(SMI_OUTPUT)
    assert parsed[A40] == GpuInfo(name="NVIDIA A40", vram_mib=46068, index=0)
    with_units = parse_smi("GPU-aaa, NVIDIA GeForce RTX 3060 Ti, 8192 MiB\n")
    assert with_units["GPU-aaa"] == GpuInfo(name="NVIDIA GeForce RTX 3060 Ti", vram_mib=8192)


def test_junk_lines_are_ignored() -> None:
    assert parse_smi("nvidia-smi: command not found\n\n") == {}


def test_vram_is_reported_as_the_card_is_sold() -> None:
    assert vram_text(46068) == "45 GB"
    assert vram_text(49140) == "48 GB"
    assert vram_text(None) == ""


def test_identical_cards_are_summarised_as_one_group() -> None:
    info = parse_smi(SMI_OUTPUT)
    assert summarize([A40, A40_TWO], info) == "2x NVIDIA A40 45 GB"


def test_mixed_and_unknown_cards_are_still_described() -> None:
    info = parse_smi("GPU-a, NVIDIA A40, 46068\n")
    assert summarize(["GPU-a", "GPU-b"], info) == "1x NVIDIA A40 45 GB, 1x unknown GPU"


def test_rows_keep_the_hosts_own_order() -> None:
    info = parse_smi(SMI_OUTPUT)
    assert rows([A40_TWO, A40], info)[0] == ("1", "NVIDIA A40", "45 GB", A40_TWO)


def entry_with_cards() -> HostEntry:
    return HostEntry(
        name="gpubox",
        kind="ssh",
        ssh="gpubox-ssh",
        gpus=[A40, A40_TWO],
        gpu_info=parse_smi(SMI_OUTPUT),
        driver_version="580.65.06",
    )


def test_host_list_names_the_cards_and_the_driver(control_env: Path, capsys) -> None:
    with registry_transaction() as registry:
        registry.put(entry_with_cards())
    assert main(["host", "list"]) == 0
    out = capsys.readouterr().out
    assert "host gpubox [ssh] gpubox-ssh  gpus 2 (2x NVIDIA A40 45 GB, driver 580.65.06)" in out
    assert f"gpu     [0] NVIDIA A40                   45 GB   {A40}" in out


def test_status_names_the_cards_and_who_holds_them() -> None:
    view = HostView(
        entry=entry_with_cards(),
        reachable=True,
        heartbeat_age_s=2.0,
        owned=[A40, A40_TWO],
        running=[JobView(job_id="20260915-1", status="running", gpus=[A40])],
    )
    text = render(view)
    assert "gpus 1/2 free (driver 580.65.06)" in text
    assert "gpu     [0] busy NVIDIA A40 45 GB" in text
    assert "gpu     [1] free NVIDIA A40 45 GB" in text
    # The UUIDs are `gpuc host list`'s job: this block answers "is there a card
    # free", and a line of hex per card is what made it unreadable.
    assert A40 not in text and A40_TWO not in text


def test_a_host_registered_before_gpu_info_existed_still_lists(control_env: Path, capsys) -> None:
    with registry_transaction() as registry:
        registry.put(HostEntry(name="old", gpus=["GPU-x"]))
    assert main(["host", "list"]) == 0
    out = capsys.readouterr().out
    assert "gpus 1 (1x unknown GPU)" in out
    assert "pkg     unknown shipped from here, never bootstrapped" in out
    assert "GPU-x" in out
    assert load_registry().hosts["old"].gpu_info == {}


def test_a_card_owned_by_index_is_named_from_the_recorded_info() -> None:
    """`host list` is offline, so the index recorded at bootstrap is what it has
    to go on; the host's own numbering is what `gpuc status` shows."""
    info = parse_smi(SMI_OUTPUT)
    assert rows(["1"], info) == [("1", "NVIDIA A40", "45 GB", A40_TWO)]
    assert summarize(["0", "1"], info) == "2x NVIDIA A40 45 GB"
    # An index nothing was ever recorded for is shown, not guessed at.
    assert rows(["5"], info) == [("5", "?", "", "?")]
