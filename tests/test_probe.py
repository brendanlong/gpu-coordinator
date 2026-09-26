from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from gpuc.control.probe import parse_probe, probe_host, probe_script
from gpuc.control.transport import CommandResult
from tests.conftest import host_entry

PROBE_SCRIPT = probe_script()

SAMPLE = """===system===
Linux 6.8.0 x86_64
user=claude home=/home/claude shell=/bin/bash
===driver===
580.173.02
===gpus===
0, GPU-2a4bad3b-9fe3-7031-914d-384254e92908, NVIDIA GeForce RTX 3060 Ti, 8192 MiB
1, GPU-deadbeef-0000-0000-0000-000000000000, NVIDIA A40, 46068 MiB
===disk===
/dev/nvme0n1  1.8T  400G  1.3T  24% /home
===killuserprocesses===
KillUserProcesses=yes
===systemd_scope===
no
===linger===
no
===uv===
not installed
===python3===
/usr/bin/python3 3.13.1
"""

BARE = """===system===
Linux 6.8.0 x86_64
===driver===
nvidia-smi not found
===gpus===
nvidia-smi not found
===disk===
/dev/sda1  50G  10G  40G  20% /
===killuserprocesses===
loginctl unavailable
===systemd_scope===
no
===uv===
not installed
===python3===
not installed
"""


def test_probe_script_is_posix_sh_with_no_gpuc_dependency() -> None:
    # Nothing in the script runs gpuc code: probe has to work on a host where
    # nothing is installed yet. What needs the tool's own rules is health's.
    assert "-m gpuc" not in PROBE_SCRIPT
    assert "bash" not in PROBE_SCRIPT
    assert "download" not in PROBE_SCRIPT and "uv_cache" not in PROBE_SCRIPT


def test_parse_finds_every_section() -> None:
    report = parse_probe("gpubox", SAMPLE)
    assert report.sections["driver"] == "580.173.02"
    assert report.sections["systemd_scope"] == "no"


def test_the_gpus_section_is_read_by_the_hosts_own_parser() -> None:
    table = parse_probe("gpubox", SAMPLE).table
    assert [gpu.index for gpu in table] == [0, 1]
    assert table[0].uuid == "GPU-2a4bad3b-9fe3-7031-914d-384254e92908"
    assert (table[1].name, table[1].memory_mib) == ("NVIDIA A40", 46068)


def test_render_warns_about_logind_and_missing_uv() -> None:
    rendered = parse_probe("gpubox", SAMPLE).render()
    assert (
        "NVIDIA GeForce RTX 3060 Ti  8192 MiB  GPU-2a4bad3b-9fe3-7031-914d-384254e92908" in rendered
    )
    assert "logind kills user processes at logout" in rendered
    assert "gpuc host bootstrap gpubox" in rendered


def test_a_host_with_nothing_still_renders() -> None:
    report = parse_probe("bare", BARE)
    assert not report.has_nvidia_smi
    assert report.table == []
    rendered = report.render()
    assert "no nvidia-smi, so this host cannot run jobs" in rendered


OVERLAY_HOME = """===system===
Linux 5.15.0 x86_64
user=brendan home=/home/brendan shell=/bin/zsh
===disk===
overlay  1.8T  1.5T  300G  84% /
===home_fs===
overlay overlay 1887436800 1572864000 314572800 84% /
===uv===
not installed
"""

DISK_HOME = """===home_fs===
/dev/nvme0n1p2 ext4 1887436800 400000000 1400000000 23% /home
"""


def test_the_script_asks_df_for_the_filesystem_type() -> None:
    assert 'df -T "$HOME"' in PROBE_SCRIPT
    assert "stat -f" in PROBE_SCRIPT


def test_an_overlay_home_is_reported_but_never_argued_with() -> None:
    """An ephemeral queue is how gpuc works -- a rented pod has nowhere durable
    to put one either -- so the probe reports the filesystem and stops there."""
    report = parse_probe("gpubox", OVERLAY_HOME)
    assert report.home_fs_type == "overlay"
    assert report.home_is_overlay
    rendered = report.render()
    # Named on its own line: `disk:` here is an overlay too, so a bare
    # substring would pass with home_fs dropped from the report entirely.
    assert "home_fs: overlay overlay" in rendered
    assert "persistent-root" not in rendered
    assert "wiped on every restart" not in rendered


def test_a_real_filesystem_is_reported_as_itself() -> None:
    report = parse_probe("desk", DISK_HOME)
    assert report.home_fs_type == "ext4"
    assert not report.home_is_overlay


def test_the_stat_fallback_form_is_parsed_too() -> None:
    report = parse_probe("gpubox", "===home_fs===\n/home/brendan overlayfs\n")
    assert (report.home_fs_type, report.home_is_overlay) == ("overlayfs", True)


def test_a_host_that_answered_nothing_is_not_called_an_overlay() -> None:
    report = parse_probe("gpubox", SAMPLE)
    assert report.home_fs_type is None
    assert not report.home_is_overlay


TI = "GPU-2a4bad3b-9fe3-7031-914d-384254e92908"
A40 = "GPU-deadbeef-0000-0000-0000-000000000000"


def test_only_the_assigned_gpus_are_shown_by_default() -> None:
    """A shared box lists every card; only some of them are ours."""
    rendered = parse_probe("gpubox", SAMPLE, None, ["1"]).render()
    assert "gpus: 1 of 2 assigned to gpubox (--all-gpus lists the rest)" in rendered
    assert A40 in rendered
    assert TI not in rendered
    assert "1 of this host's 2 GPUs are neither assigned to gpubox" in rendered
    assert "gpuc host set gpubox --gpus <list>" in rendered


def test_all_gpus_shows_the_whole_box_with_ours_marked() -> None:
    rendered = parse_probe("gpubox", SAMPLE, None, ["1"]).render(all_gpus=True)
    assert f"NVIDIA GeForce RTX 3060 Ti  8192 MiB  {TI}\n" in rendered + "\n"
    assert f"NVIDIA A40  46068 MiB  {A40}  (assigned)" in rendered
    assert "--all-gpus lists the rest" not in rendered


def test_an_assignment_by_uuid_is_matched_as_well_as_by_index() -> None:
    report = parse_probe("gpubox", SAMPLE, None, [A40])
    assert [gpu.uuid for gpu in report.owned_rows] == [A40]
    assert report.owned_missing == []


def test_owning_every_card_needs_no_note_and_hides_nothing() -> None:
    rendered = parse_probe("gpubox", SAMPLE, None, ["0", "1"]).render()
    assert "gpus: 2 of 2 assigned to gpubox\n" in rendered
    assert TI in rendered and A40 in rendered
    assert "not assigned" not in rendered


def test_all_gpus_marks_nothing_when_every_card_is_ours() -> None:
    """A mark on every line distinguishes nothing."""
    assert "(assigned)" not in parse_probe("gpubox", SAMPLE, None, ["0", "1"]).render(all_gpus=True)


def test_a_host_with_no_list_owns_every_card_it_sees() -> None:
    report = parse_probe("gpubox", SAMPLE)
    rendered = report.render()
    assert "  gpus: 2 of 2 assigned to gpubox\n" in rendered
    assert TI in rendered and A40 in rendered
    assert "no GPUs are assigned" not in rendered
    assert report.owned_missing == []
    assert report.document()["assigned_gpus"] is None


def test_a_host_assigned_none_sees_every_card_and_is_told_to_assign_some() -> None:
    rendered = parse_probe("gpubox", SAMPLE, None, []).render()
    assert "  gpus:\n" in rendered
    assert TI in rendered and A40 in rendered
    assert "no GPUs are assigned to gpubox, so only jobs asking for none can run on it" in rendered


def test_an_assigned_card_the_host_cannot_see_is_called_out() -> None:
    """The failure this catches: jobs queue behind a card that is not there."""
    report = parse_probe("gpubox", SAMPLE, None, ["1", "7"])
    assert report.owned_missing == ["7"]
    assert "assigned but not present on this host: 7" in report.render()


def test_a_host_with_no_driver_calls_nothing_missing() -> None:
    assert parse_probe("bare", BARE, None, ["0"]).owned_missing == []


def test_the_document_flags_every_card_assigned_or_not() -> None:
    document = parse_probe("gpubox", SAMPLE, None, ["1"]).document()
    assert [(gpu["uuid"], gpu["assigned"]) for gpu in document["gpus"]] == [
        (TI, False),
        (A40, True),
    ]
    assert document["assigned_gpus"] == ["1"]
    assert document["assigned_missing"] == []


class OneAnswerTransport:
    """Says the same thing to every command: the probe only asks once."""

    host = "gpubox"

    def __init__(self, output: str) -> None:
        self.output = output

    def run(self, command: str, *, timeout: float = 120.0, check: bool = True) -> CommandResult:
        return CommandResult(self.host, ["ssh", command], 0, self.output, "")

    def put_file(self, content: str | bytes, remote_path: str, mode: int = 0o600) -> None: ...

    def rsync(
        self,
        local_root: Path,
        remote_path: str,
        files: Sequence[str] | None = None,
        excludes: Sequence[str] = (),
    ) -> CommandResult:
        return CommandResult(self.host, ["rsync"], 0, "", "")

    def pull(self, remote_root: str, local_root: Path, files: Sequence[str]) -> CommandResult:
        return CommandResult(self.host, ["rsync"], 0, "", "")

    def tail(self, remote_path: str, lines: int = 200, follow: bool = False) -> CommandResult:
        return CommandResult(self.host, ["tail"], 0, "", "")

    def argv(self, command: str) -> list[str]:
        return ["sh", "-c", command]

    def interactive_argv(self, command: str) -> list[str]:
        return ["sh", "-c", command]


def test_probe_host_carries_the_registered_assignment_into_the_report() -> None:
    """The seam every other test here stubs: the registry's `--gpus` reaches the report."""
    entry = host_entry(name="gpubox", kind="ssh", ssh="me@box", gpus=["1"])
    report = probe_host(entry, transport=OneAnswerTransport(SAMPLE))
    assert report.owned == ["1"]
    assert [gpu.uuid for gpu in report.owned_rows] == [A40]
    assert TI not in report.render()


def test_probe_host_carries_the_registered_persistent_root_too() -> None:
    entry = host_entry(name="gpubox", kind="ssh", ssh="me@box", persistent_root="/mnt/ssd-2/me/")
    report: Any = probe_host(entry, transport=OneAnswerTransport(OVERLAY_HOME))
    assert report.persistent_root == "/mnt/ssd-2/me"
    assert report.document()["persistent_root"] == "/mnt/ssd-2/me"
    # An overlay $HOME that already has a root is the other half of the note
    # that used to be here: it says nothing about it either.
    assert "persistent-root" not in report.render()


def test_two_entries_naming_one_card_is_called_out() -> None:
    """`gpuc host bootstrap` refuses this, so the probe has to be the one to say why."""
    report = parse_probe("gpubox", SAMPLE, None, ["1", A40])
    assert [gpu.uuid for gpu in report.owned_rows] == [A40]
    assert f"{A40} name a card already named" in report.render()


def test_an_assignment_that_resolves_to_nothing_is_not_blamed_on_other_owners() -> None:
    rendered = parse_probe("gpubox", SAMPLE, None, ["7", "9"]).render()
    assert "assigned but not present on this host: 7, 9" in rendered
    assert "are neither assigned to gpubox" not in rendered


def test_the_probe_finds_an_interpreter_good_enough_to_read_a_host_with() -> None:
    """A host somebody else bootstrapped should be readable from here at once,
    so `host add` records the `python3` it found -- if it is new enough."""
    assert parse_probe("h", "===python3===\n/usr/bin/python3 3.12.3\n").host_python == (
        "/usr/bin/python3"
    )
    for useless in ("not installed", "/usr/bin/python3 3.9.18", "", "/usr/bin/python3"):
        assert parse_probe("h", f"===python3===\n{useless}\n").host_python is None


def test_a_shared_card_is_shown_and_marked_as_shared() -> None:
    """On a shared box the three states are ours, borrowed and none of our
    business, and a probe that folds the last two together is the one that
    says gpuc will never use a card it is about to use."""
    report = parse_probe("gpubox", SAMPLE, None, ["0"], ["1"])
    rendered = report.render()
    assert "gpus: 1 of 2 assigned to gpubox, 1 shared" in rendered
    assert f"NVIDIA A40  46068 MiB  {A40}  (shared)" in rendered
    assert "are neither assigned to gpubox" not in rendered


def test_a_shared_entry_the_host_cannot_see_is_called_out() -> None:
    report = parse_probe("gpubox", SAMPLE, None, ["0"], ["7"])
    assert report.shared_missing == ["7"]
    assert "shared but not present on this host: 7" in report.render()


def test_the_document_flags_the_shared_cards_too() -> None:
    document = parse_probe("gpubox", SAMPLE, None, ["0"], ["1"]).document()
    assert [(gpu["uuid"], gpu["assigned"], gpu["shared"]) for gpu in document["gpus"]] == [
        (TI, True, False),
        (A40, False, True),
    ]
    assert document["shared_gpus"] == ["1"]
    assert document["shared_missing"] == []


def test_render_says_scopes_need_linger() -> None:
    no_linger = SAMPLE.replace("===systemd_scope===\nno", "===systemd_scope===\nyes")
    assert "loginctl enable-linger" in parse_probe("gpubox", no_linger).render()
    lingering = no_linger.replace("===linger===\nno", "===linger===\nyes")
    assert "loginctl enable-linger" not in parse_probe("gpubox", lingering).render()
    # A lingering user instance holds the dispatcher's scope past logout.
    assert "logind kills user processes" not in parse_probe("gpubox", lingering).render()
