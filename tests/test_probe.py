from __future__ import annotations

from gpuc.control.probe import PROBE_SCRIPT, parse_probe

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
===uv===
not installed
===python3===
/usr/bin/python3 3.13.1
===download===
20 MB in 0.4s = 50.0 MB/s
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
===download===
no python3 and no curl: cannot time a download
"""


def test_probe_script_is_posix_sh_with_no_gpuc_dependency() -> None:
    assert "gpuc" not in PROBE_SCRIPT.replace("gpuc host bootstrap", "")
    assert "bash" not in PROBE_SCRIPT
    assert "python3 -" in PROBE_SCRIPT


def test_parse_finds_every_section() -> None:
    report = parse_probe("spar", SAMPLE)
    assert report.sections["driver"] == "580.173.02"
    assert report.sections["systemd_scope"] == "no"
    assert report.sections["download"].endswith("50.0 MB/s")


def test_gpu_rows_are_index_uuid_name() -> None:
    rows = parse_probe("spar", SAMPLE).gpu_rows
    assert [row[0] for row in rows] == ["0", "1"]
    assert rows[0][1] == "GPU-2a4bad3b-9fe3-7031-914d-384254e92908"
    assert rows[1][2] == "NVIDIA A40"


def test_render_warns_about_logind_and_missing_uv() -> None:
    rendered = parse_probe("spar", SAMPLE).render()
    assert "GPU-2a4bad3b-9fe3-7031-914d-384254e92908  NVIDIA GeForce RTX 3060 Ti" in rendered
    assert "logind kills user processes at logout" in rendered
    assert "gpuc host bootstrap spar" in rendered


def test_a_host_with_nothing_still_renders() -> None:
    report = parse_probe("bare", BARE)
    assert not report.has_nvidia_smi
    assert report.gpu_rows == []
    rendered = report.render()
    assert "only run gpus: 0 jobs" in rendered
    assert "cannot time a download" in rendered
