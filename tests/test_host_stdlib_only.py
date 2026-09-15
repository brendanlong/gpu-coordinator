"""gpuc.host must import on a bare interpreter with nothing installed."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PROBE = r"""
import importlib
import pkgutil
import sys

import gpuc.host

ALLOWED = set(sys.stdlib_module_names) | {"gpuc"}


class BlockThirdParty:
    def find_spec(self, fullname, path=None, target=None):
        top = fullname.split(".")[0]
        if top not in ALLOWED:
            raise ImportError(f"third-party import blocked: {fullname}")
        return None


sys.meta_path.insert(0, BlockThirdParty())
names = sorted(m.name for m in pkgutil.iter_modules(gpuc.host.__path__, "gpuc.host."))
for name in names:
    importlib.import_module(name)
print(" ".join(names))
"""

EXPECTED_MODULES = {
    "gpuc.host.__main__",
    "gpuc.host.dispatcher",
    "gpuc.host.gpus",
    "gpuc.host.health",
    "gpuc.host.jobs",
    "gpuc.host.paths",
    "gpuc.host.queue",
    "gpuc.host.runner",
    "gpuc.host.sync",
    "gpuc.host.terminate",
}


def test_every_host_module_imports_without_third_party_packages() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    imported = set(proc.stdout.split())
    assert imported >= EXPECTED_MODULES, EXPECTED_MODULES - imported
