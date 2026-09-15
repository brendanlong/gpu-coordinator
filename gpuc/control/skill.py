"""Find the agent guide, print it, or drop it into a project.

`skills/gpuc/SKILL.md` is the canonical copy -- it is what the README and the
docs link to -- so it cannot move into the package directory. The wheel gets it
as `gpuc/SKILL.md` through hatch's `force-include`, which is what makes
`gpuc skill` work for a `uv tool install` user who has no checkout at all. A
source checkout has no such copy, hence the fallback: the same file, found
relative to the package, which is the only thing both layouts share.
"""

from __future__ import annotations

from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path

import gpuc

RESOURCE = "SKILL.md"
INSTALL_PARTS = (".claude", "skills", "gpuc", RESOURCE)


class SkillError(RuntimeError):
    pass


def packaged() -> Traversable:
    """The copy `force-include` puts in the wheel. Absent in a checkout."""
    return files(gpuc) / RESOURCE


def source_copy() -> Path:
    """The canonical file, from a checkout or an editable install."""
    return Path(gpuc.__file__).resolve().parents[1] / "skills" / "gpuc" / RESOURCE


def read_skill() -> str:
    resource = packaged()
    if resource.is_file():
        return resource.read_text(encoding="utf-8")
    fallback = source_copy()
    if fallback.is_file():
        return fallback.read_text(encoding="utf-8")
    raise SkillError(
        f"this build carries no {RESOURCE} and none is at {fallback}; "
        "reinstall gpuc, or read skills/gpuc/SKILL.md in the repo"
    )


def target_path(directory: Path | None = None) -> Path:
    """Absolute, so what the command prints can be pasted somewhere else."""
    return (directory or Path.cwd()).resolve().joinpath(*INSTALL_PARTS)


def install_skill(directory: Path | None = None, force: bool = False) -> Path:
    """Write the guide under `directory/.claude/skills/gpuc/`, never silently."""
    text = read_skill()  # before touching the filesystem: no half-made directories
    target = target_path(directory)
    if target.exists() and not force:
        raise SkillError(f"{target} already exists; pass --force to overwrite it")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target
