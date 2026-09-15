"""`gpuc skill`: the agent guide has to be findable from a wheel *and* a checkout.

The canonical file is `skills/gpuc/SKILL.md`, which is outside the package, so
the wheel gets a `force-include`d copy at `gpuc/SKILL.md` and a checkout has
none. Both lookups are covered here because only one of them is ever live in
any given install, and the dead one is the one that breaks silently.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from gpuc.control import skill
from gpuc.control.cli import EXIT_ERROR, EXIT_OK, main

REPO_SKILL = Path(__file__).resolve().parents[1] / "skills" / "gpuc" / "SKILL.md"


@pytest.fixture
def no_packaged_copy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """What a source checkout looks like: `gpuc/SKILL.md` is not there."""
    monkeypatch.setattr(skill, "packaged", lambda: tmp_path / "absent.md")


def test_the_packaged_copy_is_preferred(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    resource = tmp_path / "SKILL.md"
    resource.write_text("# from the wheel\n")
    monkeypatch.setattr(skill, "packaged", lambda: resource)
    assert skill.read_skill() == "# from the wheel\n"


def test_a_checkout_falls_back_to_the_canonical_file(no_packaged_copy: None) -> None:
    assert skill.source_copy() == REPO_SKILL
    assert skill.read_skill() == REPO_SKILL.read_text()


def test_neither_copy_is_an_error_that_says_where_to_look(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(skill, "packaged", lambda: tmp_path / "absent.md")
    monkeypatch.setattr(skill, "source_copy", lambda: tmp_path / "also-absent.md")
    with pytest.raises(skill.SkillError, match=re.escape("skills/gpuc/SKILL.md")):
        skill.read_skill()


def test_skill_prints_the_file_verbatim(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["skill"]) == EXIT_OK
    assert capsys.readouterr().out == REPO_SKILL.read_text()


def test_install_writes_it_under_dot_claude_and_says_where(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["skill", "--install", str(tmp_path)]) == EXIT_OK
    target = tmp_path / ".claude" / "skills" / "gpuc" / "SKILL.md"
    assert target.read_text() == REPO_SKILL.read_text()
    assert str(target) in capsys.readouterr().out


def test_install_defaults_to_the_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["skill", "--install"]) == EXIT_OK
    assert (tmp_path / ".claude" / "skills" / "gpuc" / "SKILL.md").is_file()


def test_install_refuses_to_overwrite_without_force(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / ".claude" / "skills" / "gpuc" / "SKILL.md"
    target.parent.mkdir(parents=True)
    target.write_text("someone else's file\n")
    assert main(["skill", "--install", str(tmp_path)]) == EXIT_ERROR
    assert "--force" in capsys.readouterr().err
    assert target.read_text() == "someone else's file\n"

    assert main(["skill", "--install", str(tmp_path), "--force"]) == EXIT_OK
    assert target.read_text() == REPO_SKILL.read_text()
