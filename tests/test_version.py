"""The one build comparison, shared by both halves."""

from __future__ import annotations

import pytest

from gpuc._version import is_other_build
from gpuc.control import version


def test_the_control_side_uses_the_stdlib_comparison() -> None:
    assert version.is_other_build is is_other_build


def test_a_prefix_is_the_same_build() -> None:
    assert not is_other_build("abcdef123456", "abcdef123456789")
    assert not is_other_build("abcdef123456789", "abcdef123456")
    assert is_other_build("abcdef123456", "fedcba654321")


def test_dirty_is_part_of_the_identity_in_both_directions() -> None:
    assert is_other_build("abcdef123456-dirty-1a2b3c4d", "abcdef123456")
    assert is_other_build("abcdef123456", "abcdef123456-dirty-1a2b3c4d")
    assert not is_other_build("abcdef123456-dirty-1a2b3c4d", "abcdef123456-dirty-1a2b3c4d")
    assert not is_other_build("abcdef123456-dirty-1a2b3c4d", "abcdef123456789-dirty-1a2b3c4d")


def test_two_dirty_trees_on_one_commit_are_two_builds() -> None:
    """A second edit to a dirty tree used to be invisible: the host already
    named `<commit>-dirty`, so nothing re-shipped it."""
    assert is_other_build("abcdef123456-dirty-1a2b3c4d", "abcdef123456-dirty-ffffffff")
    # A bare `-dirty` recorded by an earlier build is another tree too.
    assert is_other_build("abcdef123456-dirty", "abcdef123456-dirty-1a2b3c4d")


def test_the_dirty_tag_hashes_the_working_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    answers = {("status", "--porcelain"): " M gpuc/x.py\n", ("diff", "HEAD"): "-a\n+b\n"}
    monkeypatch.setattr(version, "_git", lambda *args: answers.get(args, ""))
    first = version.dirty_tag()
    assert first.startswith("-dirty-") and len(first) == len("-dirty-") + 8
    answers[("diff", "HEAD")] = "-a\n+c\n"
    assert version.dirty_tag() != first
    answers[("status", "--porcelain")] = ""
    assert version.dirty_tag() == ""
    assert version.short("a" * 40 + first) == "a" * 12 + first
    assert version.short("a" * 40) == "a" * 12


def test_unknowns_are_asymmetric() -> None:
    assert is_other_build(None, "abcdef123456"), "a host that recorded nothing was never shipped"
    assert not is_other_build("abcdef123456", None), "nothing to compare against claims nothing"
    assert not is_other_build(None, None)
