"""The one build comparison, shared by both halves."""

from __future__ import annotations

from gpuc._version import is_other_build
from gpuc.control import version


def test_the_control_side_uses_the_stdlib_comparison() -> None:
    assert version.is_other_build is is_other_build


def test_a_prefix_is_the_same_build() -> None:
    assert not is_other_build("abcdef123456", "abcdef123456789")
    assert not is_other_build("abcdef123456789", "abcdef123456")
    assert is_other_build("abcdef123456", "fedcba654321")


def test_dirty_is_part_of_the_identity_in_both_directions() -> None:
    assert is_other_build("abcdef123456-dirty", "abcdef123456")
    assert is_other_build("abcdef123456", "abcdef123456-dirty")
    assert not is_other_build("abcdef123456-dirty", "abcdef123456-dirty")


def test_unknowns_are_asymmetric() -> None:
    assert is_other_build(None, "abcdef123456"), "a host that recorded nothing was never shipped"
    assert not is_other_build("abcdef123456", None), "nothing to compare against claims nothing"
    assert not is_other_build(None, None)
