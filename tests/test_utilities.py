from __future__ import annotations

from operator import mod

from functional_itertools.utilities import sentinel


def test_sentinel() -> None:
    assert repr(sentinel) == "<sentinel>"


def is_even(x: int) -> bool:
    return mod(x, 2) == 0
