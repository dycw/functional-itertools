from __future__ import annotations

from itertools import chain
from operator import mod
from operator import neg
from typing import Tuple

from functional_itertools.utilities import sentinel


def test_sentinel() -> None:
    assert repr(sentinel) == "<sentinel>"


def is_even(x: int) -> bool:
    return mod(x, 2) == 0


def neg_key_and_value(key: int, value: int) -> Tuple[int, int]:
    return neg(key), neg(value)


def sum_varargs(x: int, *xs: int) -> int:
    return sum(chain([x], xs))
