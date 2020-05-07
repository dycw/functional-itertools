from __future__ import annotations

from operator import neg
from typing import List
from typing import Tuple

from hypothesis import given
from hypothesis.strategies import integers
from hypothesis.strategies import lists
from pytest import mark

from functional_itertools import CIterable
from tests.strategies import Case
from tests.strategies import CASES


@mark.parametrize("case", CASES)
@given(x=lists(lists(integers(), min_size=1, max_size=10).map(tuple), min_size=1, max_size=10))
def test_map_nested(case: Case, x: List[Tuple[int, ...]]) -> None:
    y = case.cls(x).map(_parallel_map_neg, parallel=True, processes=1)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast(max(map(neg, x_i)) for x_i in case.cast(x))


def _parallel_map_neg(x: List[int]) -> int:
    return CIterable(x).map(neg, parallel=True, processes=1).max()
