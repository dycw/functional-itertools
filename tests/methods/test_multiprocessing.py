from __future__ import annotations

from itertools import starmap
from operator import neg
from typing import Iterable
from typing import Tuple

from hypothesis import given
from hypothesis.strategies import integers
from hypothesis.strategies import tuples
from pytest import mark

from functional_itertools import CIterable
from tests.strategies import Case
from tests.strategies import CASES
from tests.strategies import real_iterables


@mark.parametrize("case", CASES)
@given(x=real_iterables(integers()))
def test_pmap(case: Case, x: Iterable[int]) -> None:
    y = case.cls(x).pmap(neg)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast(map(neg, x))


@mark.parametrize("case", CASES)
@given(x=real_iterables(tuples(integers(), integers())))
def test_pstarmap(case: Case, x: Iterable[Tuple[int, int]]) -> None:
    y = case.cls(x).pstarmap(max)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast(starmap(max, x))


@mark.parametrize("case", CASES)
@given(
    x=real_iterables(real_iterables(integers(), min_size=1, max_size=10), min_size=1, max_size=10),
)
def test_map_nested(case: Case, x: Iterable[Iterable[int]]) -> None:
    y = case.cls(x).map(_parallel_map_neg, parallel=True, processes=1)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast(max(map(neg, x_i)) for x_i in x)


def _parallel_map_neg(x: Iterable[int]) -> int:
    return CIterable(x).map(neg, parallel=True, processes=1).max()
