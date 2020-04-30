from __future__ import annotations

from typing import List

from hypothesis import given
from hypothesis.strategies import integers
from hypothesis.strategies import lists
from more_itertools import chunked
from more_itertools import distribute
from more_itertools import divide
from pytest import mark

from functional_itertools import CTuple
from tests.strategies import Case
from tests.strategies import CASES


@mark.parametrize("case", CASES)
@given(x=lists(integers(), max_size=1000), n=integers(0, 10))
def test_chunked(case: Case, x: List[int], n: int) -> None:
    y = case.cls(x).chunked(n)
    assert isinstance(y, case.cls)
    z = list(y)
    for zi in z:
        assert isinstance(zi, CTuple)
    assert case.cast(map(case.cast, z)) == case.cast(map(case.cast, chunked(case.cast(x), n)))


@mark.parametrize("case", CASES)
@given(x=lists(integers(), max_size=1000), n=integers(1, 10))
def test_distribute(case: Case, x: List[int], n: int) -> None:
    y = case.cls(x).distribute(n)
    assert isinstance(y, case.cls)
    z = list(y)
    for zi in z:
        assert isinstance(zi, CTuple)
    assert case.cast(map(case.cast, z)) == case.cast(map(case.cast, distribute(n, case.cast(x))))


@mark.parametrize("case", CASES)
@given(x=lists(integers(), max_size=1000), n=integers(1, 10))
def test_divide(case: Case, x: List[int], n: int) -> None:
    y = case.cls(x).divide(n)
    assert isinstance(y, case.cls)
    z = list(y)
    for zi in z:
        assert isinstance(zi, CTuple)
    assert case.cast(map(case.cast, z)) == case.cast(map(case.cast, divide(n, case.cast(x))))
