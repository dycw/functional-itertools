from __future__ import annotations

from typing import Iterable

import more_itertools
from hypothesis import given
from hypothesis.strategies import integers
from pytest import mark

from functional_itertools import CIterable
from functional_itertools import CList
from tests.strategies import Case
from tests.strategies import CASES
from tests.strategies import real_iterables


@mark.parametrize("case", CASES)
@given(x=real_iterables(integers(), max_size=1000), n=integers(0, 10))
def test_chunked(case: Case, x: Iterable[int], n: int) -> None:
    y = case.cls(x).chunked(n)
    expected = CIterable if case.cls is CIterable else CList
    assert isinstance(y, expected)
    for yi, zi in zip(y, more_itertools.chunked(case.cast(x), n)):
        assert isinstance(yi, expected)
        assert list(yi) == list(zi)


@mark.parametrize("case", CASES)
@mark.parametrize("name", ["distribute", "divide"])
@given(x=real_iterables(integers(), max_size=1000), n=integers(1, 10))
def test_distribute_and_divide(case: Case, name: str, x: Iterable[int], n: int) -> None:
    y = getattr(case.cls(x), name)(n)
    expected = CIterable if case.cls is CIterable else CList
    assert isinstance(y, expected)
    for yi, zi in zip(y, getattr(more_itertools, name)(n, case.cast(x))):
        assert isinstance(yi, expected)
        assert list(yi) == list(zi)
