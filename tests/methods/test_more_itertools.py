from __future__ import annotations

from typing import Iterable
from typing import Type

from hypothesis import given
from hypothesis.strategies import integers
from more_itertools import chunked
from more_itertools import distribute
from more_itertools import divide
from pytest import mark

from functional_itertools import CIterable
from functional_itertools import CList
from tests.strategies import CLASSES
from tests.strategies import ORDERED_CLASSES
from tests.strategies import real_iterables


@mark.parametrize("cls", CLASSES)
@given(x=real_iterables(integers(), max_size=1000), n=integers(0, 10))
def test_chunked(cls: Type, x: Iterable[int], n: int) -> None:
    y = cls(x).chunked(n)
    assert isinstance(y, CIterable if cls is CIterable else CList)
    for yi in y:
        assert isinstance(yi, CIterable if cls is CIterable else CList)
    if cls in ORDERED_CLASSES:
        assert list(map(list, cls(x).chunked(n))) == list(map(list, chunked(x, n)))


@mark.parametrize("cls", CLASSES)
@given(x=real_iterables(integers(), max_size=1000), n=integers(1, 10))
def test_distribute(cls: Type, x: Iterable[int], n: int) -> None:
    y = cls(x).distribute(n)
    assert isinstance(y, CIterable if cls is CIterable else CList)
    for yi in y:
        assert isinstance(yi, CIterable if cls is CIterable else CList)
    if cls in ORDERED_CLASSES:
        assert list(map(list, cls(x).distribute(n))) == list(map(list, distribute(n, x)))


@mark.parametrize("cls", CLASSES)
@given(x=real_iterables(integers(), max_size=1000), n=integers(1, 10))
def test_divide(cls: Type, x: Iterable[int], n: int) -> None:
    y = cls(x).divide(n)
    assert isinstance(y, CIterable if cls is CIterable else CList)
    for yi in y:
        assert isinstance(yi, CIterable if cls is CIterable else CList)
    if cls in ORDERED_CLASSES:
        assert list(map(list, cls(x).divide(n))) == list(map(list, divide(n, x)))
