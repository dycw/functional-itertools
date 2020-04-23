from __future__ import annotations

from itertools import starmap
from operator import neg
from typing import Iterable
from typing import Tuple
from typing import Type

from hypothesis import given
from hypothesis.strategies import integers
from hypothesis.strategies import tuples
from pytest import mark

from functional_itertools import CIterable
from tests.strategies import CLASSES
from tests.strategies import get_cast
from tests.strategies import real_iterables


@mark.parametrize("cls", CLASSES)
@given(x=real_iterables(integers(), max_size=10))
def test_pmap(cls: Type, x: Iterable[int]) -> None:
    y = cls(x).pmap(neg, processes=1)
    assert isinstance(y, cls)
    cast = get_cast(cls)
    assert cast(y) == cast(map(neg, x))


@mark.parametrize("cls", CLASSES)
@given(
    x=real_iterables(real_iterables(integers(), min_size=1, max_size=10), min_size=1, max_size=10),
)
def test_pmap_nested(cls: Type, x: Iterable[Iterable[int]]) -> None:
    y = cls(x).pmap(_pmap_neg, processes=1)
    assert isinstance(y, cls)
    cast = get_cast(cls)
    assert cast(y) == cast(max(map(neg, x_i)) for x_i in x)


def _pmap_neg(x: Iterable[int]) -> int:
    return CIterable(x).pmap(neg, processes=1).max()


@mark.parametrize("cls", CLASSES)
@given(x=real_iterables(tuples(integers(), integers()), min_size=1))
def test_pstarmap(cls: Type, x: Iterable[Tuple[int, int]]) -> None:
    y = cls(x).pstarmap(max, processes=1)
    assert isinstance(y, cls)
    cast = get_cast(cls)
    assert cast(y) == cast(starmap(max, x))
