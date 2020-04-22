from __future__ import annotations

from itertools import starmap
from operator import neg
from typing import Iterable
from typing import Type

from hypothesis import given
from hypothesis.strategies import data
from hypothesis.strategies import DataObject
from hypothesis.strategies import integers
from hypothesis.strategies import tuples
from pytest import mark

from functional_itertools import CFrozenSet
from functional_itertools import CIterable
from functional_itertools import CList
from functional_itertools import CSet
from tests.strategies import nested_siterables
from tests.strategies import siterables


@mark.parametrize("cls", [CIterable, CList, CSet, CFrozenSet])
@given(data=data())
def test_pmap(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    y = cls(x).pmap(neg, processes=1)
    assert isinstance(y, cls)
    assert cast(y) == cast(map(neg, x))


@mark.parametrize("cls", [CIterable, CList, CSet, CFrozenSet])
@given(data=data())
def test_pmap_nested(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(nested_siterables(cls, integers(), min_size=1))
    y = cls(x).pmap(_pmap_neg, processes=1)
    assert isinstance(y, cls)
    assert cast(y) == cast(max(map(neg, x_i)) for x_i in x)


def _pmap_neg(x: Iterable[int]) -> int:
    return CIterable(x).pmap(neg, processes=1).max()


@mark.parametrize("cls", [CIterable, CList, CSet, CFrozenSet])
@given(data=data())
def test_pstarmap(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(siterables(cls, tuples(integers(), integers())))
    y = cls(x).pstarmap(max, processes=1)
    assert isinstance(y, cls)
    assert cast(y) == cast(starmap(max, x))
