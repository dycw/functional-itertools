from __future__ import annotations
from pytest import mark
from functools import partial
from itertools import islice
from operator import add
from operator import neg
from sys import maxsize
from typing import List
from typing import Optional
from typing import Type

from hypothesis import given
from hypothesis.strategies import data
from hypothesis.strategies import DataObject
from hypothesis.strategies import integers
from hypothesis.strategies import none
from more_itertools import all_equal
from more_itertools import consume
from more_itertools import dotproduct
from more_itertools import flatten
from more_itertools import ncycles
from more_itertools import nth
from more_itertools import padnone
from more_itertools import pairwise
from more_itertools import prepend
from more_itertools import quantify
from more_itertools import repeatfunc
from more_itertools import tabulate
from more_itertools import tail
from more_itertools import take

from functional_itertools import CIterable
from functional_itertools import CList
from functional_itertools import CTuple
from tests.strategies import CLASSES
from tests.strategies import islice_ints
from tests.strategies import nested_siterables
from tests.strategies import siterables
from tests.strategies import slists
from tests.strategies import small_ints
from tests.test_utilities import is_even


@mark.parametrize("cls", CLASSES)
@given(data=data(), n=integers(0, maxsize))
def test_take(cls: Type, data: DataObject, n: int) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    y = cls(x).take(n)
    assert isinstance(y, cls)
    if cls in {CIterable, CList, CTuple}:
        assert cast(y) == cast(take(n, x))


@mark.parametrize("cls", CLASSES)
@given(data=data(), value=integers())
def test_prepend(cls: Type, data: DataObject, value: int) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    y = cls(x).prepend(value)
    assert isinstance(y, cls)
    assert cast(y) == cast(prepend(value, x))


@given(start=integers(), n=islice_ints)
def test_tabulate(start: int, n: int) -> None:
    x = CIterable.tabulate(neg, start=start)
    assert isinstance(x, CIterable)
    assert list(islice(x, n)) == list(islice(tabulate(neg, start=start), n))


@mark.parametrize("cls", CLASSES)
@given(data=data(), n=small_ints)
def test_tail(cls: Type, data: DataObject, n: int) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    y = cls(x).tail(n)
    assert isinstance(y, cls)
    if cls in {CIterable, CList, CTuple}:
        assert cast(y) == cast(tail(n, x))


@mark.parametrize("cls", CLASSES)
@given(data=data(), n=none() | small_ints)
def test_consume(cls: Type, data: DataObject, n: Optional[int]) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    y = cls(x).consume(n=n)
    assert isinstance(y, cls)
    if cls in {CIterable, CList, CTuple}:
        iter_x = iter(x)
        consume(iter_x, n=n)
        assert cast(y) == cast(iter_x)


@mark.parametrize("cls", CLASSES)
@given(data=data(), n=small_ints, default=none() | small_ints)
def test_nth(cls: Type, data: DataObject, n: int, default: Optional[int]) -> None:
    x, _ = data.draw(siterables(cls, integers()))
    y = cls(x).nth(n, default=default)
    assert isinstance(y, int) or (y is None)
    if cls in {CIterable, CList, CTuple}:
        assert y == nth(x, n, default=default)


@mark.parametrize("cls", CLASSES)
@given(data=data())
def test_all_equal(cls: Type, data: DataObject) -> None:
    x, _ = data.draw(siterables(cls, integers()))
    y = cls(x).all_equal()
    assert isinstance(y, bool)
    assert y == all_equal(x)


@mark.parametrize("cls", CLASSES)
@given(data=data())
def test_quantify(cls: Type, data: DataObject) -> None:
    x, _ = data.draw(siterables(cls, integers()))
    y = cls(x).quantify(pred=is_even)
    assert isinstance(y, int)
    assert y == quantify(x, pred=is_even)


@given(x=slists(integers()), n=islice_ints)
def test_padnone(x: List[int], n: int) -> None:
    y = CIterable(x).padnone()
    assert isinstance(y, CIterable)
    assert list(y[:n]) == list(islice(padnone(x), n))


@mark.parametrize("cls", CLASSES)
@given(data=data(), n=small_ints)
def test_ncycles(cls: Type, data: DataObject, n: int) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    y = cls(x).ncycles(n)
    assert isinstance(y, cls)
    assert cast(y) == cast(ncycles(x, n))


@mark.parametrize("cls", CLASSES)
@given(data=data())
def test_dotproduct(cls: Type, data: DataObject) -> None:
    x, _ = data.draw(siterables(cls, integers()))
    y, _ = data.draw(siterables(cls, integers(), min_size=len(x), max_size=len(x)))
    z = cls(x).dotproduct(y)
    assert isinstance(z, int)
    if cls in {CIterable, CList, CTuple}:
        assert z == dotproduct(x, y)


@mark.parametrize("cls", CLASSES)
@given(data=data())
def test_flatten(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(nested_siterables(cls, integers()))
    y = cls(x).flatten()
    assert isinstance(y, cls)
    assert cast(y) == cast(flatten(x))


@mark.parametrize("cls", CLASSES)
@given(data=data(), n=islice_ints)
def test_repeatfunc(cls: Type, data: DataObject, n: int) -> None:
    add1 = partial(add, 1)
    if cls is CIterable:
        times = data.draw(none() | small_ints)
    else:
        times = data.draw(small_ints)

    y = cls.repeatfunc(add1, times, 0)
    assert isinstance(y, cls)
    _, cast = data.draw(siterables(cls, none()))
    z = repeatfunc(add1, times, 0)
    if (cls is CIterable) and (times is None):
        assert cast(y[:n]) == cast(islice(z, n))
    else:
        assert cast(y) == cast(z)


@mark.parametrize("cls", CLASSES)
@given(data=data())
def test_pairwise(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    y = cls(x).pairwise()
    assert isinstance(y, cls)
    if cls in {CIterable, CList, CTuple}:
        assert cast(y) == cast(pairwise(x))
