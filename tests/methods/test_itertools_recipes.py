from __future__ import annotations

from functools import partial
from itertools import islice
from operator import add
from operator import neg
from sys import maxsize
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type

from hypothesis import given
from hypothesis.strategies import data
from hypothesis.strategies import DataObject
from hypothesis.strategies import integers
from hypothesis.strategies import none
from hypothesis.strategies import sets
from hypothesis.strategies import tuples
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
from pytest import mark

from functional_itertools import CIterable
from functional_itertools import CList
from functional_itertools import CTuple
from tests.strategies import CLASSES
from tests.strategies import islice_ints
from tests.strategies import ORDERED_CLASSES
from tests.strategies import real_iterables
from tests.strategies import slists
from tests.test_utilities import is_even


@mark.parametrize("cls", CLASSES)
@given(x=real_iterables(integers()))
def test_all_equal(cls: Type, x: Iterable[int]) -> None:
    y = cls(x).all_equal()
    assert isinstance(y, bool)
    assert y == all_equal(x)


@given(x=real_iterables(integers()), n=none() | integers(0, maxsize))
def test_consume(x: Iterable[int], n: Optional[int]) -> None:
    y = CIterable(x).consume(n=n)
    assert isinstance(y, CIterable)
    iter_x = iter(x)
    consume(iter_x, n=n)
    assert list(y) == list(iter_x)


@mark.parametrize("cls", CLASSES)
@given(pairs=real_iterables(tuples(integers(), integers()), min_size=1))
def test_dotproduct(cls: Type, pairs: Iterable[Tuple[int, int]]) -> None:
    x, y = zip(*pairs)
    z = cls(x).dotproduct(y)
    assert isinstance(z, int)
    if cls in ORDERED_CLASSES:
        assert z == dotproduct(x, y)


@mark.parametrize("cls", CLASSES)
@given(x=real_iterables(real_iterables(integers())))
def test_flatten(cls: Type, x: Iterable[Iterable[int]]) -> None:
    y = cls(x).flatten()
    assert isinstance(y, CIterable if cls is CIterable else CList)
    if cls in ORDERED_CLASSES:
        assert list(y) == list(flatten(x))


@mark.parametrize("cls", CLASSES)
@given(x=real_iterables(integers(), max_size=10), n=integers(0, 5))
def test_ncycles(cls: Type, x: Iterable[int], n: int) -> None:
    y = cls(x).ncycles(n)
    assert isinstance(y, CIterable if cls is CIterable else CList)
    if cls in ORDERED_CLASSES:
        assert list(y) == list(ncycles(x, n))


@mark.parametrize("cls", CLASSES)
@given(
    x=real_iterables(integers()), n=integers(0, maxsize), default=none() | integers(),
)
def test_nth(cls: Type, x: Iterable[int], n: int, default: Optional[int]) -> None:
    y = cls(x).nth(n, default=default)
    assert isinstance(y, int) or (y is None)
    if cls in ORDERED_CLASSES:
        assert y == nth(x, n, default=default)


@mark.parametrize("cls", CLASSES)
@given(x=real_iterables(integers()), n=integers(0, maxsize))
def test_take(cls: Type, x: Iterable[int], n: int) -> None:
    y = cls(x).take(n)
    assert isinstance(y, CIterable if cls is CIterable else CList)
    if cls in ORDERED_CLASSES:
        assert list(y) == list(take(n, x))


@given(x=slists(integers()), n=islice_ints)
def test_padnone(x: List[int], n: int) -> None:
    y = CIterable(x).padnone()
    assert isinstance(y, CIterable)
    assert list(y[:n]) == list(islice(padnone(x), n))


@mark.parametrize("cls", CLASSES)
@given(x=real_iterables(integers()))
def test_pairwise(cls: Type, x: Iterable[int]) -> None:
    y = cls(x).pairwise()
    assert isinstance(y, CIterable if cls is CIterable else CList)
    z = list(y)
    for zi in z:
        assert isinstance(zi, CTuple)
        zi0, zi1 = zi
        assert isinstance(zi0, int)
        assert isinstance(zi1, int)
    if cls in ORDERED_CLASSES:
        assert z == list(pairwise(x))


@mark.parametrize("cls", CLASSES)
@given(x=real_iterables(integers()), value=integers())
def test_prepend(cls: Type, x: Iterable[int], value: int) -> None:
    y = cls(x).prepend(value)
    assert isinstance(y, CIterable if cls is CIterable else CList)
    if cls in ORDERED_CLASSES:
        assert list(y) == list(prepend(value, x))


@mark.parametrize("cls", CLASSES)
@given(x=sets(integers()))
def test_quantify(cls: Type, x: Set[int]) -> None:
    y = cls(x).quantify(pred=is_even)
    assert isinstance(y, int)
    assert y == quantify(x, pred=is_even)


@mark.parametrize("cls", CLASSES)
@given(data=data(), n=islice_ints)
def test_repeatfunc(cls: Type, data: DataObject, n: int) -> None:
    add1 = partial(add, 1)
    if cls is CIterable:
        times = data.draw(none() | integers(0, 10))
    else:
        times = data.draw(integers(0, 10))
    y = cls.repeatfunc(add1, times, 0)
    assert isinstance(y, CIterable if cls is CIterable else CList)
    z = repeatfunc(add1, times, 0)
    if (cls is CIterable) and (times is None):
        assert list(y[:n]) == list(islice(z, n))
    else:
        assert list(y) == list(z)


@given(start=integers(), n=islice_ints)
def test_tabulate(start: int, n: int) -> None:
    x = CIterable.tabulate(neg, start=start)
    assert isinstance(x, CIterable)
    assert list(islice(x, n)) == list(islice(tabulate(neg, start=start), n))


@mark.parametrize("cls", CLASSES)
@given(x=real_iterables(integers()), n=integers(0, maxsize))
def test_tail(cls: Type, x: Iterable[int], n: int) -> None:
    y = cls(x).tail(n)
    assert isinstance(y, CIterable if cls is CIterable else CList)
    if cls in ORDERED_CLASSES:
        assert list(y) == list(tail(n, x))
