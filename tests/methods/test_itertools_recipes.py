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
from functional_itertools import CTuple
from tests.strategies import Case
from tests.strategies import CASES
from tests.strategies import islice_ints
from tests.strategies import real_iterables
from tests.test_utilities import is_even


@mark.parametrize("case", CASES)
@given(x=real_iterables(integers()))
def test_all_equal(case: Case, x: Iterable[int]) -> None:
    y = case.cls(x).all_equal()
    assert isinstance(y, bool)
    assert y == all_equal(x)


@mark.parametrize("case", CASES)
@given(x=real_iterables(integers()), n=none() | integers(0, maxsize))
def test_consume(case: Case, x: Iterable[int], n: Optional[int]) -> None:
    y = case.cls(x).consume(n=n)
    assert isinstance(y, case.cls)
    iter_x = iter(x)
    consume(iter_x, n=n)
    assert case.cast(y) == case.cast(iter_x)


@mark.parametrize("case", CASES)
@given(pairs=real_iterables(tuples(integers(), integers()), min_size=1))
def test_dotproduct(case: Case, pairs: Iterable[Tuple[int, int]]) -> None:
    x, y = zip(*pairs)
    z = case.cls(x).dotproduct(y)
    assert isinstance(z, int)
    if case.ordered:
        assert z == dotproduct(x, y)


@mark.parametrize("case", CASES)
@given(x=real_iterables(real_iterables(integers())))
def test_flatten(case: Case, x: Iterable[Iterable[int]]) -> None:
    y = case.cls(x).flatten()
    assert isinstance(y, case.cls)
    if case.ordered:
        assert case.cast(y) == case.cast(flatten(x))


@mark.parametrize("case", CASES)
@given(x=real_iterables(integers(), max_size=10), n=integers(0, 5))
def test_ncycles(case: Case, x: Iterable[int], n: int) -> None:
    y = case.cls(x).ncycles(n)
    assert isinstance(y, case.cls)
    if case.ordered:
        assert case.cast(y) == case.cast(ncycles(x, n))


@mark.parametrize("case", CASES)
@given(
    x=real_iterables(integers()), n=integers(0, maxsize), default=none() | integers(),
)
def test_nth(case: Case, x: Iterable[int], n: int, default: Optional[int]) -> None:
    y = case.cls(x).nth(n, default=default)
    assert isinstance(y, int) or (y is None)
    if case.ordered:
        assert y == nth(x, n, default=default)


@mark.parametrize("case", CASES)
@given(x=real_iterables(integers()), n=islice_ints)
def test_padnone(case: Case, x: List[int], n: int) -> None:
    y = case.cls(x).padnone()
    assert isinstance(y, CIterable)
    assert case.cast(y[:n]) == case.cast(islice(padnone(x), n))


@mark.parametrize("case", CASES)
@given(x=real_iterables(integers()))
def test_pairwise(case: Case, x: Iterable[int]) -> None:
    y = case.cls(x).pairwise()
    assert isinstance(y, case.cls)
    z = case.cast(y)
    for zi in z:
        assert isinstance(zi, CTuple)
        zi0, zi1 = zi
        assert isinstance(zi0, int)
        assert isinstance(zi1, int)
    if case.ordered:
        assert z == case.cast(pairwise(x))


@mark.parametrize("case", CASES)
@given(x=real_iterables(integers()), value=integers())
def test_prepend(case: Case, x: Iterable[int], value: int) -> None:
    y = case.cls(x).prepend(value)
    assert isinstance(y, case.cls)
    if case.ordered:
        assert case.cast(y) == case.cast(prepend(value, x))


@mark.parametrize("case", CASES)
@given(x=sets(integers()))
def test_quantify(case: Case, x: Set[int]) -> None:
    y = case.cls(x).quantify(pred=is_even)
    assert isinstance(y, int)
    assert y == quantify(x, pred=is_even)


@mark.parametrize("case", CASES)
@given(data=data(), n=islice_ints)
def test_repeatfunc(case: Case, data: DataObject, n: int) -> None:
    add1 = partial(add, 1)
    if case.cls is CIterable:
        times = data.draw(none() | integers(0, 10))
    else:
        times = data.draw(integers(0, 10))
    y = case.cls.repeatfunc(add1, times, 0)
    assert isinstance(y, case.cls)
    z = repeatfunc(add1, times, 0)
    if (case.cls is CIterable) and (times is None):
        assert case.cast(y[:n]) == case.cast(islice(z, n))
    else:
        assert case.cast(y) == case.cast(z)


@given(start=integers(), n=islice_ints)
def test_tabulate(start: int, n: int) -> None:
    x = CIterable.tabulate(neg, start=start)
    assert isinstance(x, CIterable)
    assert list(islice(x, n)) == list(islice(tabulate(neg, start=start), n))


@mark.parametrize("case", CASES)
@given(x=real_iterables(integers()), n=integers(0, maxsize))
def test_tail(case: Case, x: Iterable[int], n: int) -> None:
    y = case.cls(x).tail(n)
    assert isinstance(y, case.cls)
    if case.ordered:
        assert case.cast(y) == case.cast(tail(n, x))


@mark.parametrize("case", CASES)
@given(x=real_iterables(integers()), n=integers(0, maxsize))
def test_take(case: Case, x: Iterable[int], n: int) -> None:
    y = case.cls(x).take(n)
    assert isinstance(y, case.cls)
    if case.ordered:
        assert case.cast(y) == case.cast(take(n, x))
