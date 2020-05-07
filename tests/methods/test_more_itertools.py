from __future__ import annotations

from itertools import islice
from operator import neg
from re import escape
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

from hypothesis import given
from hypothesis.strategies import fixed_dictionaries
from hypothesis.strategies import integers
from hypothesis.strategies import just
from hypothesis.strategies import lists
from hypothesis.strategies import none
from more_itertools import chunked
from more_itertools import distribute
from more_itertools import divide
from more_itertools import first
from more_itertools import iterate
from more_itertools import last
from more_itertools import map_except
from more_itertools import one
from more_itertools import only
from pytest import mark
from pytest import raises

from functional_itertools import CIterable
from functional_itertools import CTuple
from functional_itertools import EmptyIterableError
from functional_itertools import MultipleElementsError
from tests.strategies import Case
from tests.strategies import CASES
from tests.strategies import islice_ints


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


@mark.parametrize("case", CASES)
@mark.parametrize("func", [first, last])
@given(
    x=lists(integers()), default=just({}) | fixed_dictionaries({"default": integers()}),
)
def test_first_and_last(
    case: Case, func: Callable[..., int], x: List[int], default: Dict[str, int],
) -> None:
    name = func.__name__
    try:
        y = getattr(case.cls(x), name)(**default)
    except EmptyIterableError:
        with raises(
            ValueError,
            match=escape(
                f"{name}() was called on an empty iterable, and no default value was provided.",
            ),
        ):
            func(case.cast(x), **default)
    else:
        assert isinstance(y, int)
        assert y == func(case.cast(x), **default)


@given(start=integers(), n=islice_ints)
def test_iterate(start: int, n: int) -> None:
    y = CIterable.iterate(neg, start)
    assert isinstance(y, CIterable)
    assert list(y[:n]) == list(islice(iterate(neg, start), n))


@mark.parametrize("case", CASES)
@given(x=lists(integers()))
def test_map_except(case: Case, x: List[int]) -> None:
    def func(n: int) -> int:
        if n % 2 == 0:
            return neg(n)
        else:
            raise ValueError("'n' must be even")

    y = case.cls(x).map_except(func, ValueError)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast(map_except(func, x, ValueError))


@mark.parametrize("case", CASES)
@given(x=lists(integers()))
def test_one(case: Case, x: List[int]) -> None:
    try:
        y = case.cls(x).one()
    except EmptyIterableError:
        with raises(ValueError, match=escape("too few items in iterable (expected 1)")):
            one(case.cast(x))
    except MultipleElementsError:
        with raises(
            ValueError,
            match=r"Expected exactly one item in iterable, but got -?\d+, -?\d+, and perhaps more",
        ):
            one(case.cast(x))
    else:
        assert isinstance(y, int)
        assert y == one(case.cast(x))


@mark.parametrize("case", CASES)
@given(
    x=lists(integers()), default=none() | integers(),
)
def test_only(case: Case, x: List[int], default: Optional[int]) -> None:
    try:
        y = case.cls(x).only(default=default)
    except EmptyIterableError:
        with raises(ValueError, match=escape("too few items in iterable (expected 1)")):
            only(case.cast(x), default=default)
    except MultipleElementsError:
        with raises(
            ValueError,
            match=r"Expected exactly one item in iterable, but got -?\d+, -?\d+, and perhaps more",
        ):
            only(case.cast(x), default=default)
    else:
        assert isinstance(y, int) or (y is None)
        assert y == only(case.cast(x), default=default)
