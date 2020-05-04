from __future__ import annotations

from itertools import chain
from re import escape
from sys import maxsize
from typing import List
from typing import Union

from hypothesis import example
from hypothesis import given
from hypothesis.strategies import integers
from hypothesis.strategies import lists
from pytest import raises

from functional_itertools import CIterable


@given(x=integers() | lists(integers()))
def test_init(x: Union[int, List[int]]) -> None:
    if isinstance(x, int):
        with raises(
            TypeError, match="CIterable expected an iterable, but 'int' object is not iterable",
        ):
            CIterable(x)  # type: ignore
    else:
        assert isinstance(CIterable(iter(x)), CIterable)


@given(x=lists(integers()), index=integers())
@example(x=[], index=-1)
@example(x=[], index=maxsize + 1)
@example(x=[], index=0.0)
def test_getitem(x: List[int], index: Union[int, float]) -> None:
    y = CIterable(x)
    if isinstance(index, int):
        num_ints = len(x)
        if 0 <= index < num_ints:
            z = y[index]
            assert isinstance(z, int)
            assert z == x[index]
        elif num_ints <= index <= maxsize:
            with raises(IndexError, match="CIterable index out of range"):
                y[index]
        else:
            with raises(
                ValueError,
                match="Indices for CIterable.__getitem__ must be an integer: 0 <= x <= sys.maxsize",
            ):
                y[index]
    else:
        with raises(
            TypeError, match=escape("Expected an int or slice; got a(n) float"),
        ):
            y[index]


@given(x=lists(integers()))
def test_dunder_iter(x: List[int]) -> None:
    assert list(CIterable(x)) == list(x)


# repr and str


@given(x=lists(integers()))
def test_repr(x: List[int]) -> None:
    assert repr(CIterable(x)) == f"CIterable({x!r})"


@given(x=lists(integers()))
def test_str(x: List[int]) -> None:
    assert str(CIterable(x)) == f"CIterable({x})"


# extra public


@given(x=lists(integers()), value=integers())
def test_append(x: List[int], value: int) -> None:
    y = CIterable(x).append(value)
    assert isinstance(y, CIterable)
    assert list(y) == list(chain(x, [value]))
