from __future__ import annotations

from itertools import permutations
from operator import neg
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

from hypothesis import given
from hypothesis.strategies import booleans
from hypothesis.strategies import integers
from hypothesis.strategies import just
from hypothesis.strategies import lists
from hypothesis.strategies import none
from hypothesis.strategies import tuples
from pytest import warns

from functional_itertools import CList


# magic methods


@given(
    x=lists(integers()), index=integers() | tuples(integers(), integers()).map(lambda x: slice(*x)),
)
def test_get_item(x: List[int], index: Union[int, slice]) -> None:
    y = CList(x)
    if isinstance(index, int):
        try:
            assert isinstance(y[index], int)
        except IndexError:
            pass
    elif isinstance(index, slice):
        assert isinstance(y[index], CList)
    else:
        raise TypeError(index)  # pragma: no cover


# built-ins


@given(x=lists(integers()))
def test_copy(x: List[int]) -> None:
    y = CList(x).copy()
    assert isinstance(y, CList)
    assert y == x


@given(x=lists(integers()))
def test_reversed(x: List[int]) -> None:
    y = CList(x).reversed()
    assert isinstance(y, CList)
    assert y == list(reversed(x))


@given(x=lists(integers()), key=none() | just(neg), reverse=booleans())
def test_sort(x: List[int], key: Optional[Callable[[int], int]], reverse: bool) -> None:
    with warns(UserWarning, match="Use the 'sorted' name instead of 'sort'"):
        y = CList(x).sort(key=key, reverse=reverse)
    assert isinstance(y, CList)
    assert y == sorted(x, key=key, reverse=reverse)


# extra public


@given(x=lists(integers()))
def test_pipe(x: List[int]) -> None:
    y = CList(x).pipe(permutations, r=2)
    assert isinstance(y, CList)
    assert y == list(permutations(x, r=2))
