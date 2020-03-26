from itertools import permutations
from typing import Callable
from typing import List
from typing import Optional

from hypothesis import given
from hypothesis.strategies import booleans
from hypothesis.strategies import integers
from hypothesis.strategies import lists
from hypothesis.strategies import none
from pytest import warns

from functional_itertools.classes import CList
from tests.test_utilities import int_to_int_funcs


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


@given(x=lists(integers()), key=none() | int_to_int_funcs, reverse=booleans())
def test_sort(x: List[int], key: Optional[Callable[[int], int]], reverse: bool) -> None:
    with warns(UserWarning, match="Use the 'sorted' method instead of 'sort'"):
        y = CList(x).sort(key=key, reverse=reverse)
    assert isinstance(y, CList)
    assert y == sorted(x, key=key, reverse=reverse)


# extra public


@given(x=lists(integers()))
def test_pipe(x: List[int]) -> None:
    y = CList(x).pipe(permutations, r=2)
    assert isinstance(y, CList)
    assert y == list(permutations(x, r=2))
