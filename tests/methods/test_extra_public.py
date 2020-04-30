from __future__ import annotations

from itertools import permutations
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Type

from _pytest.python_api import raises
from hypothesis import given
from hypothesis.strategies import integers
from hypothesis.strategies import lists
from pytest import mark

from functional_itertools import CDict
from functional_itertools import CIterable
from functional_itertools import CList
from functional_itertools import EmptyIterableError
from functional_itertools import MultipleElementsError
from tests.strategies import Case
from tests.strategies import CASES
from tests.strategies import real_iterables
from tests.test_utilities import sum_varargs


@given(x=lists(integers()))
@mark.parametrize("method_name, index", [("first", 0), ("last", -1)])
def test_first_and_last(x: List[int], method_name: str, index: int) -> None:
    method = getattr(CIterable(x), method_name)
    if x:
        assert method() == x[index]
    else:
        with raises(EmptyIterableError):
            method()


@mark.parametrize("case", CASES)
@mark.parametrize("kwargs", [{}, {"parallel": True, "processes": 1}])
@given(x=real_iterables(integers()), xs=real_iterables(real_iterables(integers())))
def test_map_dict(
    case: Case, x: Iterable[int], xs: Iterable[Iterable[int]], kwargs: Dict[str, Any],
) -> None:
    y = case.cls(x).map_dict(sum_varargs, *xs, **kwargs)
    assert isinstance(y, CDict)
    assert y == dict(zip(x, map(sum_varargs, x, *xs)))


@mark.parametrize("case", CASES)
@given(x=real_iterables(integers()))
def test_one(case: Case, x: Iterable[int]) -> None:
    length = len(x)
    if length == 0:
        with raises(EmptyIterableError):
            case.cls(x).one()
    elif length == 1:
        assert case.cls(x).one() == next(iter(x))
    else:
        if case.ordered:
            with raises(MultipleElementsError, match=r"^-?\d+, -?\d+$"):
                case.cls(x).one()


@given(x=lists(integers()))
def test_pipe(x: List[int]) -> None:
    y = CIterable(x).pipe(permutations, r=2)
    assert isinstance(y, CIterable)
    assert list(y) == list(permutations(x, r=2))


@mark.parametrize("cls", [CIterable, CList])
@given(x=lists(integers(), min_size=1))
def test_unzip(cls: Type, x: List[int]) -> None:
    indices, ints = cls(x).enumerate().unzip()
    assert isinstance(indices, cls)
    assert list(indices) == list(range(len(x)))
    assert isinstance(ints, cls)
    assert list(ints) == x
