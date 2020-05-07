from __future__ import annotations

from itertools import permutations
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

from hypothesis import given
from hypothesis.strategies import integers
from hypothesis.strategies import lists
from hypothesis.strategies import tuples
from pytest import mark

from functional_itertools import CDict
from functional_itertools import CIterable
from tests.strategies import Case
from tests.strategies import CASES
from tests.test_utilities import is_even
from tests.test_utilities import sum_varargs


@mark.parametrize("case", CASES)
@mark.parametrize("kwargs", [{}, {"parallel": True, "processes": 1}])
@given(x=lists(integers()), xs=lists(lists(integers())))
def test_map_dict(case: Case, x: List[int], xs: List[List[int]], kwargs: Dict[str, Any]) -> None:
    y = case.cls(x).map_dict(sum_varargs, *xs, **kwargs)
    assert isinstance(y, CDict)
    z = case.cast(x)
    if xs:
        keys = zip(z, *xs)
    else:
        keys = z
    assert y == dict(zip(keys, map(sum_varargs, z, *xs)))


@given(x=lists(integers()))
def test_pipe(x: List[int]) -> None:
    y = CIterable(x).pipe(permutations, r=2)
    assert isinstance(y, CIterable)
    assert list(y) == list(permutations(x, r=2))


@mark.parametrize("case", CASES)
@given(x=lists(tuples(integers(), integers())))
def test_starfilter(case: Case, x: List[Tuple[int, int]]) -> None:
    def func(key: int, value: int) -> bool:
        return is_even(key) and is_even(value)

    y = case.cls(x).starfilter(func)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast((i, j) for (i, j) in x if func(i, j))
