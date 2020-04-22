from __future__ import annotations

from operator import neg
from typing import Dict
from typing import Tuple

from hypothesis import given
from hypothesis.strategies import dictionaries
from hypothesis.strategies import integers

from functional_itertools import CDict
from functional_itertools import CIterable
from tests.test_utilities import is_even


@given(x=dictionaries(integers(), integers()))
def test_keys(x: Dict[int, int]) -> None:
    y = CDict(x).keys()
    assert isinstance(y, CIterable)
    assert list(y) == list(x.keys())


@given(x=dictionaries(integers(), integers()))
def test_values(x: Dict[int, int]) -> None:
    y = CDict(x).values()
    assert isinstance(y, CIterable)
    assert list(y) == list(x.values())


@given(x=dictionaries(integers(), integers()))
def test_items(x: Dict[str, int]) -> None:
    y = CDict(x).items()
    assert isinstance(y, CIterable)
    assert list(y) == list(x.items())


# built-ins
@given(x=dictionaries(integers(), integers()))
def test_filter_keys(x: Dict[int, int]) -> None:
    y = CDict(x).filter_keys(is_even)
    assert isinstance(y, CDict)
    assert y == {k: v for k, v in x.items() if is_even(k)}


@given(x=dictionaries(integers(), integers()))
def test_filter_values(x: Dict[int, int]) -> None:
    y = CDict(x).filter_values(is_even)
    assert isinstance(y, CDict)
    assert y == {k: v for k, v in x.items() if is_even(v)}


@given(x=dictionaries(integers(), integers()))
def test_filter_items(x: Dict[int, int]) -> None:
    def func(key: int, value: int) -> bool:
        return is_even(key) and is_even(value)

    y = CDict(x).filter_items(func)
    assert isinstance(y, CDict)
    assert y == {k: v for k, v in x.items() if func(k, v)}


@given(x=dictionaries(integers(), integers()))
def test_map_keys(x: Dict[int, int]) -> None:
    y = CDict(x).map_keys(neg)
    assert isinstance(y, CDict)
    assert y == {neg(k): v for k, v in x.items()}


@given(x=dictionaries(integers(), integers()))
def test_map_values(x: Dict[int, int]) -> None:
    y = CDict(x).map_values(neg)
    assert isinstance(y, CDict)
    assert y == {k: neg(v) for k, v in x.items()}


@given(x=dictionaries(integers(), integers()))
def test_map_items(x: Dict[int, int]) -> None:
    def func(key: int, value: int) -> Tuple[int, int]:
        return neg(key), neg(value)

    y = CDict(x).map_items(func)
    assert isinstance(y, CDict)
    assert y == {neg(k): neg(v) for k, v in x.items()}
