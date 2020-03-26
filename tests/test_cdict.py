from typing import Callable
from typing import Dict
from typing import Tuple

from hypothesis import given
from hypothesis import infer
from hypothesis.strategies import booleans
from hypothesis.strategies import dictionaries
from hypothesis.strategies import integers

from functional_itertools.classes import CDict
from functional_itertools.classes import CFrozenSet
from functional_itertools.classes import CIterable
from functional_itertools.classes import CList
from functional_itertools.classes import CSet
from tests.test_utilities import int_and_int_to_bool_funcs
from tests.test_utilities import int_and_int_to_int_and_int_funcs
from tests.test_utilities import int_to_bool_funcs
from tests.test_utilities import int_to_int_funcs


@given(x=infer)
def test_keys(x: Dict[str, int]) -> None:
    y = CDict(x).keys()
    assert isinstance(y, CIterable)
    assert list(y) == list(x.keys())


@given(x=infer)
def test_values(x: Dict[str, int]) -> None:
    y = CDict(x).values()
    assert isinstance(y, CIterable)
    assert list(y) == list(x.values())


@given(x=infer)
def test_items(x: Dict[str, int]) -> None:
    y = CDict(x).items()
    assert isinstance(y, CIterable)
    assert list(y) == list(x.items())


# built-in


@given(x=dictionaries(booleans(), integers()))
def test_all_keys(x: Dict[bool, int]) -> None:
    y = CDict(x).all_keys()
    assert isinstance(y, bool)
    assert y == all(x.keys())


@given(x=dictionaries(integers(), booleans()))
def test_all_values(x: Dict[int, bool]) -> None:
    y = CDict(x).all_values()
    assert isinstance(y, bool)
    assert y == all(x.values())


@given(x=dictionaries(integers(), booleans()))
def test_all_items(x: Dict[int, int]) -> None:
    y = CDict(x).all_items()
    assert isinstance(y, bool)
    assert y == all(x.items())


@given(x=dictionaries(booleans(), integers()))
def test_any_keys(x: Dict[bool, int]) -> None:
    y = CDict(x).any_keys()
    assert isinstance(y, bool)
    assert y == any(x.keys())


@given(x=dictionaries(integers(), booleans()))
def test_any_values(x: Dict[bool, bool]) -> None:
    y = CDict(x).any_values()
    assert isinstance(y, bool)
    assert y == any(x.values())


@given(x=dictionaries(integers(), booleans()))
def test_any_items(x: Dict[int, bool]) -> None:
    y = CDict(x).any_items()
    assert isinstance(y, bool)
    assert y == any(x.items())


@given(x=dictionaries(integers(), integers()), func=int_to_bool_funcs)
def test_filter_keys(x: Dict[int, int], func: Callable[[int], bool]) -> None:
    y = CDict(x).filter_keys(func)
    assert isinstance(y, CDict)
    assert y == {k: v for k, v in x.items() if func(k)}


@given(x=dictionaries(integers(), integers()), func=int_to_bool_funcs)
def test_filter_values(x: Dict[int, int], func: Callable[[int], bool]) -> None:
    y = CDict(x).filter_values(func)
    assert isinstance(y, CDict)
    assert y == {k: v for k, v in x.items() if func(v)}


@given(x=dictionaries(integers(), integers()), func=int_and_int_to_bool_funcs)
def test_filter_items(x: Dict[int, int], func: Callable[[int, int], bool]) -> None:
    y = CDict(x).filter_items(func)
    assert isinstance(y, CDict)
    assert y == {k: v for k, v in x.items() if func(k, v)}


@given(x=dictionaries(integers(), integers()))
def test_frozenset_keys(x: Dict[int, int]) -> None:
    y = CDict(x).frozenset_keys()
    assert isinstance(y, CFrozenSet)
    assert y == frozenset(x.keys())


@given(x=dictionaries(integers(), integers()))
def test_frozenset_values(x: Dict[int, int]) -> None:
    y = CDict(x).frozenset_values()
    assert isinstance(y, CFrozenSet)
    assert y == frozenset(x.values())


@given(x=dictionaries(integers(), integers()))
def test_frozenset_items(x: Dict[int, int]) -> None:
    y = CDict(x).frozenset_items()
    assert isinstance(y, CFrozenSet)
    assert y == frozenset(x.items())


@given(x=dictionaries(integers(), integers()))
def test_list_keys(x: Dict[int, int]) -> None:
    y = CDict(x).list_keys()
    assert isinstance(y, CList)
    assert y == list(x.keys())


@given(x=dictionaries(integers(), integers()))
def test_list_values(x: Dict[int, int]) -> None:
    y = CDict(x).list_values()
    assert isinstance(y, CList)
    assert y == list(x.values())


@given(x=dictionaries(integers(), integers()))
def test_list_items(x: Dict[int, int]) -> None:
    y = CDict(x).list_items()
    assert isinstance(y, CList)
    assert y == list(x.items())


@given(x=dictionaries(integers(), integers()), func=int_to_int_funcs)
def test_map_keys(x: Dict[int, int], func: Callable[[int], int]) -> None:
    y = CDict(x).map_keys(func)
    assert isinstance(y, CDict)
    assert y == {func(k): v for k, v in x.items()}


@given(x=dictionaries(integers(), integers()), func=int_to_int_funcs)
def test_map_values(x: Dict[int, int], func: Callable[[int], int]) -> None:
    y = CDict(x).map_values(func)
    assert isinstance(y, CDict)
    assert y == {k: func(v) for k, v in x.items()}


@given(
    x=dictionaries(integers(), integers()), func=int_and_int_to_int_and_int_funcs,
)
def test_map_items(x: Dict[int, int], func: Callable[[int, int], Tuple[int, int]]) -> None:
    y = CDict(x).map_items(func)
    assert isinstance(y, CDict)
    assert y == dict(func(k, v) for k, v in x.items())


@given(x=dictionaries(integers(), integers()))
def test_set_keys(x: Dict[int, int]) -> None:
    y = CDict(x).set_keys()
    assert isinstance(y, CSet)
    assert y == set(x.keys())


@given(x=infer)
def test_set_values(x: Dict[int, int]) -> None:
    y = CDict(x).set_values()
    assert isinstance(y, CSet)
    assert y == set(x.values())


@given(x=infer)
def test_set_items(x: Dict[int, int]) -> None:
    y = CDict(x).set_items()
    assert isinstance(y, CSet)
    assert y == set(x.items())
