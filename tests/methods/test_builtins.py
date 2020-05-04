from __future__ import annotations

from operator import neg
from re import escape
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from hypothesis import assume
from hypothesis import given
from hypothesis.strategies import booleans
from hypothesis.strategies import fixed_dictionaries
from hypothesis.strategies import integers
from hypothesis.strategies import just
from hypothesis.strategies import lists
from hypothesis.strategies import none
from hypothesis.strategies import tuples
from pytest import mark
from pytest import raises

from functional_itertools import CDict
from functional_itertools import CFrozenSet
from functional_itertools import CIterable
from functional_itertools import CList
from functional_itertools import CSet
from functional_itertools import CTuple
from functional_itertools.utilities import drop_none
from functional_itertools.utilities import drop_sentinel
from functional_itertools.utilities import Sentinel
from functional_itertools.utilities import sentinel
from functional_itertools.utilities import VERSION
from functional_itertools.utilities import Version
from tests.strategies import Case
from tests.strategies import CASES
from tests.strategies import MAX_SIZE
from tests.test_utilities import is_even
from tests.test_utilities import sum_varargs


@mark.parametrize("case", CASES)
@given(x=lists(booleans()))
def test_all(case: Case, x: List[bool]) -> None:
    y = case.cls(x).all()
    assert isinstance(y, bool)
    assert y == all(x)


@mark.parametrize("case", CASES)
@given(x=lists(booleans()))
def test_any(case: Case, x: List[bool]) -> None:
    y = case.cls(x).any()
    assert isinstance(y, bool)
    assert y == any(x)


@mark.parametrize("case", CASES)
@given(x=lists(tuples(integers(), integers())))
def test_dict(case: Case, x: List[Tuple[int, int]]) -> None:
    y = case.cls(x).dict()
    assert isinstance(y, CDict)
    assert y == dict(case.cast(x))


@mark.parametrize("case", CASES)
@given(x=lists(integers()), start=integers())
def test_enumerate(case: Case, x: List[int], start: int) -> None:
    y = case.cls(x).enumerate(start=start)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast(enumerate(case.cast(x), start=start))


@mark.parametrize("case", CASES)
@given(x=lists(integers()))
def test_filter(case: Case, x: List[int]) -> None:
    y = case.cls(x).filter(is_even)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast(filter(is_even, x))


@mark.parametrize("case", CASES)
@given(x=lists(integers()))
def test_frozenset(case: Case, x: List[int]) -> None:
    y = case.cls(x).frozenset()
    assert isinstance(y, CFrozenSet)
    assert y == frozenset(y)


@mark.parametrize("case", CASES)
@given(x=lists(integers()))
def test_iter(case: Case, x: List[int]) -> None:
    y = case.cls(x).iter()
    assert isinstance(y, CIterable)
    assert case.cast(y) == case.cast(iter(x))


@mark.parametrize("case", CASES)
@given(x=lists(integers()))
def test_len(case: Case, x: List[int]) -> None:
    y = case.cls(x)
    if case.cls is CIterable:
        with raises(AttributeError, match="'CIterable' object has no attribute 'len'"):
            y.len()
    else:
        z = y.len()
        assert isinstance(z, int)
        assert z == len(case.cast(x))


@mark.parametrize("case", CASES)
@given(x=lists(integers()))
def test_list(case: Case, x: List[int]) -> None:
    y = case.cls(x).list()
    assert isinstance(y, CList)
    assert case.cast(y) == case.cast(x)


@mark.parametrize("case", CASES)
@mark.parametrize("kwargs", [{}, {"parallel": True, "processes": 1}])
@given(
    x=lists(integers()), xs=lists(lists(integers())),
)
def test_map(case: Case, x: List[int], xs: List[List[int]], kwargs: Dict[str, Any]) -> None:
    y = case.cls(x).map(sum_varargs, *xs, **kwargs)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast(map(sum_varargs, case.cast(x), *xs))


@mark.parametrize("case", CASES)
@mark.parametrize("func", [max, min])
@given(
    x=lists(integers()),
    key=just({})
    | (
        just({"key": neg})
        if VERSION is Version.py37
        else fixed_dictionaries({"key": none() | just(neg)})
    ),
    default=just({}) | fixed_dictionaries({"default": integers()}),
)
def test_max_and_min(
    case: Case,
    func: Callable[..., int],
    x: List[int],
    key: Dict[str, int],
    default: Dict[str, int],
) -> None:
    name = func.__name__
    try:
        y = getattr(case.cls(x), name)(**key, **default)
    except ValueError:
        with raises(
            ValueError, match=escape(f"{name}() arg is an empty sequence"),
        ):
            func(x, **key, **default)
    else:
        assert isinstance(y, int)
        assert y == func(x, **key, **default)


@mark.parametrize("case", CASES)
@given(
    start=integers(0, MAX_SIZE), stop=none() | integers(0, MAX_SIZE), step=none() | integers(1, 10),
)
def test_range(case: Case, start: int, stop: Optional[int], step: Optional[int]) -> None:
    if step is not None:
        assume(stop is not None)
    x = case.cls.range(start, stop, step)
    assert isinstance(x, case.cls)
    args, _ = drop_none(stop, step)
    assert case.cast(x) == case.cast(range(start, *args))


@mark.parametrize("case", CASES)
@given(x=lists(integers()))
def test_set(case: Case, x: List[int]) -> None:
    y = case.cls(x).set()
    assert isinstance(y, CSet)
    assert y == set(x)


@mark.parametrize("case", CASES)
@given(x=lists(integers()), key=none() | just(neg), reverse=booleans())
def test_sorted(
    case: Case, x: List[int], key: Optional[Callable[[int], int]], reverse: bool,
) -> None:
    y = case.cls(x).sorted(key=key, reverse=reverse)
    assert isinstance(y, CTuple if case.cls is CTuple else CList)
    assert case.cast(y) == case.cast(sorted(x, key=key, reverse=reverse))


@mark.parametrize("case", CASES)
@given(x=lists(integers()), start=integers() | just(sentinel))
def test_sum(case: Case, x: List[int], start: Union[int, Sentinel]) -> None:
    y = case.cls(x).sum(start=start)
    assert isinstance(y, int)
    args, _ = drop_sentinel(start)
    assert y == sum(case.cast(x), *args)


@mark.parametrize("case", CASES)
@given(x=lists(integers()))
def test_tuple(case: Case, x: List[int]) -> None:
    y = case.cls(x).tuple()
    assert isinstance(y, CTuple)
    assert y == tuple(case.cast(x))


@mark.parametrize("case", CASES)
@given(x=lists(integers()), xs=lists(lists(integers())))
def test_zip(case: Case, x: List[int], xs: List[List[int]]) -> None:
    y = case.cls(x).zip(*xs)
    assert isinstance(y, case.cls)
    z = list(y)
    for zi in z:
        assert isinstance(zi, CTuple)
    assert case.cast(z) == case.cast(zip(case.cast(x), *xs))
