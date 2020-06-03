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
from hypothesis.strategies import booleans
from hypothesis.strategies import integers
from hypothesis.strategies import just
from hypothesis.strategies import lists
from hypothesis.strategies import none
from hypothesis.strategies import tuples
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
from tests import given
from tests import parametrize
from tests.strategies import Case
from tests.strategies import CASES
from tests.strategies import MAX_SIZE
from tests.test_utilities import is_even
from tests.test_utilities import sum_varargs


@given(x=lists(booleans()))
@parametrize("case", CASES)
def test_all(x: List[bool], case: Case) -> None:
    y = case.cls(x).all()
    assert isinstance(y, bool)
    assert y == all(x)


@given(x=lists(booleans()))
@parametrize("case", CASES)
def test_any(x: List[bool], case: Case) -> None:
    y = case.cls(x).any()
    assert isinstance(y, bool)
    assert y == any(x)


@given(x=lists(tuples(integers(), integers())))
@parametrize("case", CASES)
def test_dict(x: List[Tuple[int, int]], case: Case) -> None:
    y = case.cls(x).dict()
    assert isinstance(y, CDict)
    assert y == dict(case.cast(x))


@given(x=lists(integers()), start=integers())
@parametrize("case", CASES)
def test_enumerate(x: List[int], start: int, case: Case) -> None:
    y = case.cls(x).enumerate(start=start)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast(enumerate(case.cast(x), start=start))


@given(x=lists(integers()))
@parametrize("case", CASES)
def test_filter(x: List[int], case: Case) -> None:
    y = case.cls(x).filter(is_even)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast(filter(is_even, x))


@given(x=lists(integers()))
@parametrize("case", CASES)
def test_frozenset(x: List[int], case: Case) -> None:
    y = case.cls(x).frozenset()
    assert isinstance(y, CFrozenSet)
    assert y == frozenset(y)


@given(x=lists(integers()))
@parametrize("case", CASES)
def test_iter(x: List[int], case: Case) -> None:
    y = case.cls(x).iter()
    assert isinstance(y, CIterable)
    assert case.cast(y) == case.cast(iter(x))


@given(x=lists(integers()))
@parametrize("case", CASES)
def test_len(x: List[int], case: Case) -> None:
    y = case.cls(x)
    if case.cls is CIterable:
        with raises(
            AttributeError, match="'CIterable' object has no attribute 'len'",
        ):
            y.len()
    else:
        z = y.len()
        assert isinstance(z, int)
        assert z == len(case.cast(x))


@given(x=lists(integers()))
@parametrize("case", CASES)
def test_list(x: List[int], case: Case) -> None:
    y = case.cls(x).list()
    assert isinstance(y, CList)
    assert case.cast(y) == case.cast(x)


@given(
    x=lists(integers()), xs=lists(lists(integers())),
)
@parametrize("case", CASES)
@parametrize("kwargs", [{}, {"parallel": True, "processes": 1}])
def test_map(
    x: List[int], xs: List[List[int]], kwargs: Dict[str, Any], case: Case,
) -> None:
    y = case.cls(x).map(sum_varargs, *xs, **kwargs)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast(map(sum_varargs, case.cast(x), *xs))


@given(
    x=lists(integers()),
    key=none() | just(neg),
    default=just(sentinel) | integers(),
)
@parametrize("case", CASES)
@parametrize("func", [max, min])
def test_max_and_min(
    case: Case,
    func: Callable[..., int],
    x: List[int],
    key: Optional[Callable[[int], int]],
    default: Union[int, Sentinel],
) -> None:
    name = func.__name__
    _, kwargs = drop_sentinel(default=default)
    try:
        y = getattr(case.cls(x), name)(key=key, default=default)
    except ValueError:
        with raises(
            ValueError, match=escape(f"{name}() arg is an empty sequence"),
        ):
            func(x, key=key, **kwargs)
    else:
        assert isinstance(y, int)
        assert y == func(x, key=key, **kwargs)


@given(
    start=integers(0, MAX_SIZE),
    stop=none() | integers(0, MAX_SIZE),
    step=none() | integers(1, 10),
)
@parametrize("case", CASES)
def test_range(
    case: Case, start: int, stop: Optional[int], step: Optional[int],
) -> None:
    if step is not None:
        assume(stop is not None)
    x = case.cls.range(start, stop, step)
    assert isinstance(x, case.cls)
    args, _ = drop_none(stop, step)
    assert case.cast(x) == case.cast(range(start, *args))


@given(x=lists(integers()))
@parametrize("case", CASES)
def test_set(x: List[int], case: Case) -> None:
    y = case.cls(x).set()
    assert isinstance(y, CSet)
    assert y == set(x)


@given(x=lists(integers()), key=none() | just(neg), reverse=booleans())
@parametrize("case", CASES)
def test_sorted(
    x: List[int],
    key: Optional[Callable[[int], int]],
    reverse: bool,
    case: Case,
) -> None:
    y = case.cls(x).sorted(key=key, reverse=reverse)
    assert isinstance(y, CTuple if case.cls is CTuple else CList)
    assert case.cast(y) == case.cast(sorted(x, key=key, reverse=reverse))


@given(x=lists(integers()), start=integers() | just(sentinel))
@parametrize("case", CASES)
def test_sum(x: List[int], start: Union[int, Sentinel], case: Case) -> None:
    y = case.cls(x).sum(start=start)
    assert isinstance(y, int)
    args, _ = drop_sentinel(start)
    assert y == sum(case.cast(x), *args)


@given(x=lists(integers()))
@parametrize("case", CASES)
def test_tuple(x: List[int], case: Case) -> None:
    y = case.cls(x).tuple()
    assert isinstance(y, CTuple)
    assert y == tuple(case.cast(x))


@given(x=lists(integers()), xs=lists(lists(integers())))
@parametrize("case", CASES)
def test_zip(x: List[int], xs: List[List[int]], case: Case) -> None:
    y = case.cls(x).zip(*xs)
    assert isinstance(y, case.cls)
    z = list(y)
    for zi in z:
        assert isinstance(zi, CTuple)
    assert case.cast(z) == case.cast(zip(case.cast(x), *xs))
