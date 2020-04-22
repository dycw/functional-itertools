from __future__ import annotations

from operator import neg
from re import escape
from typing import Callable
from typing import Dict
from typing import FrozenSet
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

from hypothesis import given
from hypothesis.strategies import booleans
from hypothesis.strategies import fixed_dictionaries
from hypothesis.strategies import frozensets
from hypothesis.strategies import integers
from hypothesis.strategies import just
from hypothesis.strategies import none
from hypothesis.strategies import sets
from hypothesis.strategies import tuples
from pytest import mark
from pytest import raises

from functional_itertools import CDict
from functional_itertools import CFrozenSet
from functional_itertools import CIterable
from functional_itertools import CList
from functional_itertools import CSet
from functional_itertools import CTuple
from functional_itertools.utilities import Sentinel
from functional_itertools.utilities import sentinel
from functional_itertools.utilities import VERSION
from functional_itertools.utilities import Version
from tests.strategies import Case
from tests.strategies import CASES
from tests.strategies import range_args
from tests.test_utilities import is_even


@mark.parametrize("case", CASES)
@given(x=sets(booleans()))
def test_all(case: Case, x: Set[bool]) -> None:
    y = case.cls(x).all()
    assert isinstance(y, bool)
    assert y == all(x)


@mark.parametrize("case", CASES)
@given(x=sets(booleans()))
def test_any(case: Case, x: Set[bool]) -> None:
    y = case.cls(x).any()
    assert isinstance(y, bool)
    assert y == any(x)


@mark.parametrize("case", CASES)
@given(x=sets(tuples(integers(), integers())))
def test_dict(case: Case, x: Set[Tuple[int, int]]) -> None:
    y = case.cls(x).dict()
    assert isinstance(y, CDict)
    assert y == dict(x)


@mark.parametrize("case", CASES)
@given(x=sets(integers()), start=integers())
def test_enumerate(case: Case, x: Set[int], start: int) -> None:
    y1, y2 = [case.cls(x).enumerate(start=start) for _ in range(2)]
    assert isinstance(y1, case.cls)
    z1, z2 = [enumerate(x, start=start) for _ in range(2)]
    assert len(list(y1)) == len(list(y2))
    for yi, (zi0, zi1) in zip(y2, z2):
        assert isinstance(yi, CTuple)
        yi0, yi1 = yi
        assert isinstance(yi0, int)
        assert isinstance(yi1, int)
        if case.cls in {CIterable, CList, CTuple}:
            assert yi0 == zi0
            assert yi1 == zi1


@mark.parametrize("case", CASES)
@given(x=sets(integers()))
def test_filter(case: Case, x: Set[int]) -> None:
    y = case.cls(x).filter(is_even)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast(filter(is_even, x))


@mark.parametrize("case", CASES)
@given(x=sets(integers()))
def test_frozenset(case: Case, x: Set[int]) -> None:
    y = case.cls(x).frozenset()
    assert isinstance(y, CFrozenSet)
    assert y == x


@mark.parametrize("case", CASES)
@given(x=sets(integers()))
def test_iter(case: Case, x: Set[int]) -> None:
    y = case.cls(x).iter()
    assert isinstance(y, CIterable)
    assert case.cast(y) == case.cast(x)


@mark.parametrize("case", [case for case in CASES if case.cls is not CIterable])
@given(x=sets(integers()))
def test_len(case: Case, x: Set[int]) -> None:
    y = case.cls(x).len()
    assert isinstance(y, int)
    assert y == len(x)


@mark.parametrize("case", CASES)
@given(x=sets(integers()))
def test_list(case: Case, x: Set[int]) -> None:
    y = case.cls(x).list()
    assert isinstance(y, CList)
    assert case.cast(y) == case.cast(x)


@mark.parametrize("case", CASES)
@given(x=sets(integers()))
def test_map(case: Case, x: Set[int]) -> None:
    y = case.cls(x).map(neg)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast(map(neg, x))


@mark.parametrize("case", CASES)
@mark.parametrize("func", [max, min])
@given(
    x=sets(integers()),
    key=just({})
    | (
        just({"key": neg})
        if VERSION is Version.py37
        else fixed_dictionaries({"key": none() | just(neg)})
    ),
    default=just({}) | fixed_dictionaries({"default": integers()}),
)
def test_max_and_min(
    case: Case, func: Callable[..., int], x: Set[int], key: Dict[str, int], default: Dict[str, int],
) -> None:
    try:
        y = getattr(case.cls(x), func.__name__)(**key, **default)
    except ValueError:
        with raises(
            ValueError, match=escape(f"{func.__name__}() arg is an empty sequence"),
        ):
            func(x, **key, **default)
    else:
        assert isinstance(y, int)
        assert y == func(x, **key, **default)


@mark.parametrize("case", CASES)
@given(args=range_args)
def test_range(case: Case, args: Tuple[int, Optional[int], Optional[int]]) -> None:
    start, stop, step = args
    x = case.cls.range(start, stop, step)
    assert isinstance(x, case.cls)
    assert case.cast(x) == case.cast(
        range(start, *(() if stop is None else (stop,)), *(() if step is None else (step,))),
    )


@mark.parametrize("case", CASES)
@given(x=sets(integers()))
def test_set(case: Case, x: Set[int]) -> None:
    y = case.cls(x).set()
    assert isinstance(y, CSet)
    assert y == x


@mark.parametrize("case", CASES)
@given(x=sets(integers()), key=none() | just(neg), reverse=booleans())
def test_sorted(
    case: Case, x: Set[int], key: Optional[Callable[[int], int]], reverse: bool,
) -> None:
    y = case.cls(x).sorted(key=key, reverse=reverse)
    assert isinstance(y, CList)
    assert y == sorted(x, key=key, reverse=reverse)


@mark.parametrize("case", CASES)
@given(x=sets(integers()), start=integers() | just(sentinel))
def test_sum(case: Case, x: Set[int], start: Union[int, Sentinel]) -> None:
    y = case.cls(x).sum(start=start)
    assert isinstance(y, int)
    assert y == sum(x, *(() if start is sentinel else (start,)))


@mark.parametrize("case", CASES)
@given(x=sets(integers()))
def test_tuple(case: Case, x: Set[int]) -> None:
    y = case.cls(x).tuple()
    assert isinstance(y, CTuple)
    assert case.cast(y) == case.cast(x)


@mark.parametrize("case", CASES)
@given(x=sets(integers()), xs=sets(frozensets(integers())))
def test_zip2(case: Case, x: Set[int], xs: Set[FrozenSet[int]]) -> None:
    y = case.cls(x).zip(*xs)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast(zip(x, *xs))
