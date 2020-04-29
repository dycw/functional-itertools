from __future__ import annotations

from operator import neg
from re import escape
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

from hypothesis import given
from hypothesis.strategies import booleans
from hypothesis.strategies import fixed_dictionaries
from hypothesis.strategies import integers
from hypothesis.strategies import just
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
from functional_itertools.utilities import Sentinel
from functional_itertools.utilities import sentinel
from functional_itertools.utilities import VERSION
from functional_itertools.utilities import Version
from tests.strategies import Case
from tests.strategies import CASES
from tests.strategies import CLASSES
from tests.strategies import get_cast
from tests.strategies import ORDERED_CLASSES
from tests.strategies import range_args
from tests.strategies import real_iterables
from tests.test_utilities import is_even


@mark.parametrize("cls", CLASSES)
@given(x=real_iterables(booleans()))
def test_all(cls: Type, x: Iterable[bool]) -> None:
    y = cls(x).all()
    assert isinstance(y, bool)
    assert y == all(x)


@mark.parametrize("cls", CLASSES)
@given(x=real_iterables(booleans()))
def test_any(cls: Type, x: Iterable[bool]) -> None:
    y = cls(x).any()
    assert isinstance(y, bool)
    assert y == any(x)


@mark.parametrize("case", CASES)
@given(x=real_iterables(tuples(integers(), integers())))
def test_dict(case: Case, x: Iterable[Tuple[int, int]]) -> None:
    y = case.cls(x).dict()
    assert isinstance(y, CDict)
    if case.ordered:
        assert y == dict(x)


@mark.parametrize("case", CASES)
@given(x=real_iterables(integers()), start=integers())
def test_enumerate(case: Case, x: Iterable[int], start: int) -> None:
    y = case.cls(x).enumerate(start=start)
    assert isinstance(y, case.cls)
    if case.ordered:
        assert case.cast(y) == case.cast(enumerate(x, start=start))


@mark.parametrize("case", CASES)
@given(x=real_iterables(integers()))
def test_filter(case: Case, x: Iterable[int]) -> None:
    y = case.cls(x).filter(is_even)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast(filter(is_even, x))


@mark.parametrize("cls", CLASSES)
@given(x=real_iterables(integers()))
def test_frozenset(cls: Type, x: Iterable[int]) -> None:
    y = cls(x).frozenset()
    assert isinstance(y, CFrozenSet)
    assert y == frozenset(y)


@mark.parametrize("cls", CLASSES)
@given(x=real_iterables(integers()))
def test_iter(cls: Type, x: Iterable[int]) -> None:
    y = cls(x).iter()
    assert isinstance(y, CIterable)
    cast = get_cast(cls)
    assert cast(y) == cast(iter(x))


@mark.parametrize("cls", [cls for cls in CLASSES if cls is not CIterable])
@given(x=real_iterables(integers()))
def test_len(cls: Type, x: Iterable[int]) -> None:
    y = cls(x).len()
    assert isinstance(y, int)
    assert y == len(x)


@mark.parametrize("cls", CLASSES)
@given(x=real_iterables(integers()))
def test_list(cls: Type, x: Iterable[int]) -> None:
    y = cls(x).list()
    assert isinstance(y, CList)
    cast = get_cast(cls)
    assert cast(y) == cast(x)


@mark.parametrize("case", CASES)
@mark.parametrize("kwargs", [{}, {"parallel": True, "processes": 1}])
@given(x=real_iterables(integers()))
def test_map(case: Case, x: Iterable[int], kwargs: Dict[str, Any]) -> None:
    y = case.cls(x).map(neg, **kwargs)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast(map(neg, x))


@mark.parametrize("cls", CLASSES)
@mark.parametrize("func", [max, min])
@given(
    x=real_iterables(integers()),
    key=just({})
    | (
        just({"key": neg})
        if VERSION is Version.py37
        else fixed_dictionaries({"key": none() | just(neg)})
    ),
    default=just({}) | fixed_dictionaries({"default": integers()}),
)
def test_max_and_min(
    cls: Type,
    func: Callable[..., int],
    x: Iterable[int],
    key: Dict[str, int],
    default: Dict[str, int],
) -> None:
    try:
        y = getattr(cls(x), func.__name__)(**key, **default)
    except ValueError:
        with raises(
            ValueError, match=escape(f"{func.__name__}() arg is an empty sequence"),
        ):
            func(x, **key, **default)
    else:
        assert isinstance(y, int)
        assert y == func(x, **key, **default)


@mark.parametrize("cls", CLASSES)
@given(args=range_args)
def test_range(cls: Type, args: Tuple[int, Optional[int], Optional[int]]) -> None:
    start, stop, step = args
    x = cls.range(start, stop, step)
    assert isinstance(x, cls)
    cast = get_cast(cls)
    assert cast(x) == cast(
        range(start, *(() if stop is None else (stop,)), *(() if step is None else (step,))),
    )


@mark.parametrize("cls", CLASSES)
@given(x=real_iterables(integers()))
def test_set(cls: Type, x: Iterable[int]) -> None:
    y = cls(x).set()
    assert isinstance(y, CSet)
    assert y == set(x)


@mark.parametrize("cls", CLASSES)
@given(x=real_iterables(integers()), key=none() | just(neg), reverse=booleans())
def test_sorted(
    cls: Type, x: Iterable[int], key: Optional[Callable[[int], int]], reverse: bool,
) -> None:
    y = cls(x).sorted(key=key, reverse=reverse)
    assert isinstance(y, CList)
    if cls in ORDERED_CLASSES:
        assert y == sorted(x, key=key, reverse=reverse)


@mark.parametrize("cls", CLASSES)
@given(x=real_iterables(integers()), start=integers() | just(sentinel))
def test_sum(cls: Type, x: Iterable[int], start: Union[int, Sentinel]) -> None:
    y = cls(x).sum(start=start)
    assert isinstance(y, int)
    if cls in ORDERED_CLASSES:
        assert y == sum(x, *(() if start is sentinel else (start,)))


@mark.parametrize("cls", CLASSES)
@given(x=real_iterables(integers()))
def test_tuple(cls: Type, x: Iterable[int]) -> None:
    y = cls(x).tuple()
    assert isinstance(y, CTuple)
    if cls in ORDERED_CLASSES:
        assert y == tuple(x)


@mark.parametrize("case", CASES)
@given(x=real_iterables(integers()), xs=real_iterables(real_iterables(integers())))
def test_zip(case: Case, x: Iterable[int], xs: Iterable[Iterable[int]]) -> None:
    y = case.cls(x).zip(*xs)
    assert isinstance(y, case.ordered_cls)
    z = list(y)
    for zi in z:
        assert isinstance(zi, CTuple)
    if case.ordered:
        assert case.cast(z) == case.cast(zip(x, *xs))
