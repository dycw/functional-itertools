from __future__ import annotations

from operator import neg
from re import escape
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Type
from typing import Union

from hypothesis import assume
from hypothesis import given
from hypothesis.strategies import booleans
from hypothesis.strategies import data
from hypothesis.strategies import DataObject
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
from tests.strategies import CLASSES
from tests.strategies import islice_ints
from tests.strategies import nested_siterables
from tests.strategies import siterables
from tests.test_utilities import is_even


@mark.parametrize("cls", CLASSES)
@given(data=data())
def test_all(cls: Type, data: DataObject) -> None:
    x, _ = data.draw(siterables(cls, booleans()))
    y = cls(x).all()
    assert isinstance(y, bool)
    assert y == all(x)


@mark.parametrize("cls", CLASSES)
@given(data=data())
def test_any(cls: Type, data: DataObject) -> None:
    x, _ = data.draw(siterables(cls, booleans()))
    y = cls(x).any()
    assert isinstance(y, bool)
    assert y == any(x)


@mark.parametrize("cls", CLASSES)
@given(data=data())
def test_dict(cls: Type, data: DataObject) -> None:
    x, _ = data.draw(siterables(cls, tuples(integers(), integers())))
    assert isinstance(cls(x).dict(), CDict)


@mark.parametrize("cls", CLASSES)
@given(data=data(), start=integers())
def test_enumerate(cls: Type, data: DataObject, start: int) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    y = cls(x).enumerate(start=start)
    assert isinstance(y, cls)
    if cls in {CIterable, CList, CTuple}:
        assert cast(y) == cast(enumerate(x, start=start))


@mark.parametrize("cls", CLASSES)
@given(data=data())
def test_filter(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    y = cls(x).filter(is_even)
    assert isinstance(y, cls)
    assert cast(y) == cast(filter(is_even, x))


@mark.parametrize("cls", CLASSES)
@given(data=data())
def test_frozenset(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(siterables(cls, tuples(integers(), integers())))
    y = cls(x).frozenset()
    assert isinstance(y, CFrozenSet)
    assert cast(y) == cast(frozenset(x))


@mark.parametrize("cls", CLASSES)
@given(data=data())
def test_iter(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(siterables(cls, tuples(integers(), integers())))
    y = cls(x).iter()
    assert isinstance(y, CIterable)
    assert cast(y) == cast(iter(x))


@mark.parametrize("cls", [CList, CTuple, CSet, CFrozenSet])
@given(data=data())
def test_len(cls: Type, data: DataObject) -> None:
    x, _ = data.draw(siterables(cls, integers()))
    y = cls(x).len()
    assert isinstance(y, int)
    assert y == len(x)


@mark.parametrize("cls", CLASSES)
@given(data=data())
def test_list(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    y = cls(x).list()
    assert isinstance(y, CList)
    assert cast(y) == cast(list(x))


@mark.parametrize("cls", CLASSES)
@given(data=data())
def test_map(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    y = cls(x).map(neg)
    assert isinstance(y, cls)
    assert cast(y) == cast(map(neg, x))


@mark.parametrize("cls", CLASSES)
@mark.parametrize("func", [max, min])
@given(
    data=data(),
    key_kwargs=just({})
    | (
        just({"key": neg})
        if VERSION is Version.py37
        else fixed_dictionaries({"key": none() | just(neg)})
    ),
    default_kwargs=just({}) | fixed_dictionaries({"default": integers()}),
)
def test_max_and_min(
    cls: Type,
    func: Callable[..., int],
    data: DataObject,
    key_kwargs: Dict[str, int],
    default_kwargs: Dict[str, int],
) -> None:
    x, _ = data.draw(siterables(cls, integers()))
    try:
        y = getattr(cls(x), func.__name__)(**key_kwargs, **default_kwargs)
    except ValueError:
        with raises(
            ValueError, match=escape(f"{func.__name__}() arg is an empty sequence"),
        ):
            func(x, **key_kwargs, **default_kwargs)
    else:
        assert isinstance(y, int)
        assert y == func(x, **key_kwargs, **default_kwargs)


@mark.parametrize("cls", CLASSES)
@given(
    data=data(), start=islice_ints, stop=none() | islice_ints, step=none() | islice_ints,
)
def test_range(
    cls: Type, data: DataObject, start: int, stop: Optional[int], step: Optional[int],
) -> None:
    if step is not None:
        assume((stop is not None) and (step != 0))
    x = cls.range(start, stop, step)
    assert isinstance(x, cls)
    _, cast = data.draw(siterables(cls, none()))
    assert cast(x) == cast(
        range(start, *(() if stop is None else (stop,)), *(() if step is None else (step,))),
    )


@mark.parametrize("cls", CLASSES)
@given(data=data())
def test_set(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    y = cls(x).set()
    assert isinstance(y, CSet)
    assert cast(y) == cast(set(x))


@mark.parametrize("cls", CLASSES)
@given(data=data(), key=none() | just(neg), reverse=booleans())
def test_sorted(
    cls: Type, data: DataObject, key: Optional[Callable[[int], int]], reverse: bool,
) -> None:
    x, _ = data.draw(siterables(cls, integers()))
    y = cls(x).sorted(key=key, reverse=reverse)
    assert isinstance(y, CList)
    assert y == sorted(x, key=key, reverse=reverse)


@mark.parametrize("cls", CLASSES)
@given(data=data(), start=integers() | just(sentinel))
def test_sum(cls: Type, data: DataObject, start: Union[int, Sentinel]) -> None:
    x, _ = data.draw(siterables(cls, integers()))
    y = cls(x).sum(start=start)
    assert isinstance(y, int)
    assert y == sum(x, *(() if start is sentinel else (start,)))


@mark.parametrize("cls", CLASSES)
@given(data=data())
def test_tuple(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    y = cls(x).tuple()
    assert isinstance(y, CTuple)
    assert cast(y) == cast(tuple(x))


@mark.parametrize("cls", CLASSES)
@given(data=data())
def test_zip(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    xs, _ = data.draw(nested_siterables(cls, integers()))
    y = cls(x).zip(*xs)
    assert isinstance(y, cls)
    if cls in {CIterable, CList, CTuple}:
        assert cast(y) == cast(zip(x, *xs))
