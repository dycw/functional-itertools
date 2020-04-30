from __future__ import annotations

from operator import neg
from typing import Any
from typing import Callable
from typing import Dict
from typing import TypeVar

from attr import Attribute
from attr import attrs
from pytest import mark

from functional_itertools import CAttrs
from functional_itertools import CDict
from functional_itertools import CTuple


T = TypeVar("T")


@attrs(auto_attribs=True)
class Foo(CAttrs[T]):
    a: T
    b: T
    c: T


@attrs(auto_attribs=True)
class Bar(CAttrs[T]):
    d: T
    e: T
    f: T


def test_dict_simple() -> None:
    x = Foo(a=1, b=2, c=3).dict()
    assert isinstance(x, CDict)
    assert x == {"a": 1, "b": 2, "c": 3}


@mark.parametrize(
    "recurse, expected",
    [
        (True, {"a": 1, "b": 2, "c": {"d": 3, "e": 4, "f": 5}}),
        (False, {"a": 1, "b": 2, "c": Bar(d=3, e=4, f=5)}),
    ],
)
def test_dict_nested_recurse(recurse: bool, expected: Dict[str, Any]) -> None:
    x = Foo(a=1, b=2, c=Bar(d=3, e=4, f=5)).dict(recurse=recurse)
    assert isinstance(x, CDict)
    assert x == expected


@mark.parametrize(
    "filter, expected",
    [
        (lambda k, v: k.name == "a", {"a": 1}),
        (lambda k, v: v >= 2, {"b": 2, "c": 3}),
        (lambda k, v: k.name == "a" or v == 3, {"a": 1, "c": 3}),
    ],
)
def test_dict_simple_filter(
    filter: Callable[[Attribute, Any], bool], expected: Dict[str, Any],  # noqa: A002
) -> None:
    x = Foo(a=1, b=2, c=3).dict(filter=filter)
    assert isinstance(x, CDict)
    assert x == expected


@mark.parametrize(
    "filter, expected",
    [
        (lambda k, v: k.name == "a", {"a": 1}),
        (lambda k, v: k.name == "c", {"c": {}}),
        (lambda k, v: isinstance(v, int) and (v >= 2), {"b": 2}),
        (
            lambda k, v: (isinstance(v, int) and ((v <= 1) or (v >= 5))) or isinstance(v, Bar),
            {"a": 1, "c": {"f": 5}},
        ),
    ],
)
def test_dict_nested_filter(
    filter: Callable[[Attribute, Any], bool], expected: Dict[str, Any],  # noqa: A002
) -> None:
    x = Foo(a=1, b=2, c=Bar(d=3, e=4, f=5)).dict(filter=filter)
    assert isinstance(x, CDict)
    assert x == expected


def test_map_simple() -> None:
    x = Foo(a=1, b=2, c=3).map(neg)
    assert isinstance(x, Foo)
    assert x == Foo(a=-1, b=-2, c=-3)


@mark.parametrize(
    "recurse, expected",
    [
        (True, Foo(a=-1, b=-2, c=Bar(d=-3, e=-4, f=-5))),
        (False, Foo(a=-1, b=-2, c=Bar(d=3, e=4, f=5))),
    ],
)
def test_map_nested(recurse: bool, expected: Any) -> None:
    x = Foo(a=1, b=2, c=Bar(d=3, e=4, f=5)).map(neg, recurse=recurse)
    assert isinstance(x, Foo)
    assert x == expected


def test_tuple_simple() -> None:
    x = Foo(a=1, b=2, c=3).tuple()
    assert isinstance(x, CTuple)
    assert x == (1, 2, 3)


@mark.parametrize(
    "recurse, expected", [(True, (1, 2, (3, 4, 5))), (False, (1, 2, Bar(d=3, e=4, f=5)))],
)
def test_tuple_nested_recurse(recurse: bool, expected: tuple[str, Any]) -> None:
    x = Foo(a=1, b=2, c=Bar(d=3, e=4, f=5)).tuple(recurse=recurse)
    assert isinstance(x, CTuple)
    assert x == expected


@mark.parametrize(
    "filter, expected",
    [
        (lambda k, v: k.name == "a", (1,)),
        (lambda k, v: v >= 2, (2, 3)),
        (lambda k, v: k.name == "a" or v == 3, (1, 3)),
    ],
)
def test_tuple_simple_filter(
    filter: Callable[[Attribute, Any], bool], expected: tuple[str, Any],  # noqa: A002
) -> None:
    x = Foo(a=1, b=2, c=3).tuple(filter=filter)
    assert isinstance(x, CTuple)
    assert x == expected


@mark.parametrize(
    "filter, expected",
    [
        (lambda k, v: k.name == "a", (1,)),
        (lambda k, v: k.name == "c", ((),)),
        (lambda k, v: isinstance(v, int) and (v >= 2), (2,)),
        (
            lambda k, v: (isinstance(v, int) and ((v <= 1) or (v >= 5))) or isinstance(v, Bar),
            (1, (5,)),
        ),
    ],
)
def test_tuple_nested_filter(
    filter: Callable[[Attribute, Any], bool], expected: tuple[str, Any],  # noqa: A002
) -> None:
    x = Foo(a=1, b=2, c=Bar(d=3, e=4, f=5)).tuple(filter=filter)
    assert isinstance(x, CTuple)
    assert x == expected
