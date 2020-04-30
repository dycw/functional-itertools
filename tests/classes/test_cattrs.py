from __future__ import annotations

from operator import neg
from typing import TypeVar

from attr import attrs

from functional_itertools import CAttrs
from functional_itertools import CDict


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


def test_dict_with_recurse() -> None:
    x = Foo(a=1, b=2, c=Bar(d=3, e=4, f=5)).dict(recurse=True)
    assert isinstance(x, CDict)
    assert x == {"a": 1, "b": 2, "c": {"d": 3, "e": 4, "f": 5}}
    assert isinstance(x["c"], CDict)


def test_dict_without_recurse() -> None:
    x = Foo(a=1, b=2, c=Bar(d=3, e=4, f=5)).dict(recurse=False)
    assert isinstance(x, CDict)
    assert x == {"a": 1, "b": 2, "c": Bar(d=3, e=4, f=5)}


def test_dict_with_filter() -> None:
    x = Foo(a=1, b=2, c=3).dict(filter=lambda k, v: k.name == "a" or v == 2)
    assert isinstance(x, CDict)
    assert x == {"a": 1, "b": 2}


def test_map_simple() -> None:
    x = Foo(a=1, b=2, c=3).map(neg)
    assert isinstance(x, Foo)
    assert x == Foo(a=-1, b=-2, c=-3)


def test_map_with_recurse() -> None:
    x = Foo(a=1, b=2, c=Bar(d=3, e=4, f=5)).map(neg)
    assert isinstance(x, Foo)
    assert x == Foo(a=-1, b=-2, c=Bar(d=-3, e=-4, f=-5))


def test_map_without_recurse() -> None:
    x = Foo(a=1, b=2, c=Bar(d=3, e=4, f=5)).map(neg, recurse=False)
    assert isinstance(x, Foo)
    assert x == Foo(a=-1, b=-2, c=Bar(d=3, e=4, f=5))
