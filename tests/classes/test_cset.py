from __future__ import annotations

from itertools import permutations
from re import search
from typing import FrozenSet
from typing import Set
from typing import Type

from hypothesis.strategies import frozensets
from hypothesis.strategies import integers
from hypothesis.strategies import sets
from pytest import warns

from functional_itertools import CFrozenSet
from functional_itertools import CSet
from tests import given
from tests import parametrize


SET_CLASSES = [CSet, CFrozenSet]


# repr and str


@given(x=sets(integers()))
@parametrize("cls", SET_CLASSES)
def test_repr(x: Set[int], cls: Type) -> None:
    y = repr(cls(x))
    name = cls.__name__
    if x:
        assert search(fr"^{name}\(\{{[\d\s\-,]*\}}\)$", y)
    else:
        assert y == f"{name}()"


@given(x=sets(integers()))
@parametrize("cls", SET_CLASSES)
def test_str(x: Set[int], cls: Type) -> None:
    y = str(cls(x))
    name = cls.__name__
    if x:
        assert search(fr"^{name}\(\{{[\d\s\-,]*\}}\)$", y)
    else:
        assert y == f"{name}()"


# set and frozenset methods


@given(x=sets(integers()), y=sets(integers()))
@parametrize("cls", SET_CLASSES)
def test_dunder_and(x: Set[int], y: Set[int], cls: Type) -> None:
    z = cls(x) & y
    assert isinstance(z, cls)
    assert z == (x & y)


@given(x=sets(integers()), y=sets(integers()))
@parametrize("cls", SET_CLASSES)
def test_dunder_or(x: Set[int], y: Set[int], cls: Type) -> None:
    z = cls(x) | y
    assert isinstance(z, cls)
    assert z == (x | y)


@given(x=sets(integers()), y=sets(integers()))
@parametrize("cls", SET_CLASSES)
def test_dunder_sub(x: Set[int], y: Set[int], cls: Type) -> None:
    z = cls(x) - y
    assert isinstance(z, cls)
    assert z == (x - y)


@given(x=sets(integers()), y=sets(integers()))
@parametrize("cls", SET_CLASSES)
def test_dunder_xor(x: Set[int], y: Set[int], cls: Type) -> None:
    z = cls(x) ^ y
    assert isinstance(z, cls)
    assert z == (x ^ y)


@given(x=sets(integers()))
@parametrize("cls", SET_CLASSES)
def test_copy(x: Set[int], cls: Type) -> None:
    y = cls(x).copy()
    assert isinstance(y, cls)
    assert y == x


@given(x=sets(integers()), xs=sets(frozensets(integers())))
@parametrize("cls", SET_CLASSES)
def test_difference(x: Set[int], xs: Set[FrozenSet[int]], cls: Type) -> None:
    y = cls(x).difference(*xs)
    assert isinstance(y, cls)
    assert y == x.difference(*xs)


@given(x=sets(integers()), xs=sets(frozensets(integers())))
@parametrize("cls", SET_CLASSES)
def test_intersection(x: Set[int], xs: Set[FrozenSet[int]], cls: Type) -> None:
    y = cls(x).intersection(*xs)
    assert isinstance(y, cls)
    assert y == x.intersection(*xs)


@given(x=sets(integers()), y=sets(integers()))
@parametrize("cls", SET_CLASSES)
def test_symmetric_difference(x: Set[int], y: Set[int], cls: Type) -> None:
    z = cls(x).symmetric_difference(y)
    assert isinstance(z, cls)
    assert z == x.symmetric_difference(y)


@given(x=sets(integers()), xs=sets(frozensets(integers())))
@parametrize("cls", SET_CLASSES)
def test_union(x: Set[int], xs: Set[FrozenSet[int]], cls: Type) -> None:
    y = cls(x).union(*xs)
    assert isinstance(y, cls)
    assert y == x.union(*xs)


# set methods


@given(x=sets(integers()), y=integers())
def test_add(x: Set[int], y: int) -> None:
    with warns(
        UserWarning, match="CSet.add is a non-functional method",
    ):
        CSet(x).add(y)


@given(x=sets(integers()))
def test_clear(x: Set[int]) -> None:
    with warns(
        UserWarning, match="CSet.clear is a non-functional method",
    ):
        CSet(x).clear()


@given(x=sets(integers()), xs=sets(frozensets(integers())))
def test_difference_update(x: Set[int], xs: Set[FrozenSet[int]]) -> None:
    with warns(
        UserWarning,
        match="CSet.difference_update is a non-functional method; did you mean "
        "CSet.difference instead?",
    ):
        CSet(x).difference_update(*xs)


@given(x=sets(integers()), y=integers())
def test_discard(x: Set[int], y: int) -> None:
    with warns(
        UserWarning, match="CSet.discard is a non-functional method",
    ):
        CSet(x).chain([y]).discard(y)


@given(x=sets(integers()), xs=sets(frozensets(integers())))
def test_intersection_update(x: Set[int], xs: Set[FrozenSet[int]]) -> None:
    with warns(
        UserWarning,
        match="CSet.intersection_update is a non-functional method; did you "
        "mean CSet.intersection instead?",
    ):
        CSet(x).intersection_update(*xs)


@given(x=sets(integers(), min_size=1))
def test_pop(x: Set[int]) -> None:
    with warns(
        UserWarning, match="CSet.pop is a non-functional method",
    ):
        CSet(x).pop()


@given(x=sets(integers()), y=integers())
def test_remove(x: Set[int], y: int) -> None:
    with warns(
        UserWarning, match="CSet.remove is a non-functional method",
    ):
        CSet(x).chain([y]).remove(y)


@given(x=sets(integers()), y=sets(integers()))
def test_symmetric_difference_update(x: Set[int], y: Set[int]) -> None:
    with warns(
        UserWarning,
        match="CSet.symmetric_difference_update is a non-functional method; "
        "did you mean CSet.symmetric_difference instead?",
    ):
        CSet(x).symmetric_difference_update(y)


@given(x=sets(integers()), xs=sets(frozensets(integers())))
def test_update(x: Set[int], xs: Set[FrozenSet[int]]) -> None:
    with warns(
        UserWarning,
        match="CSet.update is a non-functional method; did you mean CSet.union "
        "instead?",
    ):
        CSet(x).update(*xs)


# extra public


@given(x=sets(integers()))
def test_pipe(x: Set[int]) -> None:
    y = CSet(x).pipe(permutations, r=2)
    assert isinstance(y, CSet)
    assert y == set(permutations(x, r=2))
