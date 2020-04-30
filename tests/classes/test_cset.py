from __future__ import annotations

from itertools import chain
from itertools import permutations
from re import search
from typing import FrozenSet
from typing import Set
from typing import Type

from hypothesis import given
from hypothesis.strategies import frozensets
from hypothesis.strategies import integers
from hypothesis.strategies import sets
from pytest import mark
from pytest import raises
from pytest import warns

from functional_itertools import CFrozenSet
from functional_itertools import CSet


SET_CLASSES = [CSet, CFrozenSet]


# repr and str


@mark.parametrize("cls", SET_CLASSES)
@given(x=sets(integers()))
def test_repr(cls: Type, x: Set[int]) -> None:
    y = repr(cls(x))
    name = cls.__name__
    if x:
        assert search(fr"^{name}\(\{{[\d\s\-,]*\}}\)$", y)
    else:
        assert y == f"{name}()"


@mark.parametrize("cls", SET_CLASSES)
@given(x=sets(integers()))
def test_str(cls: Type, x: Set[int]) -> None:
    y = str(cls(x))
    name = cls.__name__
    if x:
        assert search(fr"^{name}\(\{{[\d\s\-,]*\}}\)$", y)
    else:
        assert y == f"{name}()"


# set and frozenset methods


@mark.parametrize("cls", SET_CLASSES)
@given(x=sets(integers()), xs=sets(frozensets(integers())))
def test_union(cls: type, x: Set[int], xs: Set[FrozenSet[int]]) -> None:
    y = cls(x).union(*xs)
    assert isinstance(y, cls)
    assert y == x.union(*xs)


@mark.parametrize("cls", SET_CLASSES)
@given(x=sets(integers()), xs=sets(frozensets(integers())))
def test_intersection(cls: Type, x: Set[int], xs: Set[FrozenSet[int]]) -> None:
    y = cls(x).intersection(*xs)
    assert isinstance(y, cls)
    assert y == x.intersection(*xs)


@mark.parametrize("cls", SET_CLASSES)
@given(x=sets(integers()), xs=sets(frozensets(integers())))
def test_difference(cls: Type, x: Set[int], xs: Set[FrozenSet[int]]) -> None:
    y = cls(x).difference(*xs)
    assert isinstance(y, cls)
    assert y == x.difference(*xs)


@mark.parametrize("cls", SET_CLASSES)
@given(x=sets(integers()), y=sets(integers()))
def test_symmetric_difference(cls: Type, x: Set[int], y: Set[int]) -> None:
    z = cls(x).symmetric_difference(y)
    assert isinstance(z, cls)
    assert z == x.symmetric_difference(y)


@mark.parametrize("cls", SET_CLASSES)
@given(x=sets(integers()))
def test_copy(cls: Type, x: Set[int]) -> None:
    y = cls(x).copy()
    assert isinstance(y, cls)
    assert y == x


# set methods


@given(x=sets(integers()), xs=sets(frozensets(integers())))
def test_update(x: Set[int], xs: Set[FrozenSet[int]]) -> None:
    with warns(
        UserWarning,
        match="CSet.update is a non-functional method, did you mean CSet.union instead?",
    ):
        CSet(x).update(*xs)


@given(x=sets(integers()), xs=sets(frozensets(integers())))
def test_intersection_update(x: Set[int], xs: Set[FrozenSet[int]]) -> None:
    with warns(
        UserWarning,
        match="CSet.intersection_update is a non-functional method, did you mean CSet.intersection instead?",
    ):
        CSet(x).intersection_update(*xs)


@given(x=sets(integers()), xs=sets(frozensets(integers())))
def test_difference_update(x: Set[int], xs: Set[FrozenSet[int]]) -> None:
    with warns(
        UserWarning,
        match="CSet.difference_update is a non-functional method, did you mean CSet.difference instead?",
    ):
        CSet(x).difference_update(*xs)


@given(x=sets(integers()), y=sets(integers()))
def test_symmetric_difference_update(x: Set[int], y: Set[int]) -> None:
    with warns(
        UserWarning,
        match="CSet.symmetric_difference_update is a non-functional method, "
        "did you mean CSet.symmetric_difference instead?",
    ):
        CSet(x).symmetric_difference_update(y)


@given(x=sets(integers()), y=integers())
def test_add(x: Set[int], y: int) -> None:
    z = CSet(x).add(y)
    assert isinstance(z, CSet)
    assert z == set(chain(x, [y]))


@given(x=sets(integers()), y=integers())
def test_remove(x: Set[int], y: int) -> None:
    z = CSet(x)
    if y in x:
        w = z.remove(y)
        assert isinstance(w, CSet)
        assert w == {i for i in x if i != y}
    else:
        with raises(KeyError, match=str(y)):
            z.remove(y)


@given(x=sets(integers()), y=integers())
def test_discard(x: Set[int], y: int) -> None:
    z = CSet(x).discard(y)
    assert isinstance(z, CSet)
    assert z == {i for i in x if i != y}


@given(x=sets(integers()))
def test_pop(x: Set[int]) -> None:
    y = CSet(x)
    if y:
        new = y.pop()
        assert isinstance(new, CSet)
        assert len(new) == (len(x) - 1)
    else:
        with raises(KeyError, match="pop from an empty set"):
            y.pop()


# extra public


@given(x=sets(integers()))
def test_pipe(x: Set[int]) -> None:
    y = CSet(x).pipe(permutations, r=2)
    assert isinstance(y, CSet)
    assert y == set(permutations(x, r=2))
