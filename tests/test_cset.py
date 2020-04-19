from __future__ import annotations

from itertools import chain
from itertools import permutations
from re import search
from typing import FrozenSet
from typing import Set
from typing import Type

from hypothesis import given
from hypothesis import infer
from hypothesis.strategies import data
from hypothesis.strategies import DataObject
from hypothesis.strategies import integers
from hypothesis.strategies import sets
from pytest import mark
from pytest import raises
from pytest import warns

from functional_itertools import CFrozenSet
from functional_itertools import CSet
from tests.strategies import nested_siterables
from tests.strategies import sfrozensets
from tests.strategies import siterables

# repr and str


@mark.parametrize("cls", [CSet, CFrozenSet])
@given(data=data())
def test_repr(cls: Type, data: DataObject) -> None:
    x, _ = data.draw(siterables(cls, integers()))
    y = repr(cls(x))
    name = cls.__name__
    if x:
        assert search(fr"^{name}\(\{{[\d\s\-,]*\}}\)$", y)
    else:
        assert y == f"{name}()"


@mark.parametrize("cls", [CSet, CFrozenSet])
@given(data=data())
def test_str(cls: Type, data: DataObject) -> None:
    x, _ = data.draw(siterables(cls, integers()))
    y = str(cls(x))
    name = cls.__name__
    if x:
        assert search(fr"^{name}\(\{{[\d\s\-,]*\}}\)$", y)
    else:
        assert y == f"{name}()"


# set and frozenset methods


@mark.parametrize("cls", [CSet, CFrozenSet])
@given(data=data())
def test_union(cls: Type, data: DataObject) -> None:
    x, _ = data.draw(siterables(cls, integers()))
    xs, _ = data.draw(nested_siterables(cls, integers()))
    y = cls(x).union(*xs)
    assert isinstance(y, cls)
    assert y == x.union(*xs)


@mark.parametrize("cls", [CSet, CFrozenSet])
@given(data=data())
def test_intersection(cls: Type, data: DataObject) -> None:
    x, _ = data.draw(siterables(cls, integers()))
    xs, _ = data.draw(nested_siterables(cls, integers()))
    y = cls(x).intersection(*xs)
    assert isinstance(y, cls)
    assert y == x.intersection(*xs)


@mark.parametrize("cls", [CSet, CFrozenSet])
@given(data=data())
def test_difference(cls: Type, data: DataObject) -> None:
    x, _ = data.draw(siterables(cls, integers()))
    xs, _ = data.draw(nested_siterables(cls, integers()))
    y = cls(x).difference(*xs)
    assert isinstance(y, cls)
    assert y == (x.difference(*xs))


@mark.parametrize("cls", [CSet, CFrozenSet])
@given(data=data())
def test_symmetric_difference(cls: Type, data: DataObject) -> None:
    x, _ = data.draw(siterables(cls, integers()))
    y, _ = data.draw(siterables(cls, integers()))
    z = cls(x).symmetric_difference(y)
    assert isinstance(z, cls)
    assert z == x.symmetric_difference(y)


@mark.parametrize("cls", [CSet, CFrozenSet])
@given(data=data())
def test_copy(cls: Type, data: DataObject) -> None:
    x, _ = data.draw(siterables(cls, integers()))
    y = cls(x).copy()
    assert isinstance(y, cls)
    assert y == x


# set methods


@given(x=sfrozensets(integers()), y=sfrozensets(integers()))
def test_update(x: FrozenSet[int], y: FrozenSet[int]) -> None:
    with warns(
        UserWarning,
        match="CSet.update is a non-functional method, did you mean CSet.union instead?",
    ):
        CSet(x).update(*y)


@given(x=infer)
def test_intersection_update(x: Set[int]) -> None:
    with raises(
        RuntimeError, match="Use the 'intersection' method instead of 'intersection_update'",
    ):
        CSet(x).intersection_update()


@given(x=infer)
def test_difference_update(x: Set[int]) -> None:
    with raises(
        RuntimeError, match="Use the 'difference' method instead of 'difference_update'",
    ):
        CSet(x).difference_update()


@given(x=infer)
def test_symmetric_difference_update(x: Set[int]) -> None:
    with raises(
        RuntimeError,
        match="Use the 'symmetric_difference' method " "instead of 'symmetric_difference_update'",
    ):
        CSet(x).symmetric_difference_update()


@given(x=infer, y=infer)
def test_add(x: Set[int], y: int) -> None:
    cset = CSet(x).add(y)
    assert isinstance(cset, CSet)
    assert cset == set(chain(x, [y]))


@given(x=infer, y=infer)
def test_remove(x: Set[int], y: int) -> None:
    cset = CSet(x)
    if y in x:
        new = cset.remove(y)
        assert isinstance(new, CSet)
        assert new == {i for i in x if i != y}
    else:
        with raises(KeyError, match=str(y)):
            cset.remove(y)


@given(x=infer, y=infer)
def test_discard(x: Set[int], y: int) -> None:
    cset = CSet(x).discard(y)
    assert isinstance(cset, CSet)
    assert cset == {i for i in x if i != y}


@given(x=infer)
def test_pop(x: Set[int]) -> None:
    cset = CSet(x)
    if cset:
        new = cset.pop()
        assert isinstance(new, CSet)
        assert len(new) == (len(x) - 1)
    else:
        with raises(KeyError, match="pop from an empty set"):
            cset.pop()


# extra public


@given(x=sets(integers()))
def test_pipe(x: Set[int]) -> None:
    y = CSet(x).pipe(permutations, r=2)
    assert isinstance(y, CSet)
    assert y == set(permutations(x, r=2))
