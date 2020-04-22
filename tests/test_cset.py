from __future__ import annotations

from itertools import chain
from itertools import permutations
from re import search
from typing import FrozenSet

from hypothesis import given
from hypothesis.strategies import data
from hypothesis.strategies import DataObject
from hypothesis.strategies import integers
from pytest import mark
from pytest import raises
from pytest import warns

from functional_itertools import CFrozenSet
from functional_itertools import CSet
from tests.strategies import nested_siterables
from tests.strategies import sfrozensets
from tests.strategies import siterables

# repr and str


@mark.parametrize("case", [CSet, CFrozenSet])
@given(data=data())
def test_repr(case: Case, data: DataObject) -> None:
    x, _ = data.draw(siterables(cls, integers()))
    y = repr(cls(x))
    name = cls.__name__
    if x:
        assert search(fr"^{name}\(\{{[\d\s\-,]*\}}\)$", y)
    else:
        assert y == f"{name}()"


@mark.parametrize("case", [CSet, CFrozenSet])
@given(data=data())
def test_str(case: Case, data: DataObject) -> None:
    x, _ = data.draw(siterables(cls, integers()))
    y = str(cls(x))
    name = cls.__name__
    if x:
        assert search(fr"^{name}\(\{{[\d\s\-,]*\}}\)$", y)
    else:
        assert y == f"{name}()"


# set and frozenset methods


@mark.parametrize("case", [CSet, CFrozenSet])
@given(data=data())
def test_union(case: Case, data: DataObject) -> None:
    x, _ = data.draw(siterables(cls, integers()))
    xs, _ = data.draw(nested_siterables(cls, integers()))
    y = cls(x).union(*xs)
    assert isinstance(y, case.cls)
    assert y == x.union(*xs)


@mark.parametrize("case", [CSet, CFrozenSet])
@given(data=data())
def test_intersection(case: Case, data: DataObject) -> None:
    x, _ = data.draw(siterables(cls, integers()))
    xs, _ = data.draw(nested_siterables(cls, integers()))
    y = cls(x).intersection(*xs)
    assert isinstance(y, case.cls)
    assert y == x.intersection(*xs)


@mark.parametrize("case", [CSet, CFrozenSet])
@given(data=data())
def test_difference(case: Case, data: DataObject) -> None:
    x, _ = data.draw(siterables(cls, integers()))
    xs, _ = data.draw(nested_siterables(cls, integers()))
    y = cls(x).difference(*xs)
    assert isinstance(y, case.cls)
    assert y == (x.difference(*xs))


@mark.parametrize("case", [CSet, CFrozenSet])
@given(data=data())
def test_symmetric_difference(case: Case, data: DataObject) -> None:
    x, _ = data.draw(siterables(cls, integers()))
    y, _ = data.draw(siterables(cls, integers()))
    z = cls(x).symmetric_difference(y)
    assert isinstance(z, case.cls)
    assert z == x.symmetric_difference(y)


@mark.parametrize("case", [CSet, CFrozenSet])
@given(data=data())
def test_copy(case: Case, data: DataObject) -> None:
    x, _ = data.draw(siterables(cls, integers()))
    y = cls(x).copy()
    assert isinstance(y, case.cls)
    assert y == x


# set methods


@given(x=sfrozensets(integers()), xs=sfrozensets(sfrozensets(integers())))
def test_update(x: FrozenSet[int], xs: FrozenSet[FrozenSet[int]]) -> None:
    with warns(
        UserWarning, match="CSet.update is a non-functional name, did you mean CSet.union instead?",
    ):
        CSet(x).update(*xs)


@given(x=sfrozensets(integers()), xs=sfrozensets(sfrozensets(integers())))
def test_intersection_update(x: FrozenSet[int], xs: FrozenSet[FrozenSet[int]]) -> None:
    with warns(
        UserWarning,
        match="CSet.intersection_update is a non-functional name, did you mean CSet.intersection instead?",
    ):
        CSet(x).intersection_update(*xs)


@given(x=sfrozensets(integers()), xs=sfrozensets(sfrozensets(integers())))
def test_difference_update(x: FrozenSet[int], xs: FrozenSet[FrozenSet[int]]) -> None:
    with warns(
        UserWarning,
        match="CSet.difference_update is a non-functional name, did you mean CSet.difference instead?",
    ):
        CSet(x).difference_update(*xs)


@given(x=sfrozensets(integers()), y=sfrozensets(integers()))
def test_symmetric_difference_update(x: FrozenSet[int], y: FrozenSet[int]) -> None:
    with warns(
        UserWarning,
        match="CSet.symmetric_difference_update is a non-functional name, "
        "did you mean CSet.symmetric_difference instead?",
    ):
        CSet(x).symmetric_difference_update(y)


@given(x=sfrozensets(integers()), y=integers())
def test_add(x: FrozenSet[int], y: int) -> None:
    z = CSet(x).add(y)
    assert isinstance(z, CSet)
    assert z == set(chain(x, [y]))


@given(x=sfrozensets(integers()), y=integers())
def test_remove(x: FrozenSet[int], y: int) -> None:
    z = CSet(x)
    if y in x:
        w = z.remove(y)
        assert isinstance(w, CSet)
        assert w == {i for i in x if i != y}
    else:
        with raises(KeyError, match=str(y)):
            z.remove(y)


@given(x=sfrozensets(integers()), y=integers())
def test_discard(x: FrozenSet[int], y: int) -> None:
    z = CSet(x).discard(y)
    assert isinstance(z, CSet)
    assert z == {i for i in x if i != y}


@given(x=sfrozensets(integers()))
def test_pop(x: FrozenSet[int]) -> None:
    y = CSet(x)
    if y:
        new = y.pop()
        assert isinstance(new, CSet)
        assert len(new) == (len(x) - 1)
    else:
        with raises(KeyError, match="pop from an empty set"):
            y.pop()


# extra public


@given(x=sfrozensets(integers()))
def test_pipe(x: FrozenSet[int]) -> None:
    y = CSet(x).pipe(permutations, r=2)
    assert isinstance(y, CSet)
    assert y == set(permutations(x, r=2))
