from itertools import permutations
from re import search
from typing import FrozenSet
from typing import Set

from hypothesis import given
from hypothesis import infer
from hypothesis.strategies import integers
from hypothesis.strategies import sets

from functional_itertools.classes import CFrozenSet


@given(x=infer)
def test_repr(x: FrozenSet[int]) -> None:
    y = repr(CFrozenSet(x))
    if x:
        assert search(r"^CFrozenSet\(\{[\d\s\-,]*\}\)$", y)
    else:
        assert y == "CFrozenSet()"


@given(x=infer)
def test_str(x: FrozenSet[int]) -> None:
    y = str(CFrozenSet(x))
    if x:
        assert search(r"^CFrozenSet\(\{[\d\s\-,]*\}\)$", y)
    else:
        assert y == "CFrozenSet()"


# extra public


@given(x=sets(integers()))
def test_pipe(x: Set[int]) -> None:
    y = CFrozenSet(x).pipe(permutations, r=2)
    assert isinstance(y, CFrozenSet)
    assert y == set(permutations(x, r=2))
