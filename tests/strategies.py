from __future__ import annotations

from operator import itemgetter
from typing import FrozenSet
from typing import List
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union

from hypothesis.strategies import frozensets
from hypothesis.strategies import integers
from hypothesis.strategies import just
from hypothesis.strategies import lists
from hypothesis.strategies import SearchStrategy
from hypothesis.strategies import tuples

from functional_itertools import CFrozenSet
from functional_itertools import CIterable
from functional_itertools import CList
from functional_itertools import CSet


TEN = 10
T = TypeVar("T")


def slists(
    elements: SearchStrategy[T], *, min_size: int = 0, max_size: int = TEN, unique: bool = False,
) -> SearchStrategy[List[T]]:
    return lists(elements, min_size=min_size, max_size=max_size, unique=unique)


def sfrozensets(
    elements: SearchStrategy[T], *, min_size: int = 0, max_size: int = TEN,
) -> SearchStrategy[FrozenSet[T]]:
    return frozensets(elements, min_size=min_size, max_size=max_size)


def siterables(
    cls: Type,
    elements: SearchStrategy[T],
    *,
    min_size: int = 0,
    max_size: int = TEN,
    unique: bool = False,
) -> SearchStrategy[Tuple[Union[List[T], FrozenSet[T]], Type]]:
    if cls in {CIterable, CList}:
        strategy = slists(elements, min_size=min_size, max_size=max_size, unique=unique)
        cast = list
    elif cls in {CSet, CFrozenSet}:
        strategy = sfrozensets(elements, min_size=min_size, max_size=max_size)
        cast = frozenset
    else:
        raise TypeError(cls)  # pragma: no cover
    return tuples(strategy, just(cast))


def nested_siterables(
    cls: Type, elements: SearchStrategy[T], *, min_size: int = 0,
) -> SearchStrategy[Tuple[Union[List[List[T]], FrozenSet[FrozenSet[T]]], Type]]:
    return siterables(
        cls,
        siterables(cls, elements, min_size=min_size, max_size=5).map(itemgetter(0)),
        min_size=min_size,
        max_size=5,
    )


small_ints = integers(0, TEN)
