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


T = TypeVar("T")


def short_iterables(
    cls: Type,
    elements: SearchStrategy[T],
    *,
    min_size: int = 0,
    max_size: int = 10,
    unique: bool = False,
) -> SearchStrategy[Tuple[Union[List[T], FrozenSet[T]], Type]]:
    if cls in {CIterable, CList}:
        strategy = lists(elements, min_size=min_size, max_size=max_size, unique=unique)
        cast = list
    elif cls in {CSet, CFrozenSet}:
        strategy = frozensets(elements, min_size=min_size, max_size=max_size)
        cast = frozenset
    else:
        raise TypeError(cls)
    return tuples(strategy, just(cast))


def short_iterables_of_iterables(
    cls: Type,
    elements: SearchStrategy[T],
    *,
    min_size_outer: int = 0,
    max_size_outer: int = 3,
    min_size_inner: int = 0,
    max_size_inner: int = 3,
) -> SearchStrategy[Tuple[Union[List[List[T]], FrozenSet[FrozenSet[T]]], Type]]:
    return short_iterables(
        cls,
        short_iterables(cls, elements, min_size=min_size_inner, max_size=max_size_inner).map(
            itemgetter(0),
        ),
        min_size=min_size_outer,
        max_size=max_size_outer,
    )


small_ints = integers(0, 10)
