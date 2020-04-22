from __future__ import annotations

from operator import itemgetter
from typing import FrozenSet
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union

from attr import attrs
from hypothesis.strategies import frozensets
from hypothesis.strategies import integers
from hypothesis.strategies import just
from hypothesis.strategies import lists
from hypothesis.strategies import none
from hypothesis.strategies import SearchStrategy
from hypothesis.strategies import sets
from hypothesis.strategies import tuples

from functional_itertools import CFrozenSet
from functional_itertools import CIterable
from functional_itertools import CList
from functional_itertools import CSet
from functional_itertools import CTuple


MAX_SIZE = 1000
T = TypeVar("T")


@attrs(auto_attribs=True)
class Case:
    cls: Type
    cast: Type


CASES = [
    Case(CIterable, list),
    Case(CList, list),
    Case(CTuple, tuple),
    Case(CSet, set),
    Case(CFrozenSet, frozenset),
]


def lists_or_sets(
    elements:SearchStrategy[T], *, min_size: int = 0, max_size: Optional[int] = None,
) -> SearchStrategy[Union[List[T], Set[T]]]:
    return lists(elements, min_size=min_size, max_size=max_size) | sets(
        elements, min_size=min_size, max_size=max_size,
    )


def slists(
    elements: SearchStrategy[T],
    *,
    min_size: int = 0,
    max_size: int = MAX_SIZE,
    unique: bool = False,
) -> SearchStrategy[List[T]]:
    return lists(elements, min_size=min_size, max_size=max_size, unique=unique)


def sfrozensets(
    elements: SearchStrategy[T], *, min_size: int = 0, max_size: int = MAX_SIZE,
) -> SearchStrategy[FrozenSet[T]]:
    return frozensets(elements, min_size=min_size, max_size=max_size)


def siterables(
    cls: Type,
    elements: SearchStrategy[T],
    *,
    min_size: int = 0,
    max_size: int = MAX_SIZE,
    unique: bool = False,
) -> SearchStrategy[Tuple[Union[List[T], FrozenSet[T]], Type]]:
    if cls in {CIterable, CList}:
        strategy = slists(elements, min_size=min_size, max_size=max_size, unique=unique)
        cast = list
    elif cls is CTuple:
        strategy = slists(elements, min_size=min_size, max_size=max_size, unique=unique).map(tuple)
        cast = tuple
    elif cls in {CSet, CFrozenSet}:
        strategy = frozensets(elements, min_size=min_size, max_size=max_size)
        cast = frozenset
    else:
        raise TypeError(cls)  # pragma: no cover
    return tuples(strategy, just(cast))


def nested_siterables(
    cls: Type, elements: SearchStrategy[T], *, min_size: int = 0, max_size: int = 10,
) -> SearchStrategy[Tuple[Union[List[List[T]], FrozenSet[FrozenSet[T]]], Type]]:
    return siterables(
        cls,
        siterables(cls, elements, min_size=min_size, max_size=max_size).map(itemgetter(0)),
        min_size=min_size,
        max_size=max_size,
    )


small_ints = integers(0, 10)
islice_ints = integers(0, MAX_SIZE)

range_args = (
    tuples(integers(0, MAX_SIZE), none(), none())
    | tuples(integers(-MAX_SIZE, 0), integers(0, MAX_SIZE), none())
    | tuples(integers(-MAX_SIZE, 0), integers(0, MAX_SIZE), integers(1, 10))
)
