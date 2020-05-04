from __future__ import annotations

from typing import Type

from attr import attrs
from hypothesis.strategies import integers
from hypothesis.strategies import lists
from hypothesis.strategies import none

from functional_itertools import CFrozenSet
from functional_itertools import CIterable
from functional_itertools import CList
from functional_itertools import CSet
from functional_itertools import CTuple


@attrs(auto_attribs=True)
class Case:
    cls: Type
    cast: Type


CASES = [
    Case(cls=CIterable, cast=list),
    Case(cls=CList, cast=list),
    Case(cls=CTuple, cast=tuple),
    Case(cls=CSet, cast=frozenset),
    Case(cls=CFrozenSet, cast=frozenset),
]


MAX_SIZE = 1000
islice_ints = integers(0, MAX_SIZE)
combinations_x = lists(integers(), min_size=1, max_size=10)
combinations_r = integers(0, 3)
permutations_x = lists(integers(), max_size=5)
permutations_r = none() | integers(0, 3)
product_x = lists(integers(), min_size=1, max_size=3)
product_xs = lists(lists(integers(), min_size=1, max_size=3), min_size=1, max_size=3)
product_repeat = integers(0, 3)
