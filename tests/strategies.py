from __future__ import annotations

from typing import Type

from attr import attrs
from hypothesis.strategies import integers
from hypothesis.strategies import lists
from hypothesis.strategies import none
from hypothesis.strategies import tuples

from functional_itertools import CFrozenSet
from functional_itertools import CIterable
from functional_itertools import CList
from functional_itertools import CSet
from functional_itertools import CTuple


@attrs(auto_attribs=True)
class Case:
    cls: Type
    cast: Type
    ordered: bool


CASES = [
    Case(cls=CIterable, cast=list, ordered=True),
    Case(cls=CList, cast=list, ordered=True),
    Case(cls=CTuple, cast=tuple, ordered=True),
    Case(cls=CSet, cast=frozenset, ordered=False),
    Case(cls=CFrozenSet, cast=frozenset, ordered=False),
]


MAX_SIZE = 1000
islice_ints = integers(0, MAX_SIZE)
range_args = (
    tuples(integers(0, MAX_SIZE), none(), none())
    | tuples(integers(0, MAX_SIZE), integers(0, MAX_SIZE), none())
    | tuples(integers(0, MAX_SIZE), integers(0, MAX_SIZE), integers(1, 10))
)
combinations_x = lists(integers(), min_size=1, max_size=10)
combinations_r = integers(0, 3)
permutations_x = lists(integers(), max_size=5)
permutations_r = none() | integers(0, 3)
product_x = lists(integers(), min_size=1, max_size=3)
product_xs = lists(lists(integers(), min_size=1, max_size=3), min_size=1, max_size=3)
product_repeat = integers(0, 3)
