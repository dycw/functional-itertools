from functools import partial
from operator import add
from operator import mod
from operator import mul
from operator import sub
from typing import Callable
from typing import FrozenSet
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union

from hypothesis import given
from hypothesis import infer
from hypothesis.strategies import frozensets
from hypothesis.strategies import integers
from hypothesis.strategies import just
from hypothesis.strategies import lists
from hypothesis.strategies import sampled_from
from hypothesis.strategies import SearchStrategy
from hypothesis.strategies import sets
from hypothesis.strategies import tuples

from functional_itertools.classes import CFrozenSet
from functional_itertools.classes import CIterable
from functional_itertools.classes import CList
from functional_itertools.classes import CSet
from functional_itertools.utilities import sentinel


_T = TypeVar("_T")


def test_sentinel() -> None:
    assert repr(sentinel) == "<sentinel>"


small_ints = integers(0, 1000)


def small_lists(strategy: SearchStrategy[_T]) -> SearchStrategy[List[_T]]:
    return lists(strategy, max_size=10)


def lists_or_sets(
    cls: Type,
    elements: SearchStrategy[_T],
    *,
    min_size: int = 0,
    max_size: Optional[int] = None,
    unique: bool = False,
) -> SearchStrategy[Tuple[Union[List[_T], Set[_T], FrozenSet[_T]], Type]]:
    if cls in {CIterable, CList}:
        strategy, func = partial(lists, unique=unique), list
    elif cls is CSet:
        strategy, func = sets, set
    elif cls is CFrozenSet:
        strategy, func = frozensets, frozenset
    else:
        raise TypeError(cls)
    return tuples(strategy(elements, min_size=min_size, max_size=max_size), just(func))


def is_even(x: int) -> bool:
    return mod(x, 2) == 0


int_to_bool_funcs = (
    integers(min_value=1)
    .flatmap(lambda x: tuples(just(x), integers(0, x - 1)))
    .map(lambda mod_rem: (lambda x: mod(x, mod_rem[0]) == mod_rem[1]))
)


@given(func=int_to_bool_funcs, x=infer)
def test_int_to_bool_funcs(func: Callable[[int], bool], x: int) -> None:
    assert isinstance(func(x), bool)


int_and_int_to_bool_funcs = int_to_bool_funcs.map(lambda f: (lambda x, y: f(x) and f(y)))


@given(func=int_and_int_to_bool_funcs, x=infer, y=infer)
def test_int_and_int_to_bool_funcs(func: Callable[[int, int], bool], x: int, y: int) -> None:
    assert isinstance(func(x, y), bool)


int_and_int_to_int_funcs = sampled_from([add, sub, mul])


@given(func=int_and_int_to_int_funcs, x=infer, y=infer)
def test_int_and_int_to_int_funcs(func: Callable[[int, int], int], x: int, y: int) -> None:
    assert isinstance(func(x, y), int)


int_to_int_funcs = tuples(int_and_int_to_int_funcs, integers()).map(
    lambda op_n: (lambda x: op_n[0](op_n[1], x)),
)


@given(func=int_to_int_funcs, x=infer)
def test_int_to_int_funcs(func: Callable[[int], int], x: int) -> None:
    assert isinstance(func(x), int)


int_and_int_to_int_and_int_funcs = tuples(int_to_int_funcs, int_to_int_funcs).map(
    lambda f_g: (lambda x, y: (f_g[0](x), f_g[1](x))),
)


@given(func=int_and_int_to_int_and_int_funcs, x=infer, y=infer)
def test_int_and_int_to_int_and_int_funcs(
    func: Callable[[int, int], Tuple[int, int]], x: int, y: int,
) -> None:
    z = func(x, y)
    assert isinstance(z, tuple)
    a, b = z
    assert isinstance(a, int)
    assert isinstance(b, int)
