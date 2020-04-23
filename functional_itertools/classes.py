from __future__ import annotations

import builtins
import functools
import itertools
from itertools import chain
from itertools import islice
from multiprocessing import Pool
from operator import add
from pathlib import Path
from re import search
from sys import maxsize
from sys import modules
from types import FunctionType
from typing import Any
from typing import Callable
from typing import Dict
from typing import FrozenSet
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
from warnings import warn

import more_itertools
from more_itertools.recipes import first_true
from more_itertools.recipes import grouper
from more_itertools.recipes import iter_except
from more_itertools.recipes import nth_combination
from more_itertools.recipes import padnone
from more_itertools.recipes import partition
from more_itertools.recipes import powerset
from more_itertools.recipes import random_combination
from more_itertools.recipes import random_combination_with_replacement
from more_itertools.recipes import random_permutation
from more_itertools.recipes import random_product
from more_itertools.recipes import roundrobin
from more_itertools.recipes import tabulate
from more_itertools.recipes import unique_everseen
from more_itertools.recipes import unique_justseen

from functional_itertools.errors import EmptyIterableError
from functional_itertools.errors import MultipleElementsError
from functional_itertools.errors import StopArgumentMissing
from functional_itertools.errors import UnsupportVersionError
from functional_itertools.methods.base import CIterableOrCList
from functional_itertools.methods.base import Template
from functional_itertools.utilities import Sentinel
from functional_itertools.utilities import sentinel
from functional_itertools.utilities import VERSION
from functional_itertools.utilities import Version
from functional_itertools.utilities import warn_non_functional


T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")
_CIterable = "CIterable"
_CList = "CList"
_CTuple = "CTuple"
_CSet = "CSet"
_CFrozenSet = "CFrozenSet"


# built-ins


def _defines_method_factory(
    doc: str, *, citerable_or_clist: bool = False,
) -> Callable[[str], FunctionType]:
    def decorator(
        factory: Union[Callable[..., FunctionType], Callable[..., FunctionType]],
    ) -> Callable[[str], FunctionType]:
        def wrapped(name: str, **kwargs: Any) -> FunctionType:
            try:
                method = factory(**kwargs)
            except TypeError as error:
                (msg,) = error.args
                if search("missing 1 required positional argument: 'name'", msg):
                    method = factory(name, **kwargs)
                else:
                    raise
            for k, v in method.__annotations__.items():
                new_v = v.replace(Template.__name__, name)
                if citerable_or_clist:
                    new_v = new_v.replace(
                        CIterableOrCList.__name__, _CIterable if name == _CIterable else _CList,
                    )
                method.__annotations__[k] = new_v
            method.__doc__ = doc.format(name=name)
            return method

        return wrapped

    return decorator


def _get_citerable_or_clist(name: str) -> Type:
    required = _CIterable if name == _CIterable else _CList
    return getattr(modules[__name__], required.lstrip("_"))


@_defines_method_factory("Return `True` if all elements of the {name} are true, or if it is empty.")
def _build_all() -> Callable:
    def all(self: Template[T]) -> bool:  # noqa: A001
        return builtins.all(self)

    return all


@_defines_method_factory("Return `True` if at least 1 element of the {name} is true.")
def _build_any() -> Callable[..., bool]:
    def any(self: Template[T]) -> bool:  # noqa: A001
        return builtins.any(self)

    return any


@_defines_method_factory("Convert the {name} into a CDict.")
def _build_dict() -> Callable[..., CDict[Any, Any]]:
    def dict(self: Template[Tuple[T, U]]) -> CDict[T, U]:  # noqa: A001
        return CDict(self)

    return dict


@_defines_method_factory("Enumerate the elements of the {name}.", citerable_or_clist=True)
def _build_enumerate(name: str) -> Callable[..., Iterable[Tuple[int, Any]]]:
    def enumerate(  # noqa: A001
        self: Template[T], start: int = 0,
    ) -> CIterableOrCList[Tuple[int, T]]:
        return _get_citerable_or_clist(name)(builtins.enumerate(self, start=start))

    return enumerate


@_defines_method_factory("Filter the elements of the {name}.")
def _build_filter() -> Callable[..., Iterable]:
    def filter(self: Template[T], func: Optional[Callable[[T], bool]]) -> Template[T]:  # noqa: A001
        return type(self)(builtins.filter(func, self))

    return filter


@_defines_method_factory("Create a CIterable from the {name}.")
def _build_iter() -> Callable[..., CIterable]:
    def iter(self: Template[T]) -> CIterable[T]:  # noqa: A001
        return CIterable(self)

    return iter


@_defines_method_factory("Convert the {name} into a CFrozenSet.")
def _build_frozenset() -> Callable[..., CFrozenSet]:
    def frozenset(self: Template[T]) -> CFrozenSet[T]:  # noqa: A001
        return CFrozenSet(self)

    return frozenset


@_defines_method_factory("Return the length of the {name}.")
def _build_len() -> Callable[..., int]:
    def len(self: Template[T]) -> int:  # noqa: A001
        return builtins.len(self)

    return len


@_defines_method_factory("Create a CList from the {name}.")
def _build_list() -> Callable[..., CList]:
    def list(self: Template[T]) -> CList[T]:  # noqa: A001
        return CList(self)

    return list


@_defines_method_factory("Map over the elements of the {name}.")
def _build_map() -> Callable[..., Iterable]:
    def map(  # noqa: A001
        self: Template[T], func: Callable[..., U], *iterables: Iterable,
    ) -> Template[U]:
        return type(self)(builtins.map(func, self, *iterables))

    return map


@_defines_method_factory("Return the max/minimum over the {name}.")
def _build_maxmin(func: Callable) -> Callable:
    if VERSION is Version.py37:

        def min_max(
            self: Template[T],
            *,
            key: Union[Callable[[T], Any], Sentinel] = sentinel,
            default: U = sentinel,
        ) -> Union[T, U]:
            return func(
                self,
                **({} if key is sentinel else {"key": key}),
                **({} if default is sentinel else {"default": default}),
            )

    elif VERSION is Version.py38:

        def min_max(
            self: Template[T], *, key: Optional[Callable[[T], Any]] = None, default: U = sentinel,
        ) -> Union[T, U]:
            return func(self, key=key, **({} if default is sentinel else {"default": default}))

    else:
        raise UnsupportVersionError(VERSION)  # pragma: no cover

    min_max.__name__ = func.__name__
    return min_max


@_defines_method_factory("Return a range of integers as a {name}.")
def _build_range() -> Callable[..., Iterable[int]]:
    def range(  # noqa: A001
        cls: Type[Template], start: int, stop: Optional[int] = None, step: Optional[int] = None,
    ) -> Template[int]:
        if (stop is None) and (step is not None):
            raise StopArgumentMissing()
        else:
            return cls(
                builtins.range(
                    start, *(() if stop is None else (stop,)), *(() if step is None else (step,)),
                ),
            )

    return range


@_defines_method_factory("Convert the {name} into a CSet.")
def _build_set() -> Callable[..., CSet]:
    def set(self: Template[T]) -> CSet[T]:  # noqa: A001
        return CSet(self)

    return set


@_defines_method_factory("Convert the {name} into a sorted CList.")
def _build_sorted() -> Callable[..., CList]:
    def sorted(  # noqa: A001
        self: Template[T], *, key: Optional[Callable[[T], Any]] = None, reverse: bool = False,
    ) -> CList[T]:
        return CList(builtins.sorted(self, key=key, reverse=reverse))

    return sorted


@_defines_method_factory("Sum the elements of the {name}.")
def _build_sum() -> Callable[..., int]:
    def sum(self: Template[T], start: Union[U, Sentinel] = sentinel) -> Union[T, U]:  # noqa: A001
        return builtins.sum(self, *(() if start is sentinel else (start,)))

    return sum


@_defines_method_factory("Convert the {name} into a CFrozenSet.")
def _build_tuple() -> Callable[..., CTuple]:
    def tuple(self: Template[T]) -> CTuple[T]:  # noqa: A001
        return CTuple(self)

    return tuple


@_defines_method_factory(
    "Zip the elements of the {name} with other iterables.", citerable_or_clist=True,
)
def _build_zip(name: str) -> Callable[..., Iterable[CTuple]]:
    def zip(  # noqa: A001
        self: Template[T], *iterables: Iterable[U],
    ) -> CIterableOrCList[CTuple[Union[T, U]]]:
        return _get_citerable_or_clist(name)(map(CTuple, builtins.zip(self, *iterables)))

    return zip


# functools


@_defines_method_factory("Apply a binary function over the elements of the {name}")
def _build_reduce() -> Callable:
    def reduce(
        self: CIterable[T], func: Callable[[T, T], T], initial: Union[U, Sentinel] = sentinel,
    ) -> Any:
        try:
            result = functools.reduce(func, self, *(() if initial is sentinel else (initial,)))
        except TypeError as error:
            (msg,) = error.args
            if msg == "reduce() of empty sequence with no initial value":
                raise EmptyIterableError from None
            else:
                raise error
        else:
            if isinstance(result, list):
                return CList(result)
            elif isinstance(result, tuple):
                return CTuple(result)
            elif isinstance(result, set):
                return CSet(result)
            elif isinstance(result, frozenset):
                return CFrozenSet(result)
            elif isinstance(result, dict):
                return CDict(result)
            else:
                return result

    return reduce


# itertools


@_defines_method_factory(
    "accumulate([1,2,3,4,5]) --> 1 3 6 10 15", citerable_or_clist=True,
)
def _build_accumulate(name: str) -> Callable[..., Iterable]:
    if VERSION is Version.py37:

        def accumulate(self: Template[T], func: Callable[[T, T], T] = add) -> CIterableOrCList[T]:
            return _get_citerable_or_clist(name)(itertools.accumulate(self, func))

    elif VERSION is Version.py38:

        def accumulate(
            self: Template[T],
            func: Callable[[Union[T, U], Union[T, U]], Union[T, U]] = add,
            *,
            initial: Optional[U] = None,
        ) -> CIterableOrCList[Union[T, U]]:
            return _get_citerable_or_clist(name)(itertools.accumulate(self, func, initial=initial))

    else:
        raise UnsupportVersionError(VERSION)  # pragma: no cover

    return accumulate


@_defines_method_factory(
    "chain('ABC', 'DEF') --> A B C D E F", citerable_or_clist=True,
)
def _build_chain(name: str) -> Callable[..., Iterable]:
    def chain(self: Template[T], *iterables: Iterable[U]) -> CIterableOrCList[Union[T, U]]:
        return _get_citerable_or_clist(name)(itertools.chain(self, *iterables))

    return chain


@_defines_method_factory(
    "\n".join(
        [
            "combinations('ABCD', 2) --> AB AC AD BC BD CD",
            "combinations(range(4), 3) --> 012 013 023 123",
        ],
    ),
    citerable_or_clist=True,
)
def _build_combinations(name: str) -> Callable[..., Iterable[CTuple]]:
    def combinations(self: Template[T], r: int) -> CIterableOrCList[CTuple[T]]:
        return _get_citerable_or_clist(name)(map(CTuple, itertools.combinations(self, r)))

    return combinations


@_defines_method_factory(
    "combinations_with_replacement('ABC', 2) --> AA AB AC BB BC CC", citerable_or_clist=True,
)
def _build_combinations_with_replacement(name: str) -> Callable[..., Iterable]:
    def combinations_with_replacement(self: Template[T], r: int) -> CIterableOrCList[CTuple[T]]:
        return _get_citerable_or_clist(name)(
            map(CTuple, itertools.combinations_with_replacement(self, r)),
        )

    return combinations_with_replacement


@_defines_method_factory(
    "compress('ABCDEF', [1,0,1,0,1,1]) --> A C E F", citerable_or_clist=True,
)
def _build_compress(name: str) -> Callable[..., Iterable]:
    def compress(self: Template[T], selectors: Iterable) -> CIterableOrCList[T]:
        return _get_citerable_or_clist(name)(itertools.compress(self, selectors))

    return compress


@_defines_method_factory(
    "\n".join(["count(10) --> 10 11 12 13 14 ...", "count(2.5, 0.5) -> 2.5 3.0 3.5 ..."]),
)
def _build_count() -> Callable[..., CIterable[int]]:
    def count(cls: Type[Template[T]], start: int = 0, step: int = 1) -> CIterable[int]:
        return CIterable(itertools.count(start=start, step=step))

    return count


@_defines_method_factory("cycle('ABCD') --> A B C D A B C D A B C D ...")
def _build_cycle() -> Callable[..., CIterable]:
    def cycle(self: Template[T]) -> CIterable[T]:
        return CIterable(itertools.cycle(self))

    return cycle


@_defines_method_factory(
    "dropwhile(lambda x: x<5, [1,4,6,4,1]) --> 6 4 1", citerable_or_clist=True,
)
def _build_dropwhile(name: str) -> Callable[..., Iterable]:
    def dropwhile(self: Template[T], func: Callable[[T], bool]) -> CIterableOrCList[T]:
        return _get_citerable_or_clist(name)(itertools.dropwhile(func, self))

    return dropwhile


@_defines_method_factory(
    "filterfalse(lambda x: x%2, range(10)) --> 0 2 4 6 8", citerable_or_clist=True,
)
def _build_filterfalse(name: str) -> Callable[..., Iterable]:
    def filterfalse(self: Template[T], func: Callable[[T], bool]) -> CIterableOrCList[T]:
        return _get_citerable_or_clist(name)(itertools.filterfalse(func, self))

    return filterfalse


@_defines_method_factory(
    "\n".join(
        [
            "[k for k, g in groupby('AAAABBBCCDAABBB')] --> A B C D A B",
            "[list(g) for k, g in groupby('AAAABBBCCD')] --> AAAA BBB CC D",
        ],
    ),
    citerable_or_clist=True,
)
def _build_groupby(name: str) -> Callable[..., Any]:
    def groupby(
        self: Template[T], key: Optional[Callable[[T], U]] = None,
    ) -> CIterableOrCList[Tuple[U, CIterableOrCList[T]]]:
        cls = _get_citerable_or_clist(name)
        return cls((k, cls(v)) for k, v in itertools.groupby(self, key=key))

    return groupby


@_defines_method_factory(
    "\n".join(
        [
            "islice('ABCDEFG', 2) --> A B",
            "islice('ABCDEFG', 2, 4) --> C D",
            "islice('ABCDEFG', 2, None) --> C D E F G",
            "islice('ABCDEFG', 0, None, 2) --> A C E G",
        ],
    ),
)
def _build_islice() -> Callable[..., CIterable]:
    def islice(
        self: Template[T], start: int, stop: Optional[int] = None, step: Optional[int] = None,
    ) -> CIterable[T]:
        if (stop is None) and (step is not None):
            raise StopArgumentMissing()
        else:
            return CIterable(
                itertools.islice(
                    self,
                    start,
                    *(() if stop is None else (stop,)),
                    *(() if step is None else (step,)),
                ),
            )

    return islice


@_defines_method_factory(
    "\n".join(
        [
            "permutations('ABCD', 2) --> AB AC AD BA BC BD CA CB CD DA DB DC",
            "permutations(range(3)) --> 012 021 102 120 201 210",
        ],
    ),
    citerable_or_clist=True,
)
def _build_permutations(name: str) -> Callable[..., Iterable[CTuple]]:
    def permutations(self: Template[T], r: Optional[int] = None) -> CIterableOrCList[CTuple[T]]:
        return _get_citerable_or_clist(name)(map(CTuple, itertools.permutations(self, r=r)))

    return permutations


@_defines_method_factory("Cartesian product of input iterables.", citerable_or_clist=True)
def _build_product(name: str) -> Callable[..., Iterable[CTuple]]:
    def product(
        self: Template[T], *iterables: Iterable[U], repeat: int = 1,
    ) -> CIterableOrCList[CTuple[T]]:
        return _get_citerable_or_clist(name)(
            map(CTuple, itertools.product(self, *iterables, repeat=repeat)),
        )

    return product


@_defines_method_factory("Repeat an element", citerable_or_clist=True)
def _build_repeat(name: str) -> Callable[..., Iterable]:
    if name == _CIterable:

        def repeat(cls: Type[CIterable[T]], x: T, times: Optional[int] = None) -> CIterable[T]:
            return CIterable(itertools.repeat(x, **({} if times is None else {"times": times})))

    else:

        def repeat(cls: Type[Template[T]], x: T, times: int) -> CList[T]:
            return CList(itertools.repeat(x, times=times))

    return repeat


@_defines_method_factory("starmap(pow, [(2,5), (3,2), (10,3)]) --> 32 9 1000")
def _build_starmap() -> Callable[Iterable]:
    def starmap(self: Template[Tuple[T, ...]], func: Callable[[Tuple[T, ...]], U]) -> Template[U]:
        return type(self)(itertools.starmap(func, self))

    return starmap


@_defines_method_factory(
    "takewhile(lambda x: x<5, [1,4,6,4,1]) --> 1 4", citerable_or_clist=True,
)
def _build_takewhile(name: str) -> Callable[Iterable]:
    def takewhile(self: Template[T], func: Callable[[T], bool]) -> CIterableOrCList[T]:
        return _get_citerable_or_clist(name)(itertools.takewhile(func, self))

    return takewhile


@_defines_method_factory("Return n independent iterators from a single iterable.")
def _build_tee() -> Callable[..., CIterable[CIterable]]:
    def tee(self: Template[T], n: int = 2) -> CIterable[CIterable[T]]:
        return CIterable(map(CIterable, itertools.tee(self, n)))

    return tee


@_defines_method_factory(
    "zip_longest('ABCD', 'xy', fillvalue='-') --> Ax By C- D-", citerable_or_clist=True,
)
def _build_zip_longest(name: str) -> Callable[..., Iterable[Tuple]]:
    def zip_longest(
        self: Template[T], *iterables: Iterable[U], fillvalue: V = None,
    ) -> CIterableOrCList[CTuple[T]]:
        return _get_citerable_or_clist(name)(
            map(CTuple, itertools.zip_longest(self, *iterables, fillvalue=fillvalue)),
        )

    return zip_longest


# itertools-recipes


@_defines_method_factory("Returns True if all the elements are equal to each other")
def _build_all_equal() -> Callable[..., bool]:
    def all_equal(self: Template[T]) -> bool:
        return more_itertools.all_equal(self)

    return all_equal


@_defines_method_factory("Advance the iterator n-steps ahead. If n is None, consume entirely.")
def _build_consume() -> Callable[..., CIterable]:
    def consume(self: CIterable[T], n: Optional[int] = None) -> CIterable[T]:
        iterator = iter(self)
        more_itertools.consume(iterator, n=n)
        return CIterable(iterator)

    return consume


@_defines_method_factory("Returns True if all the elements are equal to each other")
def _build_dotproduct() -> Callable[..., Any]:
    def dotproduct(self: Template[T], x: Iterable[T]) -> T:
        return more_itertools.dotproduct(self, x)

    return dotproduct


@_defines_method_factory(
    "Flatten one level of nesting", citerable_or_clist=True,
)
def _build_flatten(name: str) -> Callable[..., Iterable]:
    def flatten(self: Template[Iterable[T]]) -> CIterableOrCList[T]:
        return _get_citerable_or_clist(name)(more_itertools.flatten(self))

    return flatten


@_defines_method_factory(
    "Returns the sequence elements n times", citerable_or_clist=True,
)
def _build_ncycles(name: str) -> Callable[..., Iterable]:
    def ncycles(self: Template[T], n: int) -> CIterableOrCList[T]:
        return _get_citerable_or_clist(name)(more_itertools.ncycles(self, n))

    return ncycles


@_defines_method_factory("Returns the nth item or a default value")
def _build_nth() -> Callable[..., Any]:
    def nth(self: Template[T], n: int, default: Optional[int] = None) -> T:
        return more_itertools.nth(self, n, default=default)

    return nth


@_defines_method_factory(
    "s -> (s0,s1), (s1,s2), (s2, s3), ...", citerable_or_clist=True,
)
def _build_pairwise(name: str) -> Callable[..., Iterable[CTuple]]:
    def pairwise(self: Template[T]) -> CIterableOrCList[CTuple[T]]:
        return _get_citerable_or_clist(name)(map(CTuple, more_itertools.pairwise(self)))

    return pairwise


@_defines_method_factory(
    "prepend(1, [2, 3, 4]) -> 1 2 3 4", citerable_or_clist=True,
)
def _build_prepend(name: str) -> Callable[..., Iterable]:
    def prepend(self: Template[T], value: U) -> CIterableOrCList[Union[T, U]]:
        return _get_citerable_or_clist(name)(more_itertools.prepend(value, self))

    return prepend


@_defines_method_factory("Count how many times the predicate is true")
def _build_quantify() -> Callable[..., int]:
    def quantify(self: Template[T], pred: Callable[[T], bool] = bool) -> int:
        return more_itertools.quantify(self, pred=pred)

    return quantify


@_defines_method_factory("Repeat calls to func with specified arguments", citerable_or_clist=True)
def _build_repeatfunc(name: str) -> Callable[..., Iterable]:
    if name == _CIterable:

        def repeatfunc(
            cls: Type[CIterable], func: Callable[..., T], times: Optional[int] = None, *args: Any,
        ) -> CIterable[T]:
            return CIterable(more_itertools.repeatfunc(func, times, *args))

    else:

        def repeatfunc(
            cls: Type[Template], func: Callable[..., T], times: int, *args: Any,
        ) -> CList[T]:
            return CList(more_itertools.repeatfunc(func, times, *args))

    return repeatfunc


@_defines_method_factory(
    "Return an iterator over the last n items", citerable_or_clist=True,
)
def _build_tail(name: str) -> Callable[..., Iterable]:
    def tail(self: Template[T], n: int) -> CIterableOrCList[T]:
        return _get_citerable_or_clist(name)(more_itertools.tail(n, self))

    return tail


@_defines_method_factory(
    "Return first n items of the iterable", citerable_or_clist=True,
)
def _build_take(name: str) -> Callable[..., Iterable]:
    def take(self: Template[T], n: int) -> CIterableOrCList[T]:
        return _get_citerable_or_clist(name)(more_itertools.take(n, self))

    return take


# more-itertools


@_defines_method_factory(
    "chunked([1, 2, 3, 4, 5, 6, 7, 8], 3) --> [[1, 2, 3], [4, 5, 6], [7, 8]]",
    citerable_or_clist=True,
)
def _build_chunked(name: str) -> Callable[..., Iterable[Iterable]]:
    def chunked(self: Template[T], n: int) -> CIterableOrCList[CIterableOrCList[T]]:
        cls = _get_citerable_or_clist(name)
        return cls(map(cls, more_itertools.chunked(self, n)))

    return chunked


@_defines_method_factory(
    "distribute(3, [1, 2, 3, 4, 5, 6, 7]) --> [[1, 4, 7], [2, 5], [3, 6]]", citerable_or_clist=True,
)
def _build_distribute(name: str) -> Callable[..., Iterable[Iterable]]:
    def distribute(self: Template[T], n: int) -> CIterableOrCList[CIterableOrCList[T]]:
        cls = _get_citerable_or_clist(name)
        return cls(map(cls, more_itertools.distribute(n, self)))

    return distribute


@_defines_method_factory(
    "divide(3, [1, 2, 3, 4, 5, 6, 7]) --> [[1, 2, 3], [4, 5], [6, 7]]", citerable_or_clist=True,
)
def _build_divide(name: str) -> Callable[Iterable[Iterable]]:
    def divide(self: Template[T], n: int) -> CIterableOrCList[CIterableOrCList[T]]:
        cls = _get_citerable_or_clist(name)
        return cls(map(cls, more_itertools.divide(n, list(self))))

    return divide


# multiprocessing


@_defines_method_factory("Map over the elements of the {name} in parallel.")
def _build_pmap() -> Callable[..., Iterable]:
    def pmap(
        self: Template[T], func: Callable[[T], U], *, processes: Optional[int] = None,
    ) -> Template[U]:
        try:
            with Pool(processes=processes) as pool:
                return type(self)(pool.map(func, self))
        except AssertionError as error:
            (msg,) = error.args
            if msg == "daemonic processes are not allowed to have children":
                return self.map(func)
            else:
                raise NotImplementedError(msg)

    return pmap


@_defines_method_factory("Star_map over the elements of the {name} in parallel.")
def _build_pstarmap() -> Callable[..., Iterable]:
    def pstarmap(
        self: Template[Tuple[T, ...]],
        func: Callable[[Tuple[T, ...]], U],
        *,
        processes: Optional[int] = None,
    ) -> Template[U]:
        try:
            with Pool(processes=processes) as pool:
                return type(self)(pool.starmap(func, self))
        except AssertionError as error:
            (msg,) = error.args
            if msg == "daemonic processes are not allowed to have children":
                return self.starmap(func)
            else:
                raise NotImplementedError(msg)

    return pstarmap


# pathlib


@_defines_method_factory("Return a collection of paths as a {name}.")
def _build_iterdir() -> Callable[..., Iterable[Path]]:
    def iterdir(cls: Type[Template], path: Union[Path, str]) -> Template[Path]:
        return cls(Path(path).iterdir())

    return iterdir


# classes


class CIterable(Iterable[T]):
    __slots__ = ("_iterable",)

    def __init__(self: CIterable[T], iterable: Iterable[T]) -> None:
        try:
            iter(iterable)
        except TypeError as error:
            (msg,) = error.args
            raise TypeError(f"{type(self).__name__} expected an iterable, but {msg}")
        else:
            self._iterable = iterable

    def __getitem__(self: CIterable[T], item: Union[int, slice]) -> Union[T, CIterable[T]]:
        if isinstance(item, int):
            if item < 0:
                raise IndexError(f"Expected a non-negative index; got {item}")
            elif item > maxsize:
                raise IndexError(f"Expected an index at most {maxsize}; got {item}")
            else:
                slice_ = islice(self._iterable, item, item + 1)
                try:
                    return next(slice_)
                except StopIteration:
                    raise IndexError(f"{type(self).__name__} index out of range")
        elif isinstance(item, slice):
            return self.islice(item.start, item.stop, item.step)
        else:
            raise TypeError(f"Expected an int or slice; got a(n) {type(item).__name__}")

    def __iter__(self: CIterable[T]) -> Iterator[T]:
        yield from self._iterable

    def __repr__(self: CIterable) -> str:
        return f"{type(self).__name__}({self._iterable!r})"

    def __str__(self: CIterable) -> str:
        return f"{type(self).__name__}({self._iterable})"

    # built-ins

    all = _build_all(_CIterable)  # noqa: A003
    any = _build_any(_CIterable)  # noqa: A003
    dict = _build_dict(_CIterable)  # noqa: A003
    enumerate = _build_enumerate(_CIterable)  # noqa: A003
    filter = _build_filter(_CIterable)  # noqa: A003
    frozenset = _build_frozenset(_CIterable)  # noqa: A003
    iter = _build_iter(_CIterable)  # noqa: A003
    list = _build_list(_CIterable)  # noqa: A003
    map = _build_map(_CIterable)  # noqa: A003
    max = _build_maxmin(_CIterable, func=max)  # noqa: A003
    min = _build_maxmin(_CIterable, func=min)  # noqa: A003
    range = classmethod(_build_range(_CIterable))  # noqa: A003
    set = _build_set(_CIterable)  # noqa: A003
    sorted = _build_sorted(_CIterable)  # noqa: A003
    sum = _build_sum(_CIterable)  # noqa: A003
    tuple = _build_tuple(_CIterable)  # noqa: A003
    zip = _build_zip(_CIterable)  # noqa: A003

    # functools

    reduce = _build_reduce(_CIterable)

    # itertools

    combinations = _build_combinations(_CIterable)
    combinations_with_replacement = _build_combinations_with_replacement(_CIterable)
    count = classmethod(_build_count(_CIterable))
    cycle = _build_cycle(_CIterable)
    repeat = classmethod(_build_repeat(_CIterable))
    accumulate = _build_accumulate(_CIterable)
    chain = _build_chain(_CIterable)
    compress = _build_compress(_CIterable)
    dropwhile = _build_dropwhile(_CIterable)
    filterfalse = _build_filterfalse(_CIterable)
    groupby = _build_groupby(_CIterable)
    islice = _build_islice(_CIterable)
    permutations = _build_permutations(_CIterable)
    product = _build_product(_CIterable)
    starmap = _build_starmap(_CIterable)
    takewhile = _build_takewhile(_CIterable)
    tee = _build_tee(_CIterable)
    zip_longest = _build_zip_longest(_CIterable)

    # itertools-recipes

    all_equal = _build_all_equal(_CIterable)
    consume = _build_consume(_CIterable)
    dotproduct = _build_dotproduct(_CIterable)
    flatten = _build_flatten(_CIterable)
    ncycles = _build_ncycles(_CIterable)
    nth = _build_nth(_CIterable)
    pairwise = _build_pairwise(_CIterable)
    prepend = _build_prepend(_CIterable)
    quantify = _build_quantify(_CIterable)
    repeatfunc = classmethod(_build_repeatfunc(_CIterable))
    tail = _build_tail(_CIterable)
    take = _build_take(_CIterable)

    @classmethod
    def tabulate(cls: Type[CIterable], func: Callable[[int], T], start: int = 0) -> CIterable[T]:
        return cls(tabulate(func, start=start))

    def padnone(self: CIterable[T]) -> CIterable[Optional[T]]:
        return CIterable(padnone(self._iterable))

    def grouper(
        self: CIterable[T], n: int, fillvalue: U = None,
    ) -> CIterable[Tuple[Union[T, U], ...]]:
        return CIterable(grouper(self._iterable, n, fillvalue=fillvalue))

    def partition(
        self: CIterable[T], func: Callable[[T], bool],
    ) -> Tuple[CIterable[T], CIterable[T]]:
        return CIterable(partition(func, self._iterable)).map(CIterable).tuple()

    def powerset(self: CIterable[T]) -> CIterable[Tuple[T, ...]]:
        return CIterable(powerset(self._iterable))

    def roundrobin(self: CIterable[T], *iterables: Iterable[U]) -> CIterable[Tuple[T, U]]:
        return CIterable(roundrobin(self._iterable, *iterables))

    def unique_everseen(
        self: CIterable[T], key: Optional[Callable[[T], Any]] = None,
    ) -> CIterable[T]:
        return CIterable(unique_everseen(self._iterable, key=key))

    def unique_justseen(
        self: CIterable[T], key: Optional[Callable[[T], Any]] = None,
    ) -> CIterable[T]:
        return CIterable(unique_justseen(self._iterable, key=key))

    @classmethod
    def iter_except(
        cls: Type[CIterable],
        func: Callable[..., T],
        exception: Type[Exception],
        first: Optional[Callable[..., U]] = None,
    ) -> CIterable[Union[T, U]]:
        return cls(iter_except(func, exception, first=first))

    def first_true(
        self: CIterable[T], default: U = False, pred: Optional[Callable[[T], Any]] = None,
    ) -> Union[T, U]:
        return first_true(self._iterable, default=default, pred=pred)

    def random_product(
        self: CIterable[T], *iterables: Iterable[U], repeat: int = 1,
    ) -> Tuple[Union[T, U], ...]:
        return random_product(self._iterable, *iterables, repeat=repeat)

    def random_permutation(self: CIterable[T], r: Optional[int] = None) -> Tuple[Union[T, U], ...]:
        return random_permutation(self._iterable, r=r)

    def random_combination(self: CIterable[T], r: int) -> Tuple[T, ...]:
        return random_combination(self._iterable, r)

    def random_combination_with_replacement(self: CIterable[T], r: int) -> Tuple[T, ...]:
        return random_combination_with_replacement(self._iterable, r)

    def nth_combination(self: CIterable[T], r: int, index: int) -> Tuple[T, ...]:
        return nth_combination(self._iterable, r, index)

    # more-itertools

    chunked = _build_chunked(_CIterable)
    distribute = _build_distribute(_CIterable)
    divide = _build_divide(_CIterable)

    # multiprocessing

    pmap = _build_pmap(_CIterable)
    pstarmap = _build_pstarmap(_CIterable)

    # pathlib

    iterdir = classmethod(_build_iterdir(_CIterable))

    # extra public

    def append(self: CIterable[T], value: U) -> CIterable[Union[T, U]]:  # dead: disable
        return self.chain([value])

    def first(self: CIterable[T]) -> T:
        try:
            return next(iter(self._iterable))
        except StopIteration:
            raise EmptyIterableError from None

    def last(self: CIterable[T]) -> T:  # dead: disable
        return self.reduce(lambda x, y: y)

    def one(self: CIterable[T]) -> T:
        head: CList[T] = self.islice(2).list()
        if head:
            try:
                (x,) = head
            except ValueError:
                x, y = head
                raise MultipleElementsError(f"{x}, {y}")
            else:
                return x
        else:
            raise EmptyIterableError

    def pipe(
        self: CIterable[T],
        func: Callable[..., Iterable[U]],
        *args: Any,
        index: int = 0,
        **kwargs: Any,
    ) -> CIterable[U]:
        new_args = chain(islice(args, index), [self._iterable], islice(args, index, None))
        return CIterable(func(*new_args, **kwargs))

    def unzip(self: CIterable[Tuple[T, ...]]) -> Tuple[CIterable[T], ...]:
        return CIterable(zip(*self)).map(CIterable).tuple()


class CList(List[T]):
    """A list with chainable methods."""

    def __getitem__(self: CList[T], item: Union[int, slice]) -> Union[T, CList[T]]:
        out = super().__getitem__(item)
        if isinstance(out, list):
            return CList(out)
        else:
            return out

    # built-ins

    all = _build_all(_CList)  # noqa: A003
    any = _build_any(_CList)  # noqa: A003
    dict = _build_dict(_CList)  # noqa: A003
    enumerate = _build_enumerate(_CList)  # noqa: A003
    filter = _build_filter(_CList)  # noqa: A003
    frozenset = _build_frozenset(_CList)  # noqa: A003
    iter = _build_iter(_CList)  # noqa: A003
    len = _build_len(_CList)  # noqa: A003
    list = _build_list(_CList)  # noqa: A003
    map = _build_map(_CList)  # noqa: A003
    max = _build_maxmin(_CList, func=max)  # noqa: A003
    min = _build_maxmin(_CList, func=min)  # noqa: A003
    range = classmethod(_build_range(_CList))  # noqa: A003
    set = _build_set(_CList)  # noqa: A003
    sorted = _build_sorted(_CList)  # noqa: A003
    sum = _build_sum(_CList)  # noqa: A003
    tuple = _build_tuple(_CList)  # noqa: A003
    zip = _build_zip(_CList)  # noqa: A003

    def copy(self: CList[T]) -> CList[T]:
        return CList(super().copy())

    def reversed(self: CList[T]) -> CList[T]:  # noqa: A003
        return CList(reversed(self))

    def sort(  # dead: disable
        self: CList[T], *, key: Optional[Callable[[T], Any]] = None, reverse: bool = False,
    ) -> CList[T]:
        warn("Use the 'sorted' name instead of 'sort'")
        return self.sorted(key=key, reverse=reverse)

    # functools

    reduce = _build_reduce(_CList)

    # itertools

    combinations = _build_combinations(_CList)
    combinations_with_replacement = _build_combinations_with_replacement(_CList)
    repeat = classmethod(_build_repeat(_CList))
    accumulate = _build_accumulate(_CList)
    chain = _build_chain(_CList)
    compress = _build_compress(_CList)
    dropwhile = _build_dropwhile(_CList)
    filterfalse = _build_filterfalse(_CList)
    groupby = _build_groupby(_CList)
    islice = _build_islice(_CList)
    permutations = _build_permutations(_CList)
    product = _build_product(_CList)
    starmap = _build_starmap(_CList)
    takewhile = _build_takewhile(_CList)
    tee = _build_tee(_CList)
    zip_longest = _build_zip_longest(_CList)

    def permutations(self: CList[T], r: Optional[int] = None) -> CList[Tuple[T, ...]]:
        return self.iter().permutations(r=r).list()

    # itertools-recipes

    all_equal = _build_all_equal(_CList)
    dotproduct = _build_dotproduct(_CList)
    flatten = _build_flatten(_CList)
    ncycles = _build_ncycles(_CList)
    nth = _build_nth(_CList)
    pairwise = _build_pairwise(_CList)
    prepend = _build_prepend(_CList)
    quantify = _build_quantify(_CList)
    repeatfunc = classmethod(_build_repeatfunc(_CList))
    tail = _build_tail(_CList)
    take = _build_take(_CList)

    def grouper(
        self: CList[T], n: int, fillvalue: Optional[T] = None,
    ) -> CList[Tuple[Union[T, U], ...]]:
        return self.iter().grouper(n, fillvalue=fillvalue).list()

    def partition(self: CList[T], func: Callable[[T], bool]) -> Tuple[CList[T], CList[T]]:
        return self.iter().partition(func).map(CList).tuple()

    def powerset(self: CList[T]) -> CList[Tuple[T, ...]]:
        return self.iter().powerset().list()

    def roundrobin(self: CList[T], *iterables: Iterable[U]) -> CList[Tuple[T, U]]:
        return self.iter().roundrobin(*iterables).list()

    def unique_everseen(self: CList[T], key: Optional[Callable[[T], Any]] = None) -> CList[T]:
        return self.iter().unique_everseen(key=key).list()

    def unique_justseen(self: CList[T], key: Optional[Callable[[T], Any]] = None) -> CList[T]:
        return self.iter().unique_justseen(key=key).list()

    @classmethod
    def iter_except(
        cls: Type[CList],
        func: Callable[..., T],
        exception: Type[Exception],
        first: Optional[Callable[..., U]] = None,
    ) -> CList[Union[T, U]]:
        return CIterable.iter_except(func, exception, first=first).list()

    def first_true(
        self: CList[T], default: U = False, pred: Optional[Callable[[T], Any]] = None,
    ) -> Union[T, U]:
        return self.iter().first_true(default=default, pred=pred).list()

    def random_product(
        self: CList[T], *iterables: Iterable[U], repeat: int = 1,
    ) -> Tuple[Union[T, U], ...]:
        return self.iter().random_product(*iterables, repeat=repeat)

    def random_permutation(self: CList[T], r: Optional[int] = None) -> Tuple[T, ...]:
        return self.iter().random_permutation(r=r)

    def random_combination(self: CList[T], r: int) -> Tuple[T, ...]:
        return self.iter().random_combination(r)

    def random_combination_with_replacement(self: CList[T], r: int) -> Tuple[T, ...]:
        return self.iter().random_combination_with_replacement(r)

    def nth_combination(self: CList[T], r: int, index: int) -> Tuple[T, ...]:
        return self.iter().nth_combination(r, index)

    # more-itertools

    chunked = _build_chunked(_CList)
    distribute = _build_distribute(_CList)
    divide = _build_divide(_CList)

    # multiprocessing

    pmap = _build_pmap(_CList)
    pstarmap = _build_pstarmap(_CList)

    # pathlib

    iterdir = classmethod(_build_iterdir(_CList))

    # extra public

    def one(self: CList[T]) -> T:
        return self.iter().one()

    def pipe(
        self: CList[T], func: Callable[..., Iterable[U]], *args: Any, index: int = 0, **kwargs: Any,
    ) -> CList[U]:
        return self.iter().pipe(func, *args, index=index, **kwargs).list()

    def unzip(self: CList[Tuple[T, ...]]) -> Tuple[CList[T], ...]:
        return CList(self.iter().unzip()).map(CList)


class CTuple(Tuple[T]):
    """A tuple with chainable methods."""

    # built-ins

    all = _build_all(_CTuple)  # noqa: A003
    any = _build_any(_CTuple)  # noqa: A003
    dict = _build_dict(_CTuple)  # noqa: A003
    enumerate = _build_enumerate(_CTuple)  # noqa: A003
    filter = _build_filter(_CTuple)  # noqa: A003
    frozenset = _build_frozenset(_CTuple)  # noqa: A003
    iter = _build_iter(_CTuple)  # noqa: A003
    len = _build_len(_CTuple)  # noqa: A003
    list = _build_list(_CTuple)  # noqa: A003
    map = _build_map(_CTuple)  # noqa: A003
    max = _build_maxmin(_CTuple, func=max)  # noqa: A003
    min = _build_maxmin(_CTuple, func=min)  # noqa: A003
    range = classmethod(_build_range(_CTuple))  # noqa: A003
    set = _build_set(_CTuple)  # noqa: A003
    sorted = _build_sorted(_CTuple)  # noqa: A003
    sum = _build_sum(_CTuple)  # noqa: A003
    tuple = _build_tuple(_CTuple)  # noqa: A003
    zip = _build_zip(_CTuple)  # noqa: A003

    # functools

    reduce = _build_reduce(_CTuple)

    # itertools

    combinations = _build_combinations(_CTuple)
    combinations_with_replacement = _build_combinations_with_replacement(_CTuple)
    repeat = classmethod(_build_repeat(_CTuple))
    accumulate = _build_accumulate(_CTuple)
    chain = _build_chain(_CTuple)
    compress = _build_compress(_CTuple)
    dropwhile = _build_dropwhile(_CTuple)
    filterfalse = _build_filterfalse(_CTuple)
    groupby = _build_groupby(_CTuple)
    islice = _build_islice(_CTuple)
    permutations = _build_permutations(_CTuple)
    product = _build_product(_CTuple)
    starmap = _build_starmap(_CTuple)
    takewhile = _build_takewhile(_CTuple)
    tee = _build_tee(_CTuple)
    zip_longest = _build_zip_longest(_CTuple)

    # itertools-recipes

    all_equal = _build_all_equal(_CTuple)
    dotproduct = _build_dotproduct(_CTuple)
    flatten = _build_flatten(_CTuple)
    ncycles = _build_ncycles(_CTuple)
    nth = _build_nth(_CTuple)
    pairwise = _build_pairwise(_CTuple)
    prepend = _build_prepend(_CTuple)
    quantify = _build_quantify(_CTuple)
    repeatfunc = classmethod(_build_repeatfunc(_CTuple))
    tail = _build_tail(_CTuple)
    take = _build_take(_CTuple)

    # more-itertools

    chunked = _build_chunked(_CTuple)
    distribute = _build_distribute(_CTuple)
    divide = _build_divide(_CTuple)

    # multiprocessing

    pmap = _build_pmap(_CTuple)
    pstarmap = _build_pstarmap(_CTuple)

    # pathlib

    iterdir = classmethod(_build_iterdir(_CTuple))


class CSet(Set[T]):
    """A set with chainable methods."""

    # built-ins

    all = _build_all(_CSet)  # noqa: A003
    any = _build_any(_CSet)  # noqa: A003
    dict = _build_dict(_CSet)  # noqa: A003
    enumerate = _build_enumerate(_CSet)  # noqa: A003
    filter = _build_filter(_CSet)  # noqa: A003
    frozenset = _build_frozenset(_CSet)  # noqa: A003
    iter = _build_iter(_CSet)  # noqa: A003
    len = _build_len(_CSet)  # noqa: A003
    list = _build_list(_CSet)  # noqa: A003
    map = _build_map(_CSet)  # noqa: A003
    max = _build_maxmin(_CSet, func=max)  # noqa: A003
    min = _build_maxmin(_CSet, func=min)  # noqa: A003
    range = classmethod(_build_range(_CSet))  # noqa: A003
    set = _build_set(_CSet)  # noqa: A003
    sorted = _build_sorted(_CSet)  # noqa: A003
    sum = _build_sum(_CSet)  # noqa: A003
    tuple = _build_tuple(_CSet)  # noqa: A003
    zip = _build_zip(_CSet)  # noqa: A003

    # set & frozenset methods

    def union(self: CSet[T], *others: Iterable[U]) -> CSet[Union[T, U]]:
        return CSet(super().union(*others))

    def intersection(self: CSet[T], *others: Iterable[U]) -> CSet[Union[T, U]]:
        return CSet(super().intersection(*others))

    def difference(self: CSet[T], *others: Iterable[U]) -> CSet[Union[T, U]]:
        return CSet(super().difference(*others))

    def symmetric_difference(self: CSet[T], other: Iterable[U]) -> CSet[Union[T, U]]:
        return CSet(super().symmetric_difference(other))

    def copy(self: CSet[T]) -> CSet[T]:
        return CSet(super().copy())

    # set methods

    def update(self: CSet[T], *other: Iterable[U]) -> None:
        warn_non_functional(CSet, "update", "union")
        super().update(*other)

    def intersection_update(self: CSet[T], *other: Iterable[U]) -> None:
        warn_non_functional(CSet, "intersection_update", "intersection")
        super().intersection_update(*other)

    def difference_update(self: CSet[T], *other: Iterable[U]) -> None:
        warn_non_functional(CSet, "difference_update", "difference")
        super().difference_update(*other)

    def symmetric_difference_update(self: CSet[T], other: Iterable[U]) -> None:
        warn_non_functional(CSet, "symmetric_difference_update", "symmetric_difference")
        super().symmetric_difference_update(other)
        return self

    def add(self: CSet[T], element: T) -> CSet[T]:
        super().add(element)
        return self

    def remove(self: CSet[T], element: T) -> CSet[T]:
        super().remove(element)
        return self

    def discard(self: CSet[T], element: T) -> CSet[T]:
        super().discard(element)
        return self

    def pop(self: CSet[T]) -> CSet[T]:
        super().pop()
        return self

    def clear(self: CSet[T]) -> CSet[T]:
        super().clear()
        return self

    # functools

    reduce = _build_reduce(_CSet)

    # itertools

    accumulate = _build_accumulate(_CSet)
    chain = _build_chain(_CSet)
    combinations = _build_combinations(_CSet)
    combinations_with_replacement = _build_combinations_with_replacement(_CSet)
    compress = _build_compress(_CSet)
    dropwhile = _build_dropwhile(_CSet)
    filterfalse = _build_filterfalse(_CSet)
    groupby = _build_groupby(_CSet)
    islice = _build_islice(_CSet)
    permutations = _build_permutations(_CSet)
    product = _build_product(_CSet)
    repeat = classmethod(_build_repeat(_CSet))
    starmap = _build_starmap(_CSet)
    takewhile = _build_takewhile(_CSet)
    tee = _build_tee(_CSet)
    zip_longest = _build_zip_longest(_CSet)

    # itertools-recipes

    all_equal = _build_all_equal(_CSet)
    dotproduct = _build_dotproduct(_CSet)
    flatten = _build_flatten(_CSet)
    ncycles = _build_ncycles(_CSet)
    nth = _build_nth(_CSet)
    pairwise = _build_pairwise(_CSet)
    prepend = _build_prepend(_CSet)
    quantify = _build_quantify(_CSet)
    repeatfunc = classmethod(_build_repeatfunc(_CSet))
    tail = _build_tail(_CSet)
    take = _build_take(_CSet)

    # more-itertools

    chunked = _build_chunked(_CSet)
    distribute = _build_distribute(_CSet)
    divide = _build_divide(_CSet)

    # multiprocessing

    pmap = _build_pmap(_CSet)
    pstarmap = _build_pstarmap(_CSet)

    # pathlib

    iterdir = classmethod(_build_iterdir(_CSet))

    # extra public

    def one(self: CSet[T]) -> T:
        return self.iter().one()

    def pipe(
        self: CSet[T], func: Callable[..., Iterable[U]], *args: Any, index: int = 0, **kwargs: Any,
    ) -> CSet[U]:
        return self.iter().pipe(func, *args, index=index, **kwargs).set()


class CFrozenSet(FrozenSet[T]):
    """A frozenset with chainable methods."""

    # built-ins

    all = _build_all(_CFrozenSet)  # noqa: A003
    any = _build_any(_CFrozenSet)  # noqa: A003
    dict = _build_dict(_CFrozenSet)  # noqa: A003
    enumerate = _build_enumerate(_CFrozenSet)  # noqa: A003
    filter = _build_filter(_CFrozenSet)  # noqa: A003
    frozenset = _build_frozenset(_CFrozenSet)  # noqa: A003
    iter = _build_iter(_CFrozenSet)  # noqa: A003
    len = _build_len(_CFrozenSet)  # noqa: A003
    list = _build_list(_CFrozenSet)  # noqa: A003
    map = _build_map(_CFrozenSet)  # noqa: A003
    max = _build_maxmin(_CFrozenSet, func=max)  # noqa: A003
    min = _build_maxmin(_CFrozenSet, func=min)  # noqa: A003
    range = classmethod(_build_range(_CFrozenSet))  # noqa: A003
    set = _build_set(_CFrozenSet)  # noqa: A003
    sorted = _build_sorted(_CFrozenSet)  # noqa: A003
    sum = _build_sum(_CFrozenSet)  # noqa: A003
    tuple = _build_tuple(_CFrozenSet)  # noqa: A003
    zip = _build_zip(_CFrozenSet)  # noqa: A003

    # set & frozenset methods

    def union(self: CFrozenSet[T], *others: Iterable[U]) -> CFrozenSet[Union[T, U]]:
        return CFrozenSet(super().union(*others))

    def intersection(self: CFrozenSet[T], *others: Iterable[U]) -> CFrozenSet[Union[T, U]]:
        return CFrozenSet(super().intersection(*others))

    def difference(self: CFrozenSet[T], *others: Iterable[U]) -> CFrozenSet[Union[T, U]]:
        return CFrozenSet(super().difference(*others))

    def symmetric_difference(self: CFrozenSet[T], other: Iterable[U]) -> CFrozenSet[Union[T, U]]:
        return CFrozenSet(super().symmetric_difference(other))

    def copy(self: CFrozenSet[T]) -> CFrozenSet[T]:
        return CFrozenSet(super().copy())

    # functools

    reduce = _build_reduce(_CFrozenSet)

    # itertools

    accumulate = _build_accumulate(_CFrozenSet)
    chain = _build_chain(_CFrozenSet)
    combinations = _build_combinations(_CFrozenSet)
    combinations_with_replacement = _build_combinations_with_replacement(_CFrozenSet)
    compress = _build_compress(_CFrozenSet)
    dropwhile = _build_dropwhile(_CFrozenSet)
    filterfalse = _build_filterfalse(_CFrozenSet)
    groupby = _build_groupby(_CFrozenSet)
    islice = _build_islice(_CFrozenSet)
    permutations = _build_permutations(_CFrozenSet)
    product = _build_product(_CFrozenSet)
    repeat = classmethod(_build_repeat(_CFrozenSet))
    starmap = _build_starmap(_CFrozenSet)
    takewhile = _build_takewhile(_CFrozenSet)
    tee = _build_tee(_CFrozenSet)
    zip_longest = _build_zip_longest(_CFrozenSet)

    # itertools-recipes

    all_equal = _build_all_equal(_CFrozenSet)
    dotproduct = _build_dotproduct(_CFrozenSet)
    flatten = _build_flatten(_CFrozenSet)
    ncycles = _build_ncycles(_CFrozenSet)
    nth = _build_nth(_CFrozenSet)
    pairwise = _build_pairwise(_CFrozenSet)
    prepend = _build_prepend(_CFrozenSet)
    quantify = _build_quantify(_CFrozenSet)
    repeatfunc = classmethod(_build_repeatfunc(_CFrozenSet))
    tail = _build_tail(_CFrozenSet)
    take = _build_take(_CFrozenSet)

    # more-itertools

    chunked = _build_chunked(_CFrozenSet)
    distribute = _build_distribute(_CFrozenSet)
    divide = _build_divide(_CFrozenSet)

    # multiprocessing

    pmap = _build_pmap(_CFrozenSet)
    pstarmap = _build_pstarmap(_CFrozenSet)

    # pathlib

    iterdir = classmethod(_build_iterdir(_CFrozenSet))

    # extra public

    def one(self: CFrozenSet[T]) -> T:
        return self.iter().one()

    def pipe(
        self: CFrozenSet[T],
        func: Callable[..., Iterable[U]],
        *args: Any,
        index: int = 0,
        **kwargs: Any,
    ) -> CFrozenSet[U]:
        return self.iter().pipe(func, *args, index=index, **kwargs).frozenset()


class CDict(Dict[T, U]):
    """A dictionary with chainable methods."""

    def keys(self: CDict[T, Any]) -> CIterable[T]:
        return CIterable(super().keys())

    def values(self: CDict[Any, U]) -> CIterable[U]:
        return CIterable(super().values())

    def items(self: CDict[T, U]) -> CIterable[Tuple[T, U]]:
        return CIterable(super().items())

    # built-ins

    def filter_keys(self: CDict[T, U], func: Callable[[T], bool]) -> CDict[T, U]:  # dead: disable
        def inner(item: Tuple[T, U]) -> bool:
            key, _ = item
            return func(key)

        return self.items().filter(inner).dict()

    def filter_values(self: CDict[T, U], func: Callable[[U], bool]) -> CDict[T, U]:  # dead: disable
        def inner(item: Tuple[T, U]) -> bool:
            _, value = item
            return func(value)

        return self.items().filter(inner).dict()

    def filter_items(  # dead: disable
        self: CDict[T, U], func: Callable[[T, U], bool],
    ) -> CDict[T, U]:
        def inner(item: Tuple[T, U]) -> bool:
            key, value = item
            return func(key, value)

        return self.items().filter(inner).dict()

    def map_keys(self: CDict[T, U], func: Callable[[T], V]) -> CDict[V, U]:  # dead: disable
        def inner(item: Tuple[T, U]) -> Tuple[V, U]:
            key, value = item
            return func(key), value

        return self.items().map(inner).dict()

    def map_values(self: CDict[T, U], func: Callable[[U], V]) -> CDict[T, V]:
        def inner(item: Tuple[T, U]) -> Tuple[T, V]:
            key, value = item
            return key, func(value)

        return self.items().map(inner).dict()

    def map_items(  # dead: disable
        self: CDict[T, U], func: Callable[[T, U], Tuple[V, W]],
    ) -> CDict[V, W]:
        def inner(item: Tuple[T, U]) -> Tuple[V, W]:
            key, value = item
            return func(key, value)

        return self.items().map(inner).dict()
