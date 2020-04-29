from __future__ import annotations

import builtins
import functools
import itertools
from functools import partial
from functools import reduce
from itertools import accumulate
from itertools import chain
from itertools import combinations
from itertools import combinations_with_replacement
from itertools import compress
from itertools import count
from itertools import cycle
from itertools import dropwhile
from itertools import filterfalse
from itertools import groupby
from itertools import islice
from itertools import permutations
from itertools import product
from itertools import repeat
from itertools import starmap
from itertools import takewhile
from itertools import tee
from itertools import zip_longest
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
from typing import Generic
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
from more_itertools.recipes import all_equal
from more_itertools.recipes import consume
from more_itertools.recipes import dotproduct
from more_itertools.recipes import first_true
from more_itertools.recipes import flatten
from more_itertools.recipes import grouper
from more_itertools.recipes import iter_except
from more_itertools.recipes import ncycles
from more_itertools.recipes import nth
from more_itertools.recipes import nth_combination
from more_itertools.recipes import padnone
from more_itertools.recipes import pairwise
from more_itertools.recipes import partition
from more_itertools.recipes import powerset
from more_itertools.recipes import prepend
from more_itertools.recipes import quantify
from more_itertools.recipes import random_combination
from more_itertools.recipes import random_combination_with_replacement
from more_itertools.recipes import random_permutation
from more_itertools.recipes import random_product
from more_itertools.recipes import repeatfunc
from more_itertools.recipes import roundrobin
from more_itertools.recipes import tabulate
from more_itertools.recipes import tail
from more_itertools.recipes import take
from more_itertools.recipes import unique_everseen
from more_itertools.recipes import unique_justseen

from functional_itertools.compat import MAX_MIN_KEY_ANNOTATION
from functional_itertools.compat import MAX_MIN_KEY_DEFAULT
from functional_itertools.errors import EmptyIterableError
from functional_itertools.errors import MultipleElementsError
from functional_itertools.errors import StopArgumentMissing
from functional_itertools.errors import UnsupportVersionError
from functional_itertools.methods.base import CIterableOrCList
from functional_itertools.methods.base import Template
from functional_itertools.utilities import drop_sentinel
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

    def __repr__(self: CIterable[Any]) -> str:
        return f"{type(self).__name__}({self._iterable!r})"

    def __str__(self: CIterable[Any]) -> str:
        return f"{type(self).__name__}({self._iterable})"

    # built-in

    def all(self: CIterable[Any]) -> bool:  # noqa: A003
        return all(self)

    def any(self: CIterable[Any]) -> bool:  # noqa: A003
        return any(self)

    def dict(self: CIterable[Tuple[T, U]]) -> CDict[T, U]:  # noqa: A003
        return CDict(dict(self))

    def enumerate(self: CIterable[T], start: int = 0) -> CIterable[Tuple[int, T]]:  # noqa: A003
        return CIterable(enumerate(self, start=start))

    def filter(  # noqa: A003
        self: CIterable[T], func: Optional[Callable[[T], bool]],
    ) -> CIterable[T]:
        return CIterable(filter(func, self))

    def frozenset(self: CIterable[T]) -> CFrozenSet[T]:  # noqa: A003
        return CFrozenSet(self)

    def iter(self: CIterable[T]) -> CIterable[T]:  # noqa: A003
        return CIterable(self)

    def list(self: CIterable[T]) -> CList[T]:  # noqa: A003
        return CList(self)

    def map(  # noqa: A003
        self: CIterable[T],
        func: Callable[..., U],
        *iterables: Iterable,
        parallel: bool = False,
        processes: Optional[int] = None,
    ) -> CIterable[U]:
        if parallel:
            if iterables:
                raise ValueError("Additional iterables cannot be used with 'parallel'")
            else:
                try:
                    with Pool(processes=processes) as pool:
                        return CIterable(pool.map(func, self))
                except AssertionError as error:
                    (msg,) = error.args
                    if msg == "daemonic processes are not allowed to have children":
                        return self.map(func)
                    else:
                        raise error
        else:
            return CIterable(map(func, self, *iterables))

    def max(  # noqa: A003
        self: CIterable[T],
        *,
        key: MAX_MIN_KEY_ANNOTATION = MAX_MIN_KEY_DEFAULT,
        default: Union[T, Sentinel] = sentinel,
    ) -> T:
        _, kwargs = drop_sentinel(key=key, default=default)
        return max(self, **kwargs)

    def min(  # noqa: A003
        self: CIterable[T],
        *,
        key: MAX_MIN_KEY_ANNOTATION = MAX_MIN_KEY_DEFAULT,
        default: Union[T, Sentinel] = sentinel,
    ) -> T:
        _, kwargs = drop_sentinel(key=key, default=default)
        return min(self, **kwargs)

    @classmethod  # noqa: A003
    def range(  # noqa: A003
        cls: Type[CIterable], start: int, stop: Optional[int] = None, step: Optional[int] = None,
    ) -> CIterable[int]:
        """
        >>> CIterable.range(5).list()
        [0, 1, 2, 3, 4]
        >>> CIterable.range(1, 5).list()
        [1, 2, 3, 4]
        >>> CIterable.range(1, 5, 2).list()
        [1, 3]
        """
        if (stop is None) and (step is not None):
            raise ValueError("'stop' cannot be None if 'step' is provided")
        else:
            return cls(
                range(
                    start, *(() if stop is None else (stop,)), *(() if step is None else (step,)),
                ),
            )

    def set(self: CIterable[T]) -> CSet[T]:  # noqa: A003
        """
        >>> CIterable([1, 2, 2, 3]).set()
        CSet({1, 2, 3})
        """
        return CSet(self)

    def sorted(  # noqa: A003
        self: CIterable[T], *, key: Optional[Callable[[T], Any]] = None, reverse: bool = False,
    ) -> CList[T]:
        return CList(sorted(self, key=key, reverse=reverse))

    def sum(self: CIterable[T], start: Union[T, int] = 0) -> Union[T, int]:  # noqa: A003
        args, _ = drop_sentinel(start)
        return sum(self, *args)

    def tuple(self: CIterable[T]) -> CTuple[T, ...]:  # noqa: A003
        return CTuple(self)

    def zip(  # noqa: A003
        self: CIterable[T], *iterables: Iterable[U],
    ) -> CIterable[CTuple[Union[T, U]]]:
        return CIterable(map(CTuple, zip(self, *iterables)))

    # functools

    def reduce(
        self: CIterable[T], func: Callable[[T, T], T], initial: Union[U, Sentinel] = sentinel,
    ) -> Any:
        args, _ = drop_sentinel(initial)
        try:
            result = reduce(func, self, *args)
        except TypeError as error:
            (msg,) = error.args
            if msg == "reduce() of empty sequence with no initial value":
                raise EmptyIterableError from None
            else:
                raise error
        else:
            if isinstance(result, list):
                return CList(result)
            elif isinstance(result, set):
                return CSet(result)
            elif isinstance(result, frozenset):
                return CFrozenSet(result)
            elif isinstance(result, dict):
                return CDict(result)
            else:
                return result

    # itertools

    @classmethod
    def count(cls: Type[CIterable], start: int = 0, step: int = 1) -> CIterable[int]:
        return cls(count(start=start, step=step))

    def cycle(self: CIterable[T]) -> CIterable[T]:
        return CIterable(cycle(self))

    @classmethod
    def repeat(cls: Type[CIterable], x: T, times: Optional[int] = None) -> CIterable[T]:
        return cls(repeat(x, *(() if times is None else (times,))))

    def accumulate(
        self: CIterable[T],
        func: Callable[[T, T], T] = add,
        *,
        initial: Union[U, Sentinel] = sentinel,
    ) -> CIterable[Union[T, U]]:
        if VERSION is Version.py37:
            if initial is sentinel:
                return CIterable(accumulate(self, func))
            else:
                raise ValueError("The 'initial' argument is introduced in Python 3.8")
        elif VERSION is Version.py38:
            return CIterable(
                accumulate(self, func, initial=None if initial is sentinel else sentinel),
            )
        else:
            raise UnsupportVersionError(VERSION)  # pragma: no cover

    def chain(self: CIterable[T], *iterables: Iterable[U]) -> CIterable[Union[T, U]]:
        return CIterable(chain(self, *iterables))

    def compress(self: CIterable[T], selectors: Iterable[Any]) -> CIterable[T]:
        return CIterable(compress(self, selectors))

    def dropwhile(self: CIterable[T], func: Callable[[T], bool]) -> CIterable[T]:
        return CIterable(dropwhile(func, self))

    def filterfalse(self: CIterable[T], func: Callable[[T], bool]) -> CIterable[T]:
        return CIterable(filterfalse(func, self))

    def groupby(
        self: CIterable[T], key: Optional[Callable[[T], U]] = None,
    ) -> CIterable[Tuple[U, CIterable[T]]]:
        def inner(x: Tuple[U, Iterator[T]]) -> Tuple[U, CIterable[T]]:
            key, group = x
            return key, CIterable(group)

        return CIterable(groupby(self, key=key)).map(inner)

    def islice(
        self: CIterable[T],
        start: int,
        stop: Union[int, Sentinel] = sentinel,
        step: Union[int, Sentinel] = sentinel,
    ) -> CIterable[T]:
        args, _ = drop_sentinel(stop, step)
        return CIterable(islice(self, start, *args))

    def starmap(
        self: CIterable[Tuple[T, ...]], func: Callable[[Tuple[T, ...]], U],
    ) -> CIterable[U]:
        return CIterable(starmap(func, self))

    def takewhile(self: CIterable[T], func: Callable[[T], bool]) -> CIterable[T]:
        return CIterable(takewhile(func, self))

    def tee(self: CIterable[T], n: int = 2) -> CIterable[Iterator[T]]:
        return CIterable(tee(self, n)).map(CIterable)

    def zip_longest(
        self: CIterable[T], *iterables: Iterable[U], fillvalue: V = None,
    ) -> CIterable[Tuple[Union[T, U, V]]]:
        return CIterable(zip_longest(self, *iterables, fillvalue=fillvalue))

    def product(
        self: CIterable[T], *iterables: Iterable[U], repeat: int = 1,
    ) -> CIterable[Tuple[Union[T, U], ...]]:
        return CIterable(product(self, *iterables, repeat=repeat))

    def permutations(self: CIterable[T], r: Optional[int] = None) -> CIterable[Tuple[T, ...]]:
        return CIterable(permutations(self, r=r))

    def combinations(self: CIterable[T], r: int) -> CIterable[Tuple[T, ...]]:
        return CIterable(combinations(self, r))

    def combinations_with_replacement(self: CIterable[T], r: int) -> CIterable[Tuple[T, ...]]:
        return CIterable(combinations_with_replacement(self, r))

    # itertools-recipes

    def take(self: CIterable[T], n: int) -> CIterable[T]:
        return CIterable(take(n, self))

    def prepend(self: CIterable[T], value: U) -> CIterable[Union[T, U]]:
        return CIterable(prepend(value, self))

    @classmethod
    def tabulate(cls: Type[CIterable], func: Callable[[int], T], start: int = 0) -> CIterable[T]:
        return cls(tabulate(func, start=start))

    def tail(self: CIterable[T], n: int) -> CIterable[T]:
        return CIterable(tail(n, self))

    def consume(self: CIterable[T], n: Optional[int] = None) -> CIterable[T]:
        iterator = iter(self)
        consume(iterator, n=n)
        return CIterable(iterator)

    def nth(self: CIterable[T], n: int, default: U = None) -> Union[T, U]:
        return nth(self, n, default=default)

    def all_equal(self: CIterable[Any]) -> bool:
        return all_equal(self)

    def quantify(self: CIterable[T], pred: Callable[[T], bool] = bool) -> int:
        return quantify(self, pred=pred)

    def padnone(self: CIterable[T]) -> CIterable[Optional[T]]:
        return CIterable(padnone(self))

    def ncycles(self: CIterable[T], n: int) -> CIterable[T]:
        return CIterable(ncycles(self, n))

    def dotproduct(self: CIterable[T], iterable: Iterable[T]) -> T:
        return dotproduct(self, iterable)

    def flatten(self: CIterable[Iterable[T]]) -> CIterable[T]:
        return CIterable(flatten(self))

    @classmethod
    def repeatfunc(
        cls: Type[CIterable], func: Callable[..., T], times: Optional[int] = None, *args: Any,
    ) -> CIterable[T]:
        return cls(repeatfunc(func, times, *args))

    def pairwise(self: CIterable[T]) -> CIterable[Tuple[T, T]]:
        return CIterable(pairwise(self))

    def grouper(
        self: CIterable[T], n: int, fillvalue: U = None,
    ) -> CIterable[Tuple[Union[T, U], ...]]:
        return CIterable(grouper(self, n, fillvalue=fillvalue))

    def partition(
        self: CIterable[T], func: Callable[[T], bool],
    ) -> Tuple[CIterable[T], CIterable[T]]:
        return CIterable(partition(func, self)).map(CIterable).tuple()

    def powerset(self: CIterable[T]) -> CIterable[Tuple[T, ...]]:
        return CIterable(powerset(self))

    def roundrobin(self: CIterable[T], *iterables: Iterable[U]) -> CIterable[Tuple[T, U]]:
        return CIterable(roundrobin(self, *iterables))

    def unique_everseen(
        self: CIterable[T], key: Optional[Callable[[T], Any]] = None,
    ) -> CIterable[T]:
        return CIterable(unique_everseen(self, key=key))

    def unique_justseen(
        self: CIterable[T], key: Optional[Callable[[T], Any]] = None,
    ) -> CIterable[T]:
        return CIterable(unique_justseen(self, key=key))

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
        return first_true(self, default=default, pred=pred)

    def random_product(
        self: CIterable[T], *iterables: Iterable[U], repeat: int = 1,
    ) -> Tuple[Union[T, U], ...]:
        return random_product(self, *iterables, repeat=repeat)

    def random_permutation(self: CIterable[T], r: Optional[int] = None) -> Tuple[Union[T, U], ...]:
        return random_permutation(self, r=r)

    def random_combination(self: CIterable[T], r: int) -> Tuple[T, ...]:
        return random_combination(self, r)

    def random_combination_with_replacement(self: CIterable[T], r: int) -> Tuple[T, ...]:
        return random_combination_with_replacement(self, r)

    def nth_combination(self: CIterable[T], r: int, index: int) -> Tuple[T, ...]:
        return nth_combination(self, r, index)

    # multiprocessing

    def pmap(
        self: CIterable[T], func: Callable[[T], U], *, processes: Optional[int] = None,
    ) -> CIterable[U]:
        warn(
            "'pmap' is going to be deprecated; use 'map(..., parallel=True)' instead",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.map(func, parallel=True, processes=processes)

    def pstarmap(
        self: CIterable[Tuple[T, ...]],
        func: Callable[[Tuple[T, ...]], U],
        *,
        processes: Optional[int] = None,
    ) -> CIterable[U]:
        with Pool(processes=processes) as pool:
            return CIterable(pool.starmap(func, self))

    # pathlib

    @classmethod
    def iterdir(cls: Type[CIterable], path: Union[Path, str]) -> CIterable[Path]:
        return cls(Path(path).iterdir())

    # extra public

    def append(self: CIterable[T], value: U) -> CIterable[Union[T, U]]:  # dead: disable
        return self.chain([value])

    def first(self: CIterable[T]) -> T:
        try:
            return next(iter(self))
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
        new_args = chain(islice(args, index), [self], islice(args, index, None))
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

    # built-in

    def all(self: CList[Any]) -> bool:  # noqa: A003
        return self.iter().all()

    def any(self: CList[Any]) -> bool:  # noqa: A003
        return self.iter().any()

    def copy(self: CList[T]) -> CList[T]:
        return CList(super().copy())

    def dict(self: CList[Tuple[T, U]]) -> CDict[T, U]:  # noqa: A003
        return self.iter().dict()

    def enumerate(self: CList[T], start: int = 0) -> CList[Tuple[int, T]]:  # noqa: A003
        return self.iter().enumerate(start=start).list()

    def filter(self: CList[T], func: Optional[Callable[[T], bool]]) -> CList[T]:  # noqa: A003
        return self.iter().filter(func).list()

    def frozenset(self: CList[T]) -> CFrozenSet[T]:  # noqa: A003
        return self.iter().frozenset()

    def iter(self: CList[T]) -> CIterable[T]:  # noqa: A003
        return CIterable(self)

    def len(self: CList[T]) -> int:  # noqa: A003
        return len(self)

    def list(self: CFrozenSet[T]) -> CList[T]:  # noqa: A003
        return self.iter().list()

    def map(
        self: CList[T],
        func: Callable[..., U],
        *iterables: Iterable,
        parallel: bool = False,
        processes: Optional[int] = None,
    ) -> CList[U]:  # noqa: A003
        return self.iter().map(func, *iterables, parallel=parallel, processes=processes).list()

    def max(  # noqa: A003
        self: CList[T],
        *,
        key: MAX_MIN_KEY_ANNOTATION = MAX_MIN_KEY_DEFAULT,
        default: Union[T, Sentinel] = sentinel,
    ) -> T:
        return self.iter().max(key=key, default=default)

    def min(  # noqa: A003
        self: CList[T],
        *,
        key: MAX_MIN_KEY_ANNOTATION = MAX_MIN_KEY_DEFAULT,
        default: Union[T, Sentinel] = sentinel,
    ) -> T:
        return self.iter().min(key=key, default=default)

    @classmethod  # noqa: A003
    def range(  # noqa: A003
        cls: Type[CList], start: int, stop: Optional[int] = None, step: Optional[int] = None,
    ) -> CList[int]:
        """
        >>> CList.range(5)
        [0, 1, 2, 3, 4]
        >>> CList.range(1, 5)
        [1, 2, 3, 4]
        >>> CList.range(1, 5, 2)
        [1, 3]
        """
        return cls(CIterable.range(start, stop=stop, step=step))

    def reversed(self: CList[T]) -> CList[T]:  # noqa: A003
        return CList(reversed(self))

    def set(self: CList[T]) -> CSet[T]:  # noqa: A003
        return self.iter().set()

    def sort(  # dead: disable
        self: CList[T], *, key: Optional[Callable[[T], Any]] = None, reverse: bool = False,
    ) -> CList[T]:
        warn("Use the 'sorted' method instead of 'sort'")
        return self.sorted(key=key, reverse=reverse)

    def sorted(  # noqa: A003
        self: CList[T], *, key: Optional[Callable[[T], Any]] = None, reverse: bool = False,
    ) -> CList[T]:
        return self.iter().sorted(key=key, reverse=reverse)

    def sum(self: CList[T], start: Union[T, int] = 0) -> Union[T, int]:  # noqa: A003
        return self.iter().sum(start=start)

    def tuple(self: CList[T]) -> CTuple[T]:  # noqa: A003
        return self.iter().tuple()

    def zip(self: CList[T], *iterables: Iterable[U]) -> CList[CTuple[Union[T, U]]]:  # noqa: A003
        return self.iter().zip(*iterables).list()

    # functools

    def reduce(
        self: CList[T], func: Callable[[T, T], T], initial: Union[U, Sentinel] = sentinel,
    ) -> Any:
        return self.iter().reduce(func, initial=initial)

    # itertools

    @classmethod
    def repeat(cls: Type[CList], x: T, times: int) -> CList[T]:
        return cls(CIterable.repeat(x, times=times))

    def accumulate(
        self: CList[T], func: Callable[[T, T], T] = add, *, initial: Union[U, Sentinel] = sentinel,
    ) -> CList[Union[T, U]]:
        return self.iter().accumulate(func, initial=initial).list()

    def chain(self: CList[T], *iterables: Iterable[U]) -> CList[Union[T, U]]:
        return self.iter().chain(*iterables).list()

    def compress(self: CList[T], selectors: Iterable[Any]) -> CList[T]:
        return self.iter().compress(selectors).list()

    def dropwhile(self: CList[T], func: Callable[[T], bool]) -> CList[T]:
        return self.iter().dropwhile(func).list()

    def filterfalse(self: CList[T], func: Callable[[T], bool]) -> CList[T]:
        return self.iter().filterfalse(func).list()

    def groupby(
        self: CList[T], key: Optional[Callable[[T], U]] = None,
    ) -> CList[Tuple[U, CList[T]]]:
        return self.iter().groupby(key=key).map(lambda x: (x[0], CList(x[1]))).list()

    def islice(
        self: CList[T],
        start: int,
        stop: Union[int, Sentinel] = sentinel,
        step: Union[int, Sentinel] = sentinel,
    ) -> CList[T]:
        return self.iter().islice(start, stop=stop, step=step).list()

    def starmap(self: CList[Tuple[T, ...]], func: Callable[[Tuple[T, ...]], U]) -> CList[U]:
        return self.iter().starmap(func).list()

    def takewhile(self: CList[T], func: Callable[[T], bool]) -> CList[T]:
        return self.iter().takewhile(func).list()

    def tee(self: CList[T], n: int = 2) -> CList[CList[T]]:
        return self.iter().tee(n=n).list().map(CList)

    def zip_longest(
        self: CList[T], *iterables: Iterable[U], fillvalue: V = None,
    ) -> CList[Tuple[Union[T, U, V]]]:
        return self.iter().zip_longest(*iterables, fillvalue=fillvalue).list()

    def product(
        self: CList[T], *iterables: Iterable[U], repeat: int = 1,
    ) -> CList[Tuple[Union[T, U], ...]]:
        return self.iter().product(*iterables, repeat=repeat).list()

    def permutations(self: CList[T], r: Optional[int] = None) -> CList[Tuple[T, ...]]:
        return self.iter().permutations(r=r).list()

    def combinations(self: CList[T], r: int) -> CList[Tuple[T, ...]]:
        return self.iter().combinations(r).list()

    def combinations_with_replacement(self: CList[T], r: int) -> CList[Tuple[T, ...]]:
        return self.iter().combinations_with_replacement(r).list()

    # itertools-recipes

    def take(self: CList[T], n: int) -> CList[T]:
        return self.iter().take(n).list()

    def prepend(self: CList[T], value: U) -> CList[Union[T, U]]:
        return self.iter().prepend(value).list()

    def tail(self: CList[T], n: int) -> CList[T]:
        return self.iter().tail(n).list()

    def consume(self: CList[T], n: Optional[int] = None) -> CList[T]:
        return self.iter().consume(n=n).list()

    def nth(self: CList[T], n: int, default: U = None) -> Union[T, U]:
        return self.iter().nth(n, default=default)

    def all_equal(self: CList[Any]) -> bool:
        return self.iter().all_equal()

    def quantify(self: CList[T], pred: Callable[[T], bool] = bool) -> int:
        return self.iter().quantify(pred=pred)

    def ncycles(self: CList[T], n: int) -> CList[T]:
        return self.iter().ncycles(n).list()

    def dotproduct(self: CList[T], iterable: Iterable[T]) -> T:
        return self.iter().dotproduct(iterable)

    def flatten(self: CList[Iterable[T]]) -> CList[T]:
        return self.iter().flatten().list()

    @classmethod
    def repeatfunc(
        cls: Type[CList], func: Callable[..., T], times: Optional[int] = None, *args: Any,
    ) -> CList[T]:
        return CIterable.repeatfunc(func, times, *args).list()

    def pairwise(self: CList[T]) -> CList[Tuple[T, T]]:
        return self.iter().pairwise().list()

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

    # multiprocessing

    def pmap(
        self: CList[T], func: Callable[[T], U], *, processes: Optional[int] = None,
    ) -> CList[U]:
        warn(
            "'pmap' is going to be deprecated; use 'map(..., parallel=True)' instead",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.iter().pmap(func, processes=processes).list()

    def pstarmap(
        self: CList[Tuple[T, ...]],
        func: Callable[[Tuple[T, ...]], U],
        *,
        processes: Optional[int] = None,
    ) -> CList[U]:
        return self.iter().pstarmap(func, processes=processes).list()

    # pathlib

    @classmethod
    def iterdir(cls: Type[CList], path: Union[Path, str]) -> CList[Path]:
        return cls(CIterable.iterdir(path))

    # extra public

    def one(self: CList[T]) -> T:
        return self.iter().one()

    def pipe(
        self: CList[T], func: Callable[..., Iterable[U]], *args: Any, index: int = 0, **kwargs: Any,
    ) -> CList[U]:
        return self.iter().pipe(func, *args, index=index, **kwargs).list()

    def unzip(self: CList[Tuple[T, ...]]) -> Tuple[CList[T], ...]:
        return CList(self.iter().unzip()).map(CList)


class CTuple(tuple, Generic[T]):
    """A homogenous tuple with chainable methods."""

    def __getitem__(self: CTuple[T], item: Union[int, slice]) -> Union[T, CTuple[T]]:
        out = super().__getitem__(item)
        if isinstance(out, tuple):
            return CTuple(out)
        else:
            return out

    # built-in

    def all(self: CTuple[Any]) -> bool:  # noqa: A003
        return self.iter().all()

    def any(self: CTuple[Any]) -> bool:  # noqa: A003
        return self.iter().any()

    def dict(self: CTuple[Tuple[T, U]]) -> CDict[T, U]:  # noqa: A003
        return self.iter().dict()

    def enumerate(self: CTuple[T], start: int = 0) -> CTuple[Tuple[int, T]]:  # noqa: A003
        return self.iter().enumerate(start=start).tuple()

    def filter(self: CTuple[T], func: Optional[Callable[[T], bool]]) -> CTuple[T]:  # noqa: A003
        return self.iter().filter(func).tuple()

    def frozenset(self: CTuple[T]) -> CFrozenSet[T]:  # noqa: A003
        return self.iter().frozenset()

    def iter(self: CTuple[T]) -> CIterable[T]:  # noqa: A003
        return CIterable(self)

    def len(self: CTuple[T]) -> int:  # noqa: A003
        return len(self)

    def list(self: CTuple[T]) -> CList[T]:  # noqa: A003
        return self.iter().list()

    def map(  # noqa: A003
        self: CTuple[T],
        func: Callable[..., U],
        *iterables: Iterable,
        parallel: bool = False,
        processes: Optional[int] = None,
    ) -> CTuple[U]:
        return self.iter().map(func, *iterables, parallel=parallel, processes=processes).tuple()

    def max(  # noqa: A003
        self: CTuple[T],
        *,
        key: MAX_MIN_KEY_ANNOTATION = MAX_MIN_KEY_DEFAULT,
        default: Union[T, Sentinel] = sentinel,
    ) -> T:
        return self.iter().max(key=key, default=default)

    def min(  # noqa: A003
        self: CTuple[T],
        *,
        key: MAX_MIN_KEY_ANNOTATION = MAX_MIN_KEY_DEFAULT,
        default: Union[T, Sentinel] = sentinel,
    ) -> T:
        return self.iter().min(key=key, default=default)

    @classmethod  # noqa: A003
    def range(  # noqa: A003
        cls: Type[CTuple], start: int, stop: Optional[int] = None, step: Optional[int] = None,
    ) -> CTuple[int]:
        return cls(CIterable.range(start, stop=stop, step=step))

    def reversed(self: CTuple[T]) -> CTuple[T]:  # noqa: A003
        return CTuple(reversed(self))

    def set(self: CTuple[T]) -> CSet[T]:  # noqa: A003
        return self.iter().set()

    def sort(  # dead: disable
        self: CTuple[T], *, key: Optional[Callable[[T], Any]] = None, reverse: bool = False,
    ) -> CTuple[T]:
        warn("Use the 'sorted' method instead of 'sort'")
        return self.sorted(key=key, reverse=reverse)

    def sorted(  # noqa: A003
        self: CTuple[T], *, key: Optional[Callable[[T], Any]] = None, reverse: bool = False,
    ) -> CTuple[T]:
        return self.iter().sorted(key=key, reverse=reverse)

    def sum(self: CTuple[T], start: Union[T, int] = 0) -> Union[T, int]:  # noqa: A003
        return self.iter().sum(start=start)

    def tuple(self: CTuple[T]) -> CTuple[T]:  # noqa: A003
        return self.iter().tuple()

    def zip(self: CTuple[T], *iterables: Iterable[U]) -> CTuple[CTuple[Union[T, U]]]:  # noqa: A003
        return self.iter().zip(*iterables).tuple()

    # functools

    def reduce(
        self: CTuple[T], func: Callable[[T, T], T], initial: Union[U, Sentinel] = sentinel,
    ) -> Any:
        return self.iter().reduce(func, initial=initial)

    # itertools

    @classmethod
    def repeat(cls: Type[CTuple], x: T, times: int) -> CTuple[T]:
        return cls(CIterable.repeat(x, times=times))

    def accumulate(
        self: CTuple[T], func: Callable[[T, T], T] = add, *, initial: Union[U, Sentinel] = sentinel,
    ) -> CTuple[Union[T, U]]:
        return self.iter().accumulate(func, initial=initial).tuple()

    def chain(self: CTuple[T], *iterables: Iterable[U]) -> CTuple[Union[T, U]]:
        return self.iter().chain(*iterables).list()

    def compress(self: CTuple[T], selectors: Iterable[Any]) -> CTuple[T]:
        return self.iter().compress(selectors).list()

    def dropwhile(self: CTuple[T], func: Callable[[T], bool]) -> CTuple[T]:
        return self.iter().dropwhile(func).list()

    def filterfalse(self: CTuple[T], func: Callable[[T], bool]) -> CTuple[T]:
        return self.iter().filterfalse(func).list()

    def groupby(
        self: CTuple[T], key: Optional[Callable[[T], U]] = None,
    ) -> CTuple[Tuple[U, CTuple[T]]]:
        return self.iter().groupby(key=key).map(lambda x: (x[0], CTuple(x[1]))).list()

    def islice(
        self: CTuple[T],
        start: int,
        stop: Union[int, Sentinel] = sentinel,
        step: Union[int, Sentinel] = sentinel,
    ) -> CTuple[T]:
        return self.iter().islice(start, stop=stop, step=step).list()

    def starmap(self: CTuple[Tuple[T, ...]], func: Callable[[Tuple[T, ...]], U]) -> CTuple[U]:
        return self.iter().starmap(func).list()

    def takewhile(self: CTuple[T], func: Callable[[T], bool]) -> CTuple[T]:
        return self.iter().takewhile(func).list()

    def tee(self: CTuple[T], n: int = 2) -> CTuple[CTuple[T]]:
        return self.iter().tee(n=n).list().map(CTuple)

    def zip_longest(
        self: CTuple[T], *iterables: Iterable[U], fillvalue: V = None,
    ) -> CTuple[Tuple[Union[T, U, V]]]:
        return self.iter().zip_longest(*iterables, fillvalue=fillvalue).list()

    def product(
        self: CTuple[T], *iterables: Iterable[U], repeat: int = 1,
    ) -> CTuple[Tuple[Union[T, U], ...]]:
        return self.iter().product(*iterables, repeat=repeat).list()

    def permutations(self: CTuple[T], r: Optional[int] = None) -> CTuple[Tuple[T, ...]]:
        return self.iter().permutations(r=r).list()

    def combinations(self: CTuple[T], r: int) -> CTuple[Tuple[T, ...]]:
        return self.iter().combinations(r).list()

    def combinations_with_replacement(self: CTuple[T], r: int) -> CTuple[Tuple[T, ...]]:
        return self.iter().combinations_with_replacement(r).list()

    # itertools-recipes

    def take(self: CTuple[T], n: int) -> CTuple[T]:
        return self.iter().take(n).list()

    def prepend(self: CTuple[T], value: U) -> CTuple[Union[T, U]]:
        return self.iter().prepend(value).list()

    def tail(self: CTuple[T], n: int) -> CTuple[T]:
        return self.iter().tail(n).list()

    def consume(self: CTuple[T], n: Optional[int] = None) -> CTuple[T]:
        return self.iter().consume(n=n).list()

    def nth(self: CTuple[T], n: int, default: U = None) -> Union[T, U]:
        return self.iter().nth(n, default=default)

    def all_equal(self: CTuple[Any]) -> bool:
        return self.iter().all_equal()

    def quantify(self: CTuple[T], pred: Callable[[T], bool] = bool) -> int:
        return self.iter().quantify(pred=pred)

    def ncycles(self: CTuple[T], n: int) -> CTuple[T]:
        return self.iter().ncycles(n).list()

    def dotproduct(self: CTuple[T], iterable: Iterable[T]) -> T:
        return self.iter().dotproduct(iterable)

    def flatten(self: CTuple[Iterable[T]]) -> CTuple[T]:
        return self.iter().flatten().list()

    @classmethod
    def repeatfunc(
        cls: Type[CTuple], func: Callable[..., T], times: Optional[int] = None, *args: Any,
    ) -> CTuple[T]:
        return CIterable.repeatfunc(func, times, *args).list()

    def pairwise(self: CTuple[T]) -> CTuple[Tuple[T, T]]:
        return self.iter().pairwise().list()

    def grouper(
        self: CTuple[T], n: int, fillvalue: Optional[T] = None,
    ) -> CTuple[Tuple[Union[T, U], ...]]:
        return self.iter().grouper(n, fillvalue=fillvalue).list()

    def partition(self: CTuple[T], func: Callable[[T], bool]) -> Tuple[CTuple[T], CTuple[T]]:
        return self.iter().partition(func).map(CTuple).tuple()

    def powerset(self: CTuple[T]) -> CTuple[Tuple[T, ...]]:
        return self.iter().powerset().list()

    def roundrobin(self: CTuple[T], *iterables: Iterable[U]) -> CTuple[Tuple[T, U]]:
        return self.iter().roundrobin(*iterables).list()

    def unique_everseen(self: CTuple[T], key: Optional[Callable[[T], Any]] = None) -> CTuple[T]:
        return self.iter().unique_everseen(key=key).list()

    def unique_justseen(self: CTuple[T], key: Optional[Callable[[T], Any]] = None) -> CTuple[T]:
        return self.iter().unique_justseen(key=key).list()

    @classmethod
    def iter_except(
        cls: Type[CTuple],
        func: Callable[..., T],
        exception: Type[Exception],
        first: Optional[Callable[..., U]] = None,
    ) -> CTuple[Union[T, U]]:
        return CIterable.iter_except(func, exception, first=first).list()

    def first_true(
        self: CTuple[T], default: U = False, pred: Optional[Callable[[T], Any]] = None,
    ) -> Union[T, U]:
        return self.iter().first_true(default=default, pred=pred).list()

    def random_product(
        self: CTuple[T], *iterables: Iterable[U], repeat: int = 1,
    ) -> Tuple[Union[T, U], ...]:
        return self.iter().random_product(*iterables, repeat=repeat)

    def random_permutation(self: CTuple[T], r: Optional[int] = None) -> Tuple[T, ...]:
        return self.iter().random_permutation(r=r)

    def random_combination(self: CTuple[T], r: int) -> Tuple[T, ...]:
        return self.iter().random_combination(r)

    def random_combination_with_replacement(self: CTuple[T], r: int) -> Tuple[T, ...]:
        return self.iter().random_combination_with_replacement(r)

    def nth_combination(self: CTuple[T], r: int, index: int) -> Tuple[T, ...]:
        return self.iter().nth_combination(r, index)

    # multiprocessing

    def pmap(
        self: CTuple[T], func: Callable[[T], U], *, processes: Optional[int] = None,
    ) -> CTuple[U]:
        warn(
            "'pmap' is going to be deprecated; use 'map(..., parallel=True)' instead",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.iter().pmap(func, processes=processes).tuple()

    def pstarmap(
        self: CTuple[Tuple[T, ...]],
        func: Callable[[Tuple[T, ...]], U],
        *,
        processes: Optional[int] = None,
    ) -> CTuple[U]:
        return self.iter().pstarmap(func, processes=processes).list()

    # pathlib

    @classmethod
    def iterdir(cls: Type[CTuple], path: Union[Path, str]) -> CTuple[Path]:
        return cls(CIterable.iterdir(path))

    # extra public

    def one(self: CTuple[T]) -> T:
        return self.iter().one()

    def pipe(
        self: CTuple[T],
        func: Callable[..., Iterable[U]],
        *args: Any,
        index: int = 0,
        **kwargs: Any,
    ) -> CTuple[U]:
        return self.iter().pipe(func, *args, index=index, **kwargs).list()

    def unzip(self: CTuple[Tuple[T, ...]]) -> Tuple[CTuple[T], ...]:
        return CTuple(self.iter().unzip()).map(CTuple)


class CSet(Set[T]):
    """A set with chainable methods."""

    # built-in

    def all(self: CSet[Any]) -> bool:  # noqa: A003
        return self.iter().all()

    def any(self: CSet[Any]) -> bool:  # noqa: A003
        return self.iter().any()

    def dict(self: CSet[Tuple[T, U]]) -> CDict[T, U]:  # noqa: A003
        return self.iter().dict()

    def enumerate(self: CSet[T], start: int = 0) -> CSet[Tuple[int, T]]:  # noqa: A003
        return self.iter().enumerate(start=start).set()

    def filter(self: CSet[T], func: Optional[Callable[[T], bool]]) -> CSet[T]:  # noqa: A003
        return self.iter().filter(func).set()

    def frozenset(self: CSet[T]) -> CFrozenSet[T]:  # noqa: A003
        return self.iter().frozenset()

    def iter(self: CSet[T]) -> CIterable[T]:  # noqa: A003
        return CIterable(self)

    def len(self: CTuple[T]) -> int:  # noqa: A003
        return len(self)

    def list(self: CSet[T]) -> CList[T]:  # noqa: A003
        return self.iter().list()

    def map(
        self: CSet[T],
        func: Callable[..., U],
        *iterables: Iterable,
        parallel: bool = False,
        processes: Optional[int] = None,
    ) -> CSet[U]:  # noqa: A003
        return self.iter().map(func, *iterables, parallel=parallel, processes=processes).set()

    def max(  # noqa: A003
        self: CSet[T],
        *,
        key: MAX_MIN_KEY_ANNOTATION = MAX_MIN_KEY_DEFAULT,
        default: Union[T, Sentinel] = sentinel,
    ) -> T:
        return self.iter().max(key=key, default=default)

    def min(  # noqa: A003
        self: CSet[T],
        *,
        key: MAX_MIN_KEY_ANNOTATION = MAX_MIN_KEY_DEFAULT,
        default: Union[T, Sentinel] = sentinel,
    ) -> T:
        return self.iter().min(key=key, default=default)

    @classmethod  # noqa: A003
    def range(  # noqa: A003
        cls: Type[CSet], start: int, stop: Optional[int] = None, step: Optional[int] = None,
    ) -> CSet[int]:
        return cls(CIterable.range(start, stop=stop, step=step))

    def set(self: CSet[T]) -> CSet[T]:  # noqa: A003
        return self.iter().set()

    def sorted(  # noqa: A003
        self: CSet[T], *, key: Optional[Callable[[T], Any]] = None, reverse: bool = False,
    ) -> CList[T]:
        return self.iter().sorted(key=key, reverse=reverse)

    def sum(self: CSet[T], start: Union[T, int] = 0) -> Union[T, int]:  # noqa: A003
        return self.iter().sum(start=start)

    def tuple(self: CSet[T]) -> CTuple[T]:  # noqa: A003
        return self.iter().tuple()

    def zip(self: CSet[T], *iterables: Iterable[U]) -> CList[CTuple[Union[T, U]]]:  # noqa: A003
        return self.iter().zip(*iterables).list()

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

    def reduce(
        self: CSet[T], func: Callable[[T, T], T], initial: Union[U, Sentinel] = sentinel,
    ) -> Any:
        return self.iter().reduce(func, initial=initial)

    # itertools

    @classmethod
    def repeat(cls: Type[CSet], x: T, times: int) -> CSet[T]:
        return cls(CIterable.repeat(x, times=times))

    def accumulate(
        self: CSet[T], func: Callable[[T, T], T] = add, *, initial: Union[U, Sentinel] = sentinel,
    ) -> CList[Union[T, U]]:
        return self.iter().accumulate(func, initial=initial).list()

    def chain(self: CSet[T], *iterables: Iterable[U]) -> CSet[Union[T, U]]:
        return self.iter().chain(*iterables).set()

    def compress(self: CSet[T], selectors: Iterable[Any]) -> CSet[T]:
        return self.iter().compress(selectors).set()

    def dropwhile(self: CSet[T], func: Callable[[T], bool]) -> CSet[T]:
        return self.iter().dropwhile(func).set()

    def filterfalse(self: CSet[T], func: Callable[[T], bool]) -> CSet[T]:
        return self.iter().filterfalse(func).set()

    def groupby(
        self: CSet[T], key: Optional[Callable[[T], U]] = None,
    ) -> CSet[Tuple[U, CFrozenSet[T]]]:
        return self.iter().groupby(key=key).map(lambda x: (x[0], CFrozenSet(x[1]))).set()

    def islice(
        self: CSet[T],
        start: int,
        stop: Union[int, Sentinel] = sentinel,
        step: Union[int, Sentinel] = sentinel,
    ) -> CSet[T]:
        return self.iter().islice(start, stop=stop, step=step).set()

    def starmap(self: CSet[Tuple[T, ...]], func: Callable[[Tuple[T, ...]], U]) -> CSet[U]:
        return self.iter().starmap(func).set()

    def takewhile(self: CSet[T], func: Callable[[T], bool]) -> CSet[T]:
        return self.iter().takewhile(func).set()

    def tee(self: CSet[T], n: int = 2) -> CSet[CFrozenSet[T]]:
        return self.iter().tee(n=n).set().map(CFrozenSet)

    def zip_longest(
        self: CSet[T], *iterables: Iterable[U], fillvalue: V = None,
    ) -> CSet[Tuple[Union[T, U, V]]]:
        return self.iter().zip_longest(*iterables, fillvalue=fillvalue).set()

    def product(
        self: CSet[T], *iterables: Iterable[U], repeat: int = 1,
    ) -> CSet[Tuple[Union[T, U], ...]]:
        return self.iter().product(*iterables, repeat=repeat).set()

    def permutations(self: CSet[T], r: Optional[int] = None) -> CSet[Tuple[T, ...]]:
        return self.iter().permutations(r=r).set()

    def combinations(self: CSet[T], r: int) -> CSet[Tuple[T, ...]]:
        return self.iter().combinations(r).set()

    def combinations_with_replacement(self: CSet[T], r: int) -> CSet[Tuple[T, ...]]:
        return self.iter().combinations_with_replacement(r).set()

    # itertools - recipes

    def take(self: CSet[T], n: int) -> CSet[T]:
        return self.iter().take(n).set()

    def prepend(self: CSet[T], value: U) -> CSet[Union[T, U]]:
        return self.iter().prepend(value).set()

    def tail(self: CSet[T], n: int) -> CSet[T]:
        return self.iter().tail(n).set()

    def consume(self: CSet[T], n: Optional[int] = None) -> CSet[T]:
        return self.iter().consume(n=n).set()

    def nth(self: CSet[T], n: int, default: U = None) -> Union[T, U]:
        return self.iter().nth(n, default=default)

    def all_equal(self: CSet[Any]) -> bool:
        return self.iter().all_equal()

    def quantify(self: CSet[T], pred: Callable[[T], bool] = bool) -> int:
        return self.iter().quantify(pred=pred)

    def ncycles(self: CSet[T], n: int) -> CSet[T]:
        return self.iter().ncycles(n).set()

    def dotproduct(self: CSet[T], iterable: Iterable[T]) -> T:
        return self.iter().dotproduct(iterable)

    def flatten(self: CSet[Iterable[T]]) -> CSet[T]:
        return self.iter().flatten().set()

    @classmethod
    def repeatfunc(
        cls: Type[CSet], func: Callable[..., T], times: Optional[int] = None, *args: Any,
    ) -> CSet[T]:
        return CIterable.repeatfunc(func, times, *args).set()

    def pairwise(self: CSet[T]) -> CSet[Tuple[T, T]]:
        return self.iter().pairwise().set()

    # multiprocessing

    def pmap(self: CSet[T], func: Callable[[T], U], *, processes: Optional[int] = None) -> CSet[U]:
        warn(
            "'pmap' is going to be deprecated; use 'map(..., parallel=True)' instead",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.iter().pmap(func, processes=processes).set()

    def pstarmap(
        self: CSet[Tuple[T, ...]],
        func: Callable[[Tuple[T, ...]], U],
        *,
        processes: Optional[int] = None,
    ) -> CSet[U]:
        return self.iter().pstarmap(func, processes=processes).set()

    # pathlib

    @classmethod
    def iterdir(cls: Type[CSet], path: Union[Path, str]) -> CSet[Path]:
        return cls(CIterable.iterdir(path))

    # extra public

    def one(self: CSet[T]) -> T:
        return self.iter().one()

    def pipe(
        self: CSet[T], func: Callable[..., Iterable[U]], *args: Any, index: int = 0, **kwargs: Any,
    ) -> CSet[U]:
        return self.iter().pipe(func, *args, index=index, **kwargs).set()


class CFrozenSet(FrozenSet[T]):
    """A frozenset with chainable methods."""

    # built-in

    def all(self: CFrozenSet[Any]) -> bool:  # noqa: A003
        return self.iter().all()

    def any(self: CFrozenSet[Any]) -> bool:  # noqa: A003
        return self.iter().any()

    def dict(self: CFrozenSet[Tuple[T, U]]) -> CDict[T, U]:  # noqa: A003
        return self.iter().dict()

    def enumerate(self: CFrozenSet[T], start: int = 0) -> CFrozenSet[Tuple[int, T]]:  # noqa: A003
        return self.iter().enumerate(start=start).frozenset()

    def filter(  # noqa: A003
        self: CFrozenSet[T], func: Optional[Callable[[T], bool]],
    ) -> CFrozenSet[T]:
        return self.iter().filter(func).frozenset()

    def frozenset(self: CFrozenSet[T]) -> CFrozenSet[T]:  # noqa: A003
        return self.iter().frozenset()

    def iter(self: CFrozenSet[T]) -> CIterable[T]:  # noqa: A003
        return CIterable(self)

    def len(self: CList[T]) -> int:  # noqa: A003
        return len(self)

    def list(self: CFrozenSet[T]) -> CList[T]:  # noqa: A003
        return self.iter().list()

    def map(  # noqa: A003
        self: CFrozenSet[T],
        func: Callable[..., U],
        *iterables: Iterable,
        parallel: bool = False,
        processes: Optional[int] = None,
    ) -> CFrozenSet[U]:
        return self.iter().map(func, *iterables, parallel=parallel, processes=processes).frozenset()

    def max(  # noqa: A003
        self: CFrozenSet[T],
        *,
        key: MAX_MIN_KEY_ANNOTATION = MAX_MIN_KEY_DEFAULT,
        default: Union[T, Sentinel] = sentinel,
    ) -> T:
        return self.iter().max(key=key, default=default)

    def min(  # noqa: A003
        self: CFrozenSet[T],
        *,
        key: MAX_MIN_KEY_ANNOTATION = MAX_MIN_KEY_DEFAULT,
        default: Union[T, Sentinel] = sentinel,
    ) -> T:
        return self.iter().min(key=key, default=default)

    @classmethod  # noqa: A003
    def range(  # noqa: A003
        cls: Type[CFrozenSet], start: int, stop: Optional[int] = None, step: Optional[int] = None,
    ) -> CFrozenSet[int]:
        """
        >>> CFrozenSet.range(5)
        CFrozenSet({0, 1, 2, 3, 4})
        >>> CFrozenSet.range(1, 5)
        CFrozenSet({1, 2, 3, 4})
        >>> CFrozenSet.range(1, 5, 2)
        CFrozenSet({1, 3})
        """
        return cls(CIterable.range(start, stop=stop, step=step))

    def set(self: CFrozenSet[T]) -> CSet[T]:  # noqa: A003
        return self.iter().set()

    def sorted(  # noqa: A003
        self: CFrozenSet[T], *, key: Optional[Callable[[T], Any]] = None, reverse: bool = False,
    ) -> CList[T]:
        return self.iter().sorted(key=key, reverse=reverse)

    def sum(self: CFrozenSet[T], start: Union[T, int] = 0) -> Union[T, int]:  # noqa: A003
        return self.iter().sum(start=start)

    def tuple(self: CFrozenSet[T]) -> CTuple[T]:  # noqa: A003
        return self.iter().tuple()

    def zip(  # noqa: A003
        self: CFrozenSet[T], *iterables: Iterable[U],
    ) -> CList[CTuple[Union[T, U]]]:
        return self.iter().zip(*iterables).list()

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

    def reduce(
        self: CFrozenSet[T], func: Callable[[T, T], T], initial: Union[U, Sentinel] = sentinel,
    ) -> Any:
        return self.iter().reduce(func, initial=initial)

    # itertools

    @classmethod
    def repeat(cls: Type[CFrozenSet], x: T, times: int) -> CFrozenSet[T]:
        return cls(CIterable.repeat(x, times=times))

    def accumulate(
        self: CFrozenSet[T],
        func: Callable[[T, T], T] = add,
        *,
        initial: Union[U, Sentinel] = sentinel,
    ) -> CList[Union[T, U]]:
        return self.iter().accumulate(func, initial=initial).list()

    def chain(self: CFrozenSet[T], *iterables: Iterable[U]) -> CFrozenSet[Union[T, U]]:
        return self.iter().chain(*iterables).frozenset()

    def compress(self: CFrozenSet[T], selectors: Iterable[Any]) -> CFrozenSet[T]:
        return self.iter().compress(selectors).frozenset()

    def dropwhile(self: CFrozenSet[T], func: Callable[[T], bool]) -> CFrozenSet[T]:
        return self.iter().dropwhile(func).frozenset()

    def filterfalse(self: CFrozenSet[T], func: Callable[[T], bool]) -> CFrozenSet[T]:
        return self.iter().filterfalse(func).frozenset()

    def groupby(
        self: CFrozenSet[T], key: Optional[Callable[[T], U]] = None,
    ) -> CFrozenSet[Tuple[U, CFrozenSet[T]]]:
        return self.iter().groupby(key=key).map(lambda x: (x[0], CFrozenSet(x[1]))).frozenset()

    def islice(
        self: CFrozenSet[T],
        start: int,
        stop: Union[int, Sentinel] = sentinel,
        step: Union[int, Sentinel] = sentinel,
    ) -> CFrozenSet[T]:
        return self.iter().islice(start, stop=stop, step=step).frozenset()

    def starmap(
        self: CFrozenSet[Tuple[T, ...]], func: Callable[[Tuple[T, ...]], U],
    ) -> CFrozenSet[U]:
        return self.iter().starmap(func).frozenset()

    def takewhile(self: CFrozenSet[T], func: Callable[[T], bool]) -> CFrozenSet[T]:
        return self.iter().takewhile(func).frozenset()

    def tee(self: CFrozenSet[T], n: int = 2) -> CFrozenSet[CFrozenSet[T]]:
        return self.iter().tee(n=n).frozenset().map(CFrozenSet)

    def zip_longest(
        self: CFrozenSet[T], *iterables: Iterable[U], fillvalue: V = None,
    ) -> CFrozenSet[Tuple[Union[T, U, V]]]:
        return self.iter().zip_longest(*iterables, fillvalue=fillvalue).frozenset()

    def product(
        self: CFrozenSet[T], *iterables: Iterable[U], repeat: int = 1,
    ) -> CFrozenSet[Tuple[Union[T, U], ...]]:
        return self.iter().product(*iterables, repeat=repeat).frozenset()

    def permutations(self: CFrozenSet[T], r: Optional[int] = None) -> CFrozenSet[Tuple[T, ...]]:
        return self.iter().permutations(r=r).frozenset()

    def combinations(self: CFrozenSet[T], r: int) -> CFrozenSet[Tuple[T, ...]]:
        return self.iter().combinations(r).frozenset()

    def combinations_with_replacement(self: CFrozenSet[T], r: int) -> CFrozenSet[Tuple[T, ...]]:
        return self.iter().combinations_with_replacement(r).frozenset()

    # itertools - recipes

    def take(self: CFrozenSet[T], n: int) -> CFrozenSet[T]:
        return self.iter().take(n).frozenset()

    def prepend(self: CFrozenSet[T], value: U) -> CFrozenSet[Union[T, U]]:
        return self.iter().prepend(value).frozenset()

    def tail(self: CFrozenSet[T], n: int) -> CFrozenSet[T]:
        return self.iter().tail(n).frozenset()

    def consume(self: CFrozenSet[T], n: Optional[int] = None) -> CFrozenSet[T]:
        return self.iter().consume(n=n).frozenset()

    def nth(self: CFrozenSet[T], n: int, default: U = None) -> Union[T, U]:
        return self.iter().nth(n, default=default)

    def all_equal(self: CFrozenSet[Any]) -> bool:
        return self.iter().all_equal()

    def quantify(self: CFrozenSet[T], pred: Callable[[T], bool] = bool) -> int:
        return self.iter().quantify(pred=pred)

    def ncycles(self: CFrozenSet[T], n: int) -> CFrozenSet[T]:
        return self.iter().ncycles(n).frozenset()

    def dotproduct(self: CFrozenSet[T], iterable: Iterable[T]) -> T:
        return self.iter().dotproduct(iterable)

    def flatten(self: CFrozenSet[Iterable[T]]) -> CFrozenSet[T]:
        return self.iter().flatten().frozenset()

    @classmethod
    def repeatfunc(
        cls: Type[CFrozenSet], func: Callable[..., T], times: Optional[int] = None, *args: Any,
    ) -> CFrozenSet[T]:
        return CIterable.repeatfunc(func, times, *args).frozenset()

    def pairwise(self: CFrozenSet[T]) -> CFrozenSet[Tuple[T, T]]:
        return self.iter().pairwise().frozenset()

    # multiprocessing

    def pmap(
        self: CFrozenSet[T], func: Callable[[T], U], *, processes: Optional[int] = None,
    ) -> CFrozenSet[U]:
        warn(
            "'pmap' is going to be deprecated; use 'map(..., parallel=True)' instead",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.iter().pmap(func, processes=processes).frozenset()

    def pstarmap(
        self: CFrozenSet[Tuple[T, ...]],
        func: Callable[[Tuple[T, ...]], U],
        *,
        processes: Optional[int] = None,
    ) -> CFrozenSet[U]:
        return self.iter().pstarmap(func, processes=processes).frozenset()

    # pathlib

    @classmethod
    def iterdir(cls: Type[CFrozenSet], path: Union[Path, str]) -> CFrozenSet[Path]:
        return cls(CIterable.iterdir(path))

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

    def map_keys(  # dead: disable
        self: CDict[T, U], func: Callable[[T], V], *, parallel: bool = False,
    ) -> CDict[V, U]:
        """Map a function of the form key_0 -> key_1 over the keys."""

        return self.items().map(partial(_apply_to_key, func=func), parallel=parallel).dict()

    def map_values(
        self: CDict[T, U], func: Callable[[U], V], *, parallel: bool = False,
    ) -> CDict[T, V]:
        """Map a function of the form value_0 -> value_1 over the values."""

        return self.items().map(partial(_apply_to_value, func=func), parallel=parallel).dict()

    def map_items(  # dead: disable
        self: CDict[T, U], func: Callable[[T, U], Tuple[V, W]], *, parallel: bool = False,
    ) -> CDict[V, W]:
        """Map a function of the form (key_0, value_0) -> (key_1, value_1) over the items."""

        return self.items().map(partial(_apply_to_item, func=func), parallel=parallel).dict()


def _apply_to_key(item: Tuple[T, U], *, func: Callable[[T], V]) -> Tuple[V, U]:
    key, value = item
    return func(key), value


def _apply_to_value(item: Tuple[T, U], *, func: Callable[[U], V]) -> Tuple[T, V]:
    key, value = item
    return key, func(value)


def _apply_to_item(item: Tuple[T, U], *, func: Callable[[T, U], Tuple[V, W]]) -> Tuple[V, W]:
    key, value = item
    return func(key, value)
