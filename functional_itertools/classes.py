from __future__ import annotations

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
from sys import maxsize
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

from more_itertools import chunked
from more_itertools import distribute
from more_itertools import divide
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
from functional_itertools.errors import UnsupportVersionError
from functional_itertools.utilities import drop_none
from functional_itertools.utilities import drop_sentinel
from functional_itertools.utilities import help_filter_items
from functional_itertools.utilities import help_filter_keys
from functional_itertools.utilities import help_filter_values
from functional_itertools.utilities import help_last
from functional_itertools.utilities import help_map_items
from functional_itertools.utilities import help_map_keys
from functional_itertools.utilities import help_map_values
from functional_itertools.utilities import Sentinel
from functional_itertools.utilities import sentinel
from functional_itertools.utilities import suppress_daemonic_processes_with_children
from functional_itertools.utilities import VERSION
from functional_itertools.utilities import Version
from functional_itertools.utilities import warn_non_functional


T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")


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

    # built-in

    def all(self: CIterable) -> bool:  # noqa: A003
        return all(self)

    def any(self: CIterable) -> bool:  # noqa: A003
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
                    with suppress_daemonic_processes_with_children(error):
                        return self.map(func)
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
        if (stop is None) and (step is not None):
            raise ValueError("'stop' cannot be None if 'step' is provided")
        else:
            args, _ = drop_none(stop, step)
            return cls(range(start, *args))

    def set(self: CIterable[T]) -> CSet[T]:  # noqa: A003
        return CSet(self)

    def sorted(  # noqa: A003
        self: CIterable[T], *, key: Optional[Callable[[T], Any]] = None, reverse: bool = False,
    ) -> CList[T]:
        return CList(sorted(self, key=key, reverse=reverse))

    def sum(self: CIterable[T], start: Union[T, int] = 0) -> Union[T, int]:  # noqa: A003
        args, _ = drop_sentinel(start)
        return sum(self, *args)

    def tuple(self: CIterable[T]) -> CTuple[T]:  # noqa: A003
        return CTuple(self)

    def zip(  # noqa: A003
        self: CIterable[T], *iterables: Iterable[U],
    ) -> CIterable[CTuple[Union[T, U]]]:
        return CIterable(zip(self, *iterables)).map(CTuple)

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

    # itertools

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
            _, kwargs = drop_sentinel(initial=initial)
            return CIterable(accumulate(self, func, **kwargs))
        else:
            raise UnsupportVersionError(VERSION)  # pragma: no cover

    def chain(self: CIterable[T], *iterables: Iterable[U]) -> CIterable[Union[T, U]]:
        return CIterable(chain(self, *iterables))

    def combinations(self: CIterable[T], r: int) -> CIterable[CTuple[T]]:
        return CIterable(combinations(self, r)).map(CTuple)

    def combinations_with_replacement(self: CIterable[T], r: int) -> CIterable[CTuple[T]]:
        return CIterable(combinations_with_replacement(self, r)).map(CTuple)

    def compress(self: CIterable[T], selectors: Iterable) -> CIterable[T]:
        return CIterable(compress(self, selectors))

    @classmethod
    def count(cls: Type[CIterable], start: int = 0, step: int = 1) -> CIterable[int]:
        return cls(count(start=start, step=step))

    def cycle(self: CIterable[T]) -> CIterable[T]:
        return CIterable(cycle(self))

    def dropwhile(self: CIterable[T], func: Callable[[T], bool]) -> CIterable[T]:
        return CIterable(dropwhile(func, self))

    def filterfalse(self: CIterable[T], func: Callable[[T], bool]) -> CIterable[T]:
        return CIterable(filterfalse(func, self))

    def groupby(
        self: CIterable[T], key: Optional[Callable[[T], U]] = None,
    ) -> CIterable[Tuple[U, CTuple[T]]]:
        return CIterable(groupby(self, key=key)).map(_help_groupby)

    def islice(
        self: CIterable[T], start: int, stop: Optional[int] = None, step: Optional[int] = None,
    ) -> CIterable[T]:
        args, _ = drop_none(stop, step)
        return CIterable(islice(self, start, *args))

    def permutations(self: CIterable[T], r: Optional[int] = None) -> CIterable[CTuple[T]]:
        return CIterable(permutations(self, r=r)).map(CTuple)

    def product(
        self: CIterable[T], *iterables: Iterable[U], repeat: int = 1,
    ) -> CIterable[CTuple[Union[T, U]]]:
        return CIterable(product(self, *iterables, repeat=repeat)).map(CTuple)

    @classmethod
    def repeat(cls: Type[CIterable], x: T, times: Optional[int] = None) -> CIterable[T]:
        args, _ = drop_none(times)
        return cls(repeat(x, *args))

    def starmap(
        self: CIterable[Tuple[T, ...]],
        func: Callable[[Tuple[T, ...]], U],
        *,
        parallel: bool = False,
        processes: Optional[int] = None,
    ) -> CIterable[U]:
        if parallel:
            try:
                with Pool(processes=processes) as pool:
                    return CIterable(pool.starmap(func, self))
            except AssertionError as error:
                with suppress_daemonic_processes_with_children(error):
                    return self.starmap(func)
        else:
            return CIterable(starmap(func, self))

    def takewhile(self: CIterable[T], func: Callable[[T], bool]) -> CIterable[T]:
        return CIterable(takewhile(func, self))

    def tee(self: CIterable[T], n: int = 2) -> CIterable[CIterable[T]]:
        return CIterable(tee(self, n)).map(CIterable)

    def zip_longest(
        self: CIterable[T], *iterables: Iterable[U], fillvalue: V = None,
    ) -> CIterable[CTuple[Union[T, U, V]]]:
        return CIterable(zip_longest(self, *iterables, fillvalue=fillvalue)).map(CTuple)

    # itertools-recipes

    def all_equal(self: CIterable) -> bool:
        return all_equal(self)

    def consume(self: CIterable[T], n: Optional[int] = None) -> CIterable[T]:
        iterator = iter(self)
        consume(iterator, n=n)
        return CIterable(iterator)

    def dotproduct(self: CIterable[T], iterable: Iterable[T]) -> T:
        return dotproduct(self, iterable)

    def first_true(
        self: CIterable[T], default: U = False, pred: Optional[Callable[[T], Any]] = None,
    ) -> Union[T, U]:
        return first_true(self, default=default, pred=pred)

    def flatten(self: CIterable[Iterable[T]]) -> CIterable[T]:
        return CIterable(flatten(self))

    def grouper(self: CIterable[T], n: int, fillvalue: U = None) -> CIterable[CTuple[Union[T, U]]]:
        return CIterable(grouper(self, n, fillvalue=fillvalue)).map(CTuple)

    @classmethod
    def iter_except(
        cls: Type[CIterable],
        func: Callable[..., T],
        exception: Type[Exception],
        first: Optional[Callable[..., U]] = None,
    ) -> CIterable[Union[T, U]]:
        return cls(iter_except(func, exception, first=first))

    def ncycles(self: CIterable[T], n: int) -> CIterable[T]:
        return CIterable(ncycles(self, n))

    def nth(self: CIterable[T], n: int, default: U = None) -> Union[T, U]:
        return nth(self, n, default=default)

    def nth_combination(self: CIterable[T], r: int, index: int) -> CTuple[T]:
        return CTuple(nth_combination(self, r, index))

    def padnone(self: CIterable[T]) -> CIterable[Optional[T]]:
        return CIterable(padnone(self))

    def pairwise(self: CIterable[T]) -> CIterable[CTuple[T]]:
        return CIterable(pairwise(self)).map(CTuple)

    def partition(self: CIterable[T], func: Callable[[T], bool]) -> CTuple[CIterable[T]]:
        return CIterable(partition(func, self)).map(CIterable).tuple()

    def powerset(self: CIterable[T]) -> CIterable[CTuple[T]]:
        return CIterable(powerset(self)).map(CTuple)

    def prepend(self: CIterable[T], value: U) -> CIterable[Union[T, U]]:
        return CIterable(prepend(value, self))

    def quantify(self: CIterable[T], pred: Callable[[T], bool] = bool) -> int:
        return quantify(self, pred=pred)

    def random_combination(self: CIterable[T], r: int) -> CTuple[T]:
        return CTuple(random_combination(self, r))

    def random_combination_with_replacement(self: CIterable[T], r: int) -> CTuple[T]:
        return CTuple(random_combination_with_replacement(self, r))

    def random_permutation(self: CIterable[T], r: Optional[int] = None) -> CTuple[T]:
        return CTuple(random_permutation(self, r=r))

    def random_product(
        self: CIterable[T], *iterables: Iterable[U], repeat: int = 1,
    ) -> CTuple[Union[T, U]]:
        return CTuple(random_product(self, *iterables, repeat=repeat))

    @classmethod
    def repeatfunc(
        cls: Type[CIterable], func: Callable[..., T], times: Optional[int] = None, *args: Any,
    ) -> CIterable[T]:
        return cls(repeatfunc(func, times, *args))

    def roundrobin(self: CIterable[T], *iterables: Iterable[U]) -> CIterable[Union[T, U]]:
        return CIterable(roundrobin(self, *iterables))

    @classmethod
    def tabulate(cls: Type[CIterable], func: Callable[[int], T], start: int = 0) -> CIterable[T]:
        return cls(tabulate(func, start=start))

    def tail(self: CIterable[T], n: int) -> CIterable[T]:
        return CIterable(tail(n, self))

    def take(self: CIterable[T], n: int) -> CIterable[T]:
        return CIterable(take(n, self))

    def unique_everseen(
        self: CIterable[T], key: Optional[Callable[[T], Any]] = None,
    ) -> CIterable[T]:
        return CIterable(unique_everseen(self, key=key))

    def unique_justseen(
        self: CIterable[T], key: Optional[Callable[[T], Any]] = None,
    ) -> CIterable[T]:
        return CIterable(unique_justseen(self, key=key))

    # more-itertools

    def chunked(self: CIterable[T], n: int) -> CIterable[CTuple[T]]:
        return CIterable(chunked(self, n)).map(CTuple)

    def distribute(self: CIterable[T], n: int) -> CIterable[CTuple[T]]:
        return CIterable(distribute(n, self)).map(CTuple)

    def divide(self: CIterable[T], n: int) -> CIterable[CTuple[T]]:
        return CIterable(divide(n, list(self))).map(CTuple)

    # multiprocessing

    def pmap(  # dead: disable
        self: CIterable[T], func: Callable[[T], U], *, processes: Optional[int] = None,
    ) -> CIterable[U]:
        warn(
            "'pmap' is going to be deprecated; use 'map(..., parallel=True)' instead",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.map(func, parallel=True, processes=processes)

    def pstarmap(  # dead: disable
        self: CIterable[Tuple[T, ...]],
        func: Callable[[Tuple[T, ...]], U],
        *,
        processes: Optional[int] = None,
    ) -> CIterable[U]:
        warn(
            "'pstarmap' is going to be deprecated; use 'starmap(..., parallel=True)' instead",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.starmap(func, parallel=True, processes=processes)

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
        return self.reduce(help_last)

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

    def map(  # noqa: A003
        self: CList[T],
        func: Callable[..., U],
        *iterables: Iterable,
        parallel: bool = False,
        processes: Optional[int] = None,
    ) -> CList[U]:
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
        return cls(CIterable.range(start, stop=stop, step=step))

    def reversed(self: CList[T]) -> CList[T]:  # noqa: A003
        return CList(reversed(self))

    def set(self: CList[T]) -> CSet[T]:  # noqa: A003
        return self.iter().set()

    def sort(
        self: CList[T], *, key: Optional[Callable[[T], Any]] = None, reverse: bool = False,
    ) -> CList[T]:
        warn_non_functional(CList, "sort", "sorted")
        super().sort(key=key, reverse=reverse)

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

    def accumulate(
        self: CList[T], func: Callable[[T, T], T] = add, *, initial: Union[U, Sentinel] = sentinel,
    ) -> CList[Union[T, U]]:
        return self.iter().accumulate(func, initial=initial).list()

    def chain(self: CList[T], *iterables: Iterable[U]) -> CList[Union[T, U]]:
        return self.iter().chain(*iterables).list()

    def combinations(self: CList[T], r: int) -> CList[CTuple[T]]:
        return self.iter().combinations(r).list()

    def combinations_with_replacement(self: CList[T], r: int) -> CList[CTuple[T]]:
        return self.iter().combinations_with_replacement(r).list()

    def compress(self: CList[T], selectors: Iterable) -> CList[T]:
        return self.iter().compress(selectors).list()

    def dropwhile(self: CList[T], func: Callable[[T], bool]) -> CList[T]:
        return self.iter().dropwhile(func).list()

    def filterfalse(self: CList[T], func: Callable[[T], bool]) -> CList[T]:
        return self.iter().filterfalse(func).list()

    def groupby(
        self: CList[T], key: Optional[Callable[[T], U]] = None,
    ) -> CList[Tuple[U, CTuple[T]]]:
        return self.iter().groupby(key=key).list()

    def islice(
        self: CList[T], start: int, stop: Optional[int] = None, step: Optional[int] = None,
    ) -> CIterable[T]:
        return self.iter().islice(start, stop=stop, step=step)

    def permutations(self: CList[T], r: Optional[int] = None) -> CList[CTuple[T]]:
        return self.iter().permutations(r=r).list()

    def product(
        self: CList[T], *iterables: Iterable[U], repeat: int = 1,
    ) -> CList[CTuple[Union[T, U]]]:
        return self.iter().product(*iterables, repeat=repeat).list()

    @classmethod
    def repeat(cls: Type[CList], x: T, times: int) -> CList[T]:
        return cls(CIterable.repeat(x, times=times))

    def starmap(
        self: CList[Tuple[T, ...]],
        func: Callable[[Tuple[T, ...]], U],
        *,
        parallel: bool = False,
        processes: Optional[int] = None,
    ) -> CList[U]:
        return self.iter().starmap(func, parallel=parallel, processes=processes).list()

    def takewhile(self: CList[T], func: Callable[[T], bool]) -> CList[T]:
        return self.iter().takewhile(func).list()

    def tee(self: CList[T], n: int = 2) -> CIterable[CIterable[T]]:
        return self.iter().tee(n=n)

    def zip_longest(
        self: CList[T], *iterables: Iterable[U], fillvalue: V = None,
    ) -> CList[CTuple[Union[T, U, V]]]:
        return self.iter().zip_longest(*iterables, fillvalue=fillvalue).list()

    # itertools-recipes

    def all_equal(self: CList[Any]) -> bool:
        return self.iter().all_equal()

    def consume(self: CList[T], n: Optional[int] = None) -> CList[T]:
        return self.iter().consume(n=n).list()

    def dotproduct(self: CList[T], iterable: Iterable[T]) -> T:
        return self.iter().dotproduct(iterable)

    def first_true(
        self: CList[T], default: U = False, pred: Optional[Callable[[T], Any]] = None,
    ) -> Union[T, U]:
        return self.iter().first_true(default=default, pred=pred)

    def flatten(self: CList[Iterable[T]]) -> CList[T]:
        return self.iter().flatten().list()

    def grouper(
        self: CList[T], n: int, fillvalue: Optional[T] = None,
    ) -> CList[CTuple[Union[T, U]]]:
        return self.iter().grouper(n, fillvalue=fillvalue).list()

    @classmethod
    def iter_except(
        cls: Type[CList],
        func: Callable[..., T],
        exception: Type[Exception],
        first: Optional[Callable[..., U]] = None,
    ) -> CList[Union[T, U]]:
        return cls(CIterable.iter_except(func, exception, first=first))

    def ncycles(self: CList[T], n: int) -> CList[T]:
        return self.iter().ncycles(n).list()

    def nth(self: CList[T], n: int, default: U = None) -> Union[T, U]:
        return self.iter().nth(n, default=default)

    def nth_combination(self: CList[T], r: int, index: int) -> CTuple[T]:
        return self.iter().nth_combination(r, index)

    def padnone(self: CList[T]) -> CIterable[Optional[T]]:
        return self.iter().padnone()

    def pairwise(self: CList[T]) -> CList[CTuple[T]]:
        return self.iter().pairwise().list()

    def partition(self: CList[T], func: Callable[[T], bool]) -> CTuple[CList[T]]:
        return self.iter().partition(func).map(CList)

    def powerset(self: CList[T]) -> CList[CTuple[T]]:
        return self.iter().powerset().list()

    def prepend(self: CList[T], value: U) -> CList[Union[T, U]]:
        return self.iter().prepend(value).list()

    def quantify(self: CList[T], pred: Callable[[T], bool] = bool) -> int:
        return self.iter().quantify(pred=pred)

    def random_combination(self: CList[T], r: int) -> CTuple[T]:
        return self.iter().random_combination(r)

    def random_combination_with_replacement(self: CList[T], r: int) -> CTuple[T]:
        return self.iter().random_combination_with_replacement(r)

    def random_permutation(self: CList[T], r: Optional[int] = None) -> CTuple[T]:
        return self.iter().random_permutation(r=r)

    def random_product(
        self: CList[T], *iterables: Iterable[U], repeat: int = 1,
    ) -> CTuple[Union[T, U]]:
        return self.iter().random_product(*iterables, repeat=repeat)

    @classmethod
    def repeatfunc(
        cls: Type[CList], func: Callable[..., T], times: Optional[int] = None, *args: Any,
    ) -> CList[T]:
        return CIterable.repeatfunc(func, times, *args).list()

    def roundrobin(self: CList[T], *iterables: Iterable[U]) -> CList[Union[T, U]]:
        return self.iter().roundrobin(*iterables).list()

    def tail(self: CList[T], n: int) -> CList[T]:
        return self.iter().tail(n).list()

    def take(self: CList[T], n: int) -> CList[T]:
        return self.iter().take(n).list()

    def unique_everseen(self: CList[T], key: Optional[Callable[[T], Any]] = None) -> CList[T]:
        return self.iter().unique_everseen(key=key).list()

    def unique_justseen(self: CList[T], key: Optional[Callable[[T], Any]] = None) -> CList[T]:
        return self.iter().unique_justseen(key=key).list()

    # more-itertools

    def chunked(self: CList[T], n: int) -> CList[CTuple[T]]:
        return self.iter().chunked(n).list()

    def distribute(self: CList[T], n: int) -> CList[CTuple[T]]:
        return self.iter().distribute(n).list()

    def divide(self: CList[T], n: int) -> CList[CTuple[T]]:
        return self.iter().divide(n).list()

    # multiprocessing

    def pmap(  # dead: disable
        self: CList[T], func: Callable[[T], U], *, processes: Optional[int] = None,
    ) -> CList[U]:
        warn(
            "'pmap' is going to be deprecated; use 'map(..., parallel=True)' instead",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.map(func, parallel=True, processes=processes)

    def pstarmap(  # dead: disable
        self: CList[Tuple[T, ...]],
        func: Callable[[Tuple[T, ...]], U],
        *,
        processes: Optional[int] = None,
    ) -> CList[U]:
        warn(
            "'pstarmap' is going to be deprecated; use 'starmap(..., parallel=True)' instead",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.starmap(func, parallel=True, processes=processes)

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

    def sorted(  # noqa: A003
        self: CTuple[T], *, key: Optional[Callable[[T], Any]] = None, reverse: bool = False,
    ) -> CTuple[T]:
        return self.iter().sorted(key=key, reverse=reverse).tuple()

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

    def accumulate(
        self: CTuple[T], func: Callable[[T, T], T] = add, *, initial: Union[U, Sentinel] = sentinel,
    ) -> CTuple[Union[T, U]]:
        return self.iter().accumulate(func, initial=initial).tuple()

    def chain(self: CTuple[T], *iterables: Iterable[U]) -> CTuple[Union[T, U]]:
        return self.iter().chain(*iterables).tuple()

    def combinations(self: CTuple[T], r: int) -> CTuple[CTuple[T]]:
        return self.iter().combinations(r).tuple()

    def combinations_with_replacement(self: CTuple[T], r: int) -> CTuple[CTuple[T]]:
        return self.iter().combinations_with_replacement(r).tuple()

    def compress(self: CTuple[T], selectors: Iterable) -> CTuple[T]:
        return self.iter().compress(selectors).tuple()

    def dropwhile(self: CTuple[T], func: Callable[[T], bool]) -> CTuple[T]:
        return self.iter().dropwhile(func).tuple()

    def filterfalse(self: CTuple[T], func: Callable[[T], bool]) -> CTuple[T]:
        return self.iter().filterfalse(func).tuple()

    def groupby(
        self: CTuple[T], key: Optional[Callable[[T], U]] = None,
    ) -> CTuple[Tuple[U, CTuple[T]]]:
        return self.iter().groupby(key=key).tuple()

    def islice(
        self: CTuple[T], start: int, stop: Optional[int] = None, step: Optional[int] = None,
    ) -> CIterable[T]:
        return self.iter().islice(start, stop=stop, step=step)

    def permutations(self: CTuple[T], r: Optional[int] = None) -> CTuple[CTuple[T]]:
        return self.iter().permutations(r=r).tuple()

    def product(
        self: CTuple[T], *iterables: Iterable[U], repeat: int = 1,
    ) -> CTuple[CTuple[Union[T, U]]]:
        return self.iter().product(*iterables, repeat=repeat).tuple()

    @classmethod
    def repeat(cls: Type[CTuple], x: T, times: int) -> CTuple[T]:
        return cls(CIterable.repeat(x, times=times))

    def starmap(
        self: CTuple[Tuple[T, ...]],
        func: Callable[[Tuple[T, ...]], U],
        *,
        parallel: bool = False,
        processes: Optional[int] = None,
    ) -> CTuple[U]:
        return self.iter().starmap(func, parallel=parallel, processes=processes).tuple()

    def takewhile(self: CTuple[T], func: Callable[[T], bool]) -> CTuple[T]:
        return self.iter().takewhile(func).tuple()

    def tee(self: CTuple[T], n: int = 2) -> CIterable[CIterable[T]]:
        return self.iter().tee(n=n)

    def zip_longest(
        self: CTuple[T], *iterables: Iterable[U], fillvalue: V = None,
    ) -> CTuple[CTuple[Union[T, U, V]]]:
        return self.iter().zip_longest(*iterables, fillvalue=fillvalue).tuple()

    # itertools-recipes

    def all_equal(self: CTuple[Any]) -> bool:
        return self.iter().all_equal()

    def consume(self: CTuple[T], n: Optional[int] = None) -> CTuple[T]:
        return self.iter().consume(n=n).tuple()

    def dotproduct(self: CTuple[T], iterable: Iterable[T]) -> T:
        return self.iter().dotproduct(iterable)

    def first_true(
        self: CTuple[T], default: U = False, pred: Optional[Callable[[T], Any]] = None,
    ) -> Union[T, U]:
        return self.iter().first_true(default=default, pred=pred)

    def flatten(self: CTuple[Iterable[T]]) -> CTuple[T]:
        return self.iter().flatten().tuple()

    def grouper(
        self: CTuple[T], n: int, fillvalue: Optional[T] = None,
    ) -> CTuple[CTuple[Union[T, U]]]:
        return self.iter().grouper(n, fillvalue=fillvalue).map(CTuple).tuple()

    @classmethod
    def iter_except(
        cls: Type[CTuple],
        func: Callable[..., T],
        exception: Type[Exception],
        first: Optional[Callable[..., U]] = None,
    ) -> CTuple[Union[T, U]]:
        return cls(CIterable.iter_except(func, exception, first=first))

    def ncycles(self: CTuple[T], n: int) -> CTuple[T]:
        return self.iter().ncycles(n).tuple()

    def nth(self: CTuple[T], n: int, default: U = None) -> Union[T, U]:
        return self.iter().nth(n, default=default)

    def nth_combination(self: CTuple[T], r: int, index: int) -> CTuple[T]:
        return self.iter().nth_combination(r, index)

    def padnone(self: CFrozenSet[T]) -> CIterable[Optional[T]]:
        return self.iter().padnone()

    def pairwise(self: CTuple[T]) -> CTuple[CTuple[T]]:
        return self.iter().pairwise().tuple()

    def partition(self: CTuple[T], func: Callable[[T], bool]) -> CTuple[CTuple[T]]:
        return self.iter().partition(func).map(CTuple)

    def powerset(self: CTuple[T]) -> CTuple[CTuple[T]]:
        return self.iter().powerset().tuple()

    def prepend(self: CTuple[T], value: U) -> CTuple[Union[T, U]]:
        return self.iter().prepend(value).tuple()

    def quantify(self: CTuple[T], pred: Callable[[T], bool] = bool) -> int:
        return self.iter().quantify(pred=pred)

    def random_combination(self: CTuple[T], r: int) -> CTuple[T]:
        return self.iter().random_combination(r)

    def random_combination_with_replacement(self: CTuple[T], r: int) -> CTuple[T]:
        return self.iter().random_combination_with_replacement(r)

    def random_permutation(self: CTuple[T], r: Optional[int] = None) -> CTuple[T]:
        return self.iter().random_permutation(r=r)

    def random_product(
        self: CTuple[T], *iterables: Iterable[U], repeat: int = 1,
    ) -> CTuple[Union[T, U]]:
        return self.iter().random_product(*iterables, repeat=repeat)

    @classmethod
    def repeatfunc(
        cls: Type[CTuple], func: Callable[..., T], times: Optional[int] = None, *args: Any,
    ) -> CTuple[T]:
        return CIterable.repeatfunc(func, times, *args).tuple()

    def roundrobin(self: CTuple[T], *iterables: Iterable[U]) -> CTuple[Union[T, U]]:
        return self.iter().roundrobin(*iterables).tuple()

    def tail(self: CTuple[T], n: int) -> CTuple[T]:
        return self.iter().tail(n).tuple()

    def take(self: CTuple[T], n: int) -> CTuple[T]:
        return self.iter().take(n).tuple()

    def unique_everseen(self: CTuple[T], key: Optional[Callable[[T], Any]] = None) -> CTuple[T]:
        return self.iter().unique_everseen(key=key).tuple()

    def unique_justseen(self: CTuple[T], key: Optional[Callable[[T], Any]] = None) -> CTuple[T]:
        return self.iter().unique_justseen(key=key).tuple()

    # more-itertools

    def chunked(self: CTuple[T], n: int) -> CTuple[CTuple[T]]:
        return self.iter().chunked(n).tuple()

    def distribute(self: CTuple[T], n: int) -> CTuple[CTuple[T]]:
        return self.iter().distribute(n).tuple()

    def divide(self: CTuple[T], n: int) -> CTuple[CTuple[T]]:
        return self.iter().divide(n).tuple()

    # multiprocessing

    def pmap(  # dead: disable
        self: CTuple[T], func: Callable[[T], U], *, processes: Optional[int] = None,
    ) -> CTuple[U]:
        warn(
            "'pmap' is going to be deprecated; use 'map(..., parallel=True)' instead",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.map(func, parallel=True, processes=processes)

    def pstarmap(  # dead: disable
        self: CTuple[Tuple[T, ...]],
        func: Callable[[Tuple[T, ...]], U],
        *,
        processes: Optional[int] = None,
    ) -> CTuple[U]:
        return self.starmap(func, parallel=True, processes=processes)

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

    def map(  # noqa: A003
        self: CSet[T],
        func: Callable[..., U],
        *iterables: Iterable,
        parallel: bool = False,
        processes: Optional[int] = None,
    ) -> CSet[U]:
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

    def zip(self: CSet[T], *iterables: Iterable[U]) -> CSet[CTuple[Union[T, U]]]:  # noqa: A003
        return self.iter().zip(*iterables).set()

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

    def accumulate(
        self: CSet[T], func: Callable[[T, T], T] = add, *, initial: Union[U, Sentinel] = sentinel,
    ) -> CSet[Union[T, U]]:
        return self.iter().accumulate(func, initial=initial).set()

    def chain(self: CSet[T], *iterables: Iterable[U]) -> CSet[Union[T, U]]:
        return self.iter().chain(*iterables).set()

    def combinations(self: CSet[T], r: int) -> CSet[CTuple[T]]:
        return self.iter().combinations(r).set()

    def combinations_with_replacement(self: CSet[T], r: int) -> CSet[CTuple[T]]:
        return self.iter().combinations_with_replacement(r).set()

    def compress(self: CSet[T], selectors: Iterable) -> CSet[T]:
        return self.iter().compress(selectors).set()

    def dropwhile(self: CSet[T], func: Callable[[T], bool]) -> CSet[T]:
        return self.iter().dropwhile(func).set()

    def filterfalse(self: CSet[T], func: Callable[[T], bool]) -> CSet[T]:
        return self.iter().filterfalse(func).set()

    def groupby(
        self: CSet[T], key: Optional[Callable[[T], U]] = None,
    ) -> CSet[Tuple[U, CTuple[T]]]:
        return self.iter().groupby(key=key).set()

    def islice(
        self: CSet[T], start: int, stop: Optional[int] = None, step: Optional[int] = None,
    ) -> CIterable[T]:
        return self.iter().islice(start, stop=stop, step=step)

    def permutations(self: CSet[T], r: Optional[int] = None) -> CSet[CTuple[T]]:
        return self.iter().permutations(r=r).set()

    def product(
        self: CSet[T], *iterables: Iterable[U], repeat: int = 1,
    ) -> CSet[CTuple[Union[T, U]]]:
        return self.iter().product(*iterables, repeat=repeat).set()

    @classmethod
    def repeat(cls: Type[CSet], x: T, times: int) -> CSet[T]:
        return cls(CIterable.repeat(x, times=times))

    def starmap(
        self: CSet[Tuple[T, ...]],
        func: Callable[[Tuple[T, ...]], U],
        *,
        parallel: bool = False,
        processes: Optional[int] = None,
    ) -> CSet[U]:
        return self.iter().starmap(func, parallel=parallel, processes=processes).set()

    def takewhile(self: CSet[T], func: Callable[[T], bool]) -> CSet[T]:
        return self.iter().takewhile(func).set()

    def tee(self: CSet[T], n: int = 2) -> CIterable[CIterable[T]]:
        return self.iter().tee(n=n)

    def zip_longest(
        self: CSet[T], *iterables: Iterable[U], fillvalue: V = None,
    ) -> CSet[CTuple[Union[T, U, V]]]:
        return self.iter().zip_longest(*iterables, fillvalue=fillvalue).set()

    # itertools - recipes

    def all_equal(self: CSet[Any]) -> bool:
        return self.iter().all_equal()

    def consume(self: CSet[T], n: Optional[int] = None) -> CSet[T]:
        return self.iter().consume(n=n).set()

    def dotproduct(self: CSet[T], iterable: Iterable[T]) -> T:
        return self.iter().dotproduct(iterable)

    def first_true(
        self: CSet[T], default: U = False, pred: Optional[Callable[[T], Any]] = None,
    ) -> Union[T, U]:
        return self.iter().first_true(default=default, pred=pred)

    def flatten(self: CSet[Iterable[T]]) -> CSet[T]:
        return self.iter().flatten().set()

    def grouper(self: CSet[T], n: int, fillvalue: Optional[T] = None) -> CSet[CTuple[Union[T, U]]]:
        return self.iter().grouper(n, fillvalue=fillvalue).set()

    @classmethod
    def iter_except(
        cls: Type[CSet],
        func: Callable[..., T],
        exception: Type[Exception],
        first: Optional[Callable[..., U]] = None,
    ) -> CSet[Union[T, U]]:
        return cls(CIterable.iter_except(func, exception, first=first))

    def ncycles(self: CSet[T], n: int) -> CSet[T]:
        return self.iter().ncycles(n).set()

    def nth(self: CSet[T], n: int, default: U = None) -> Union[T, U]:
        return self.iter().nth(n, default=default)

    def nth_combination(self: CSet[T], r: int, index: int) -> CTuple[T]:
        return self.iter().nth_combination(r, index)

    def padnone(self: CSet[T]) -> CIterable[Optional[T]]:
        return self.iter().padnone()

    def pairwise(self: CSet[T]) -> CSet[CTuple[T]]:
        return self.iter().pairwise().set()

    def partition(self: CSet[T], func: Callable[[T], bool]) -> CTuple[CSet[T]]:
        return self.iter().partition(func).map(CSet)

    def powerset(self: CSet[T]) -> CSet[CTuple[T]]:
        return self.iter().powerset().set()

    def prepend(self: CSet[T], value: U) -> CSet[Union[T, U]]:
        return self.iter().prepend(value).set()

    def quantify(self: CSet[T], pred: Callable[[T], bool] = bool) -> int:
        return self.iter().quantify(pred=pred)

    def random_combination(self: CSet[T], r: int) -> CTuple[T]:
        return self.iter().random_combination(r)

    def random_combination_with_replacement(self: CSet[T], r: int) -> CTuple[T]:
        return self.iter().random_combination_with_replacement(r)

    def random_permutation(self: CSet[T], r: Optional[int] = None) -> CTuple[T]:
        return self.iter().random_permutation(r=r)

    def random_product(
        self: CSet[T], *iterables: Iterable[U], repeat: int = 1,
    ) -> CTuple[Union[T, U]]:
        return self.iter().random_product(*iterables, repeat=repeat)

    @classmethod
    def repeatfunc(
        cls: Type[CSet], func: Callable[..., T], times: Optional[int] = None, *args: Any,
    ) -> CSet[T]:
        return CIterable.repeatfunc(func, times, *args).set()

    def roundrobin(self: CSet[T], *iterables: Iterable[U]) -> CSet[Union[T, U]]:
        return self.iter().roundrobin(*iterables).set()

    def tail(self: CSet[T], n: int) -> CSet[T]:
        return self.iter().tail(n).set()

    def take(self: CSet[T], n: int) -> CSet[T]:
        return self.iter().take(n).set()

    def unique_everseen(self: CSet[T], key: Optional[Callable[[T], Any]] = None) -> CSet[T]:
        return self.iter().unique_everseen(key=key).set()

    def unique_justseen(self: CSet[T], key: Optional[Callable[[T], Any]] = None) -> CSet[T]:
        return self.iter().unique_justseen(key=key).set()

    # more-itertools

    def chunked(self: CSet[T], n: int) -> CSet[CTuple[T]]:
        return self.iter().chunked(n).set()

    def distribute(self: CSet[T], n: int) -> CSet[CTuple[T]]:
        return self.iter().distribute(n).set()

    def divide(self: CSet[T], n: int) -> CSet[CTuple[T]]:
        return self.iter().divide(n).set()

    # multiprocessing

    def pmap(  # dead: disable
        self: CSet[T], func: Callable[[T], U], *, processes: Optional[int] = None,
    ) -> CSet[U]:
        warn(
            "'pmap' is going to be deprecated; use 'map(..., parallel=True)' instead",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.map(func, parallel=True, processes=processes)

    def pstarmap(  # dead: disable
        self: CSet[Tuple[T, ...]],
        func: Callable[[Tuple[T, ...]], U],
        *,
        processes: Optional[int] = None,
    ) -> CSet[U]:
        warn(
            "'pstarmap' is going to be deprecated; use 'starmap(..., parallel=True)' instead",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.starmap(func, parallel=True, processes=processes)

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
    ) -> CFrozenSet[CTuple[Union[T, U]]]:
        return self.iter().zip(*iterables).frozenset()

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

    def accumulate(
        self: CFrozenSet[T],
        func: Callable[[T, T], T] = add,
        *,
        initial: Union[U, Sentinel] = sentinel,
    ) -> CFrozenSet[Union[T, U]]:
        return self.iter().accumulate(func, initial=initial).frozenset()

    def chain(self: CFrozenSet[T], *iterables: Iterable[U]) -> CFrozenSet[Union[T, U]]:
        return self.iter().chain(*iterables).frozenset()

    def combinations(self: CFrozenSet[T], r: int) -> CFrozenSet[CTuple[T]]:
        return self.iter().combinations(r).frozenset()

    def combinations_with_replacement(self: CFrozenSet[T], r: int) -> CFrozenSet[CTuple[T]]:
        return self.iter().combinations_with_replacement(r).frozenset()

    def compress(self: CFrozenSet[T], selectors: Iterable) -> CFrozenSet[T]:
        return self.iter().compress(selectors).frozenset()

    def dropwhile(self: CFrozenSet[T], func: Callable[[T], bool]) -> CFrozenSet[T]:
        return self.iter().dropwhile(func).frozenset()

    def filterfalse(self: CFrozenSet[T], func: Callable[[T], bool]) -> CFrozenSet[T]:
        return self.iter().filterfalse(func).frozenset()

    def groupby(
        self: CFrozenSet[T], key: Optional[Callable[[T], U]] = None,
    ) -> CFrozenSet[Tuple[U, CTuple[T]]]:
        return self.iter().groupby(key=key).frozenset()

    def islice(
        self: CFrozenSet[T], start: int, stop: Optional[int] = None, step: Optional[int] = None,
    ) -> CIterable[T]:
        return self.iter().islice(start, stop=stop, step=step)

    def permutations(self: CFrozenSet[T], r: Optional[int] = None) -> CFrozenSet[CTuple[T]]:
        return self.iter().permutations(r=r).frozenset()

    def product(
        self: CFrozenSet[T], *iterables: Iterable[U], repeat: int = 1,
    ) -> CFrozenSet[CTuple[Union[T, U]]]:
        return self.iter().product(*iterables, repeat=repeat).frozenset()

    @classmethod
    def repeat(cls: Type[CFrozenSet], x: T, times: int) -> CFrozenSet[T]:
        return cls(CIterable.repeat(x, times=times))

    def starmap(
        self: CFrozenSet[Tuple[T, ...]],
        func: Callable[[Tuple[T, ...]], U],
        *,
        parallel: bool = False,
        processes: Optional[int] = None,
    ) -> CFrozenSet[U]:
        return self.iter().starmap(func, parallel=parallel, processes=processes).frozenset()

    def takewhile(self: CFrozenSet[T], func: Callable[[T], bool]) -> CFrozenSet[T]:
        return self.iter().takewhile(func).frozenset()

    def tee(self: CFrozenSet[T], n: int = 2) -> CIterable[CIterable[T]]:
        return self.iter().tee(n=n)

    def zip_longest(
        self: CFrozenSet[T], *iterables: Iterable[U], fillvalue: V = None,
    ) -> CFrozenSet[CTuple[Union[T, U, V]]]:
        return self.iter().zip_longest(*iterables, fillvalue=fillvalue).frozenset()

    # itertools - recipes
    def all_equal(self: CFrozenSet[Any]) -> bool:
        return self.iter().all_equal()

    def consume(self: CFrozenSet[T], n: Optional[int] = None) -> CFrozenSet[T]:
        return self.iter().consume(n=n).frozenset()

    def dotproduct(self: CFrozenSet[T], iterable: Iterable[T]) -> T:
        return self.iter().dotproduct(iterable)

    def first_true(
        self: CFrozenSet[T], default: U = False, pred: Optional[Callable[[T], Any]] = None,
    ) -> Union[T, U]:
        return self.iter().first_true(default=default, pred=pred)

    def flatten(self: CFrozenSet[Iterable[T]]) -> CFrozenSet[T]:
        return self.iter().flatten().frozenset()

    def grouper(
        self: CFrozenSet[T], n: int, fillvalue: Optional[T] = None,
    ) -> CFrozenSet[CTuple[Union[T, U]]]:
        return self.iter().grouper(n, fillvalue=fillvalue).frozenset()

    @classmethod
    def iter_except(
        cls: Type[CFrozenSet],
        func: Callable[..., T],
        exception: Type[Exception],
        first: Optional[Callable[..., U]] = None,
    ) -> CFrozenSet[Union[T, U]]:
        return cls(CIterable.iter_except(func, exception, first=first))

    def ncycles(self: CFrozenSet[T], n: int) -> CFrozenSet[T]:
        return self.iter().ncycles(n).frozenset()

    def nth(self: CFrozenSet[T], n: int, default: U = None) -> Union[T, U]:
        return self.iter().nth(n, default=default)

    def nth_combination(self: CFrozenSet[T], r: int, index: int) -> CTuple[T]:
        return self.iter().nth_combination(r, index)

    def padnone(self: CFrozenSet[T]) -> CIterable[Optional[T]]:
        return self.iter().padnone()

    def pairwise(self: CFrozenSet[T]) -> CFrozenSet[CTuple[T]]:
        return self.iter().pairwise().frozenset()

    def partition(self: CFrozenSet[T], func: Callable[[T], bool]) -> CTuple[CFrozenSet[T]]:
        return self.iter().partition(func).map(CFrozenSet)

    def powerset(self: CFrozenSet[T]) -> CFrozenSet[CTuple[T]]:
        return self.iter().powerset().frozenset()

    def prepend(self: CFrozenSet[T], value: U) -> CFrozenSet[Union[T, U]]:
        return self.iter().prepend(value).frozenset()

    def quantify(self: CFrozenSet[T], pred: Callable[[T], bool] = bool) -> int:
        return self.iter().quantify(pred=pred)

    def random_combination(self: CFrozenSet[T], r: int) -> CTuple[T]:
        return self.iter().random_combination(r)

    def random_combination_with_replacement(self: CFrozenSet[T], r: int) -> CTuple[T]:
        return self.iter().random_combination_with_replacement(r)

    def random_permutation(self: CFrozenSet[T], r: Optional[int] = None) -> CTuple[T]:
        return self.iter().random_permutation(r=r)

    def random_product(
        self: CFrozenSet[T], *iterables: Iterable[U], repeat: int = 1,
    ) -> CTuple[Union[T, U]]:
        return self.iter().random_product(*iterables, repeat=repeat)

    @classmethod
    def repeatfunc(
        cls: Type[CFrozenSet], func: Callable[..., T], times: Optional[int] = None, *args: Any,
    ) -> CFrozenSet[T]:
        return CIterable.repeatfunc(func, times, *args).frozenset()

    def roundrobin(self: CFrozenSet[T], *iterables: Iterable[U]) -> CFrozenSet[Union[T, U]]:
        return self.iter().roundrobin(*iterables).frozenset()

    def tail(self: CFrozenSet[T], n: int) -> CFrozenSet[T]:
        return self.iter().tail(n).frozenset()

    def take(self: CFrozenSet[T], n: int) -> CFrozenSet[T]:
        return self.iter().take(n).frozenset()

    def unique_everseen(
        self: CFrozenSet[T], key: Optional[Callable[[T], Any]] = None,
    ) -> CFrozenSet[T]:
        return self.iter().unique_everseen(key=key).frozenset()

    def unique_justseen(
        self: CFrozenSet[T], key: Optional[Callable[[T], Any]] = None,
    ) -> CFrozenSet[T]:
        return self.iter().unique_justseen(key=key).frozenset()

    # more-itertools

    def chunked(self: CFrozenSet[T], n: int) -> CFrozenSet[CTuple[T]]:
        return self.iter().chunked(n).frozenset()

    def distribute(self: CFrozenSet[T], n: int) -> CFrozenSet[CTuple[T]]:
        return self.iter().distribute(n).frozenset()

    def divide(self: CFrozenSet[T], n: int) -> CFrozenSet[CTuple[T]]:
        return self.iter().divide(n).frozenset()

    # multiprocessing

    def pmap(  # dead: disable
        self: CFrozenSet[T], func: Callable[[T], U], *, processes: Optional[int] = None,
    ) -> CFrozenSet[U]:
        warn(
            "'pmap' is going to be deprecated; use 'map(..., parallel=True)' instead",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.map(func, parallel=True, processes=processes)

    def pstarmap(  # dead: disable
        self: CFrozenSet[Tuple[T, ...]],
        func: Callable[[Tuple[T, ...]], U],
        *,
        processes: Optional[int] = None,
    ) -> CFrozenSet[U]:
        warn(
            "'pstarmap' is going to be deprecated; use 'starmap(..., parallel=True)' instead",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.starmap(func, parallel=True, processes=processes)

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
        return self.items().filter(partial(help_filter_keys, func=func)).dict()

    def filter_values(self: CDict[T, U], func: Callable[[U], bool]) -> CDict[T, U]:  # dead: disable
        return self.items().filter(partial(help_filter_values, func=func)).dict()

    def filter_items(  # dead: disable
        self: CDict[T, U], func: Callable[[T, U], bool],
    ) -> CDict[T, U]:
        return self.items().filter(partial(help_filter_items, func=func)).dict()

    def map_keys(  # dead: disable
        self: CDict[T, U],
        func: Callable[[T], V],
        *,
        parallel: bool = False,
        processes: Optional[int] = None,
    ) -> CDict[V, U]:
        """Map a function of the form key_0 -> key_1 over the keys."""

        return (
            self.items()
            .map(partial(help_map_keys, func=func), parallel=parallel, processes=processes)
            .dict()
        )

    def map_values(
        self: CDict[T, U],
        func: Callable[[U], V],
        *,
        parallel: bool = False,
        processes: Optional[int] = None,
    ) -> CDict[T, V]:
        """Map a function of the form value_0 -> value_1 over the values."""

        return (
            self.items()
            .map(partial(help_map_values, func=func), parallel=parallel, processes=processes)
            .dict()
        )

    def map_items(  # dead: disable
        self: CDict[T, U],
        func: Callable[[T, U], Tuple[V, W]],
        *,
        parallel: bool = False,
        processes: Optional[int] = None,
    ) -> CDict[V, W]:
        """Map a function of the form (key_0, value_0) -> (key_1, value_1) over the items."""

        return (
            self.items()
            .map(partial(help_map_items, func=func), parallel=parallel, processes=processes)
            .dict()
        )


# helpers


def _help_groupby(pair: Tuple[T, Iterable[T]]) -> Tuple[T, CTuple[T]]:
    key, group = pair
    return key, CTuple(group)
