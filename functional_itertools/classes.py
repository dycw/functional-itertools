from __future__ import annotations

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
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NoReturn
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
from warnings import warn

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
from functional_itertools.utilities import drop_sentinel
from functional_itertools.utilities import Sentinel
from functional_itertools.utilities import sentinel
from functional_itertools.utilities import VERSION
from functional_itertools.utilities import Version


T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")


if VERSION is Version.py37:

    def _accumulate_citerable(
        self: CIterable[T], func: Callable[[T, U], U] = add,
    ) -> CIterable[Union[T, U]]:
        return CIterable(accumulate(self._iterable, func))

    def _accumulate_clist(self: CList[T], func: Callable[[T, U], U] = add) -> CList[Union[T, U]]:
        return self.iter().accumulate(func=func).list()

    def _accumulate_cset(self: CSet[T], func: Callable[[T, U], U] = add) -> CSet[Union[T, U]]:
        return self.iter().accumulate(func=func).set()

    def _accumulate_cfrozenset(
        self: CFrozenSet[T], func: Callable[[T, U], U] = add,
    ) -> CFrozenSet[Union[T, U]]:
        return self.iter().accumulate(func=func).frozenset()


elif VERSION is Version.py38:

    def _accumulate_citerable(
        self: CIterable[T], func: Callable[[T, U], U] = add, *, initial: Optional[U] = None,
    ) -> CIterable[Union[T, U]]:
        return CIterable(accumulate(self._iterable, func, initial=initial))

    def _accumulate_clist(
        self: CList[T], func: Callable[[T, U], U] = add, *, initial: Optional[U] = None,
    ) -> CList[Union[T, U]]:
        return self.iter().accumulate(func=func, initial=initial).list()

    def _accumulate_cset(
        self: CSet[T], func: Callable[[T, T], T] = add, *, initial: Optional[U] = None,
    ) -> CSet[T]:
        return self.iter().accumulate(func=func, initial=initial).set()

    def _accumulate_cfrozenset(
        self: CFrozenSet[T], func: Callable[[T, T], T] = add, *, initial: Optional[U] = None,
    ) -> CFrozenSet[T]:
        return self.iter().accumulate(func=func, initial=initial).frozenset()


else:
    raise UnsupportVersionError(VERSION)  # pragma: no cover


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
        return all(self._iterable)

    def any(self: CIterable[Any]) -> bool:  # noqa: A003
        return any(self._iterable)

    def dict(self: CIterable[Tuple[T, U]]) -> CDict[T, U]:  # noqa: A003
        return CDict(dict(self._iterable))

    def enumerate(self: CIterable[T], start: int = 0) -> CIterable[Tuple[int, T]]:  # noqa: A003
        return CIterable(enumerate(self._iterable, start=start))

    def filter(  # noqa: A003
        self: CIterable[T], func: Optional[Callable[[T], bool]],
    ) -> CIterable[T]:
        return CIterable(filter(func, self._iterable))

    def frozenset(self: CIterable[T]) -> CFrozenSet[T]:  # noqa: A003
        return CFrozenSet(self._iterable)

    def iter(self: CIterable[T]) -> CIterable[T]:  # noqa: A003
        return CIterable(self._iterable)

    def list(self: CIterable[T]) -> CList[T]:  # noqa: A003
        return CList(self._iterable)

    def map(  # noqa: A003
        self: CIterable[T], func: Callable[..., U], *iterables: Iterable,
    ) -> CIterable[U]:
        return CIterable(map(func, self._iterable, *iterables))

    def max(  # noqa: A003
        self: CIterable[T],
        *,
        key: MAX_MIN_KEY_ANNOTATION = MAX_MIN_KEY_DEFAULT,
        default: Union[T, Sentinel] = sentinel,
    ) -> T:
        _, kwargs = drop_sentinel(key=key, default=default)
        return max(self._iterable, **kwargs)

    def min(  # noqa: A003
        self: CIterable[T],
        *,
        key: MAX_MIN_KEY_ANNOTATION = MAX_MIN_KEY_DEFAULT,
        default: Union[T, Sentinel] = sentinel,
    ) -> T:
        _, kwargs = drop_sentinel(key=key, default=default)
        return min(self._iterable, **kwargs)

    @classmethod  # noqa: A003
    def range(  # noqa: A003
        cls: Type[CIterable],
        start: int,
        stop: Union[int, Sentinel] = sentinel,
        step: Union[int, Sentinel] = sentinel,
    ) -> CIterable[int]:
        args, _ = drop_sentinel(stop, step)
        return cls(range(start, *args))

    def set(self: CIterable[T]) -> CSet[T]:  # noqa: A003
        return CSet(self._iterable)

    def sorted(  # noqa: A003
        self: CIterable[T], *, key: Optional[Callable[[T], Any]] = None, reverse: bool = False,
    ) -> CList[T]:
        return CList(sorted(self._iterable, key=key, reverse=reverse))

    def sum(self: CIterable[T], start: Union[T, int] = 0) -> Union[T, int]:  # noqa: A003
        args, _ = drop_sentinel(start)
        return sum(self._iterable, *args)

    def tuple(self: CIterable[T]) -> Tuple[T, ...]:  # noqa: A003
        return tuple(self._iterable)

    def zip(  # noqa: A003
        self: CIterable[T], *iterables: Iterable[U],
    ) -> CIterable[Tuple[Union[T, U]]]:
        return CIterable(zip(self._iterable, *iterables))

    # functools

    def reduce(
        self: CIterable[T], func: Callable[[T, T], T], initial: Union[U, Sentinel] = sentinel,
    ) -> Any:
        args, _ = drop_sentinel(initial)
        try:
            result = reduce(func, self._iterable, *args)
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
        return CIterable(cycle(self._iterable))

    @classmethod
    def repeat(
        cls: Type[CIterable[T]], x: T, times: Union[int, Sentinel] = sentinel,
    ) -> CIterable[T]:
        args, _ = drop_sentinel(times)
        return cls(repeat(x, *args))

    accumulate = _accumulate_citerable

    def chain(self: CIterable[T], *iterables: Iterable[U]) -> CIterable[Union[T, U]]:
        return CIterable(chain(self._iterable, *iterables))

    def compress(self: CIterable[T], selectors: Iterable[Any]) -> CIterable[T]:
        return CIterable(compress(self._iterable, selectors))

    def dropwhile(self: CIterable[T], func: Callable[[T], bool]) -> CIterable[T]:
        return CIterable(dropwhile(func, self._iterable))

    def filterfalse(self: CIterable[T], func: Callable[[T], bool]) -> CIterable[T]:
        return CIterable(filterfalse(func, self._iterable))

    def groupby(
        self: CIterable[T], key: Optional[Callable[[T], U]] = None,
    ) -> CIterable[Tuple[U, CIterable[T]]]:
        def inner(x: Tuple[U, Iterator[T]]) -> Tuple[U, CIterable[T]]:
            key, group = x
            return key, CIterable(group)

        return CIterable(groupby(self._iterable, key=key)).map(inner)

    def islice(
        self: CIterable[T],
        start: int,
        stop: Union[int, Sentinel] = sentinel,
        step: Union[int, Sentinel] = sentinel,
    ) -> CIterable[T]:
        args, _ = drop_sentinel(stop, step)
        return CIterable(islice(self._iterable, start, *args))

    def starmap(
        self: CIterable[Tuple[T, ...]], func: Callable[[Tuple[T, ...]], U],
    ) -> CIterable[U]:
        return CIterable(starmap(func, self._iterable))

    def takewhile(self: CIterable[T], func: Callable[[T], bool]) -> CIterable[T]:
        return CIterable(takewhile(func, self._iterable))

    def tee(self: CIterable[T], n: int = 2) -> CIterable[Iterator[T]]:
        return CIterable(tee(self._iterable, n)).map(CIterable)

    def zip_longest(
        self: CIterable[T], *iterables: Iterable[U], fillvalue: V = None,
    ) -> CIterable[Tuple[Union[T, U, V]]]:
        return CIterable(zip_longest(self._iterable, *iterables, fillvalue=fillvalue))

    def product(
        self: CIterable[T], *iterables: Iterable[U], repeat: int = 1,
    ) -> CIterable[Tuple[Union[T, U], ...]]:
        return CIterable(product(self._iterable, *iterables, repeat=repeat))

    def permutations(self: CIterable[T], r: Optional[int] = None) -> CIterable[Tuple[T, ...]]:
        return CIterable(permutations(self._iterable, r=r))

    def combinations(self: CIterable[T], r: int) -> CIterable[Tuple[T, ...]]:
        return CIterable(combinations(self._iterable, r))

    def combinations_with_replacement(self: CIterable[T], r: int) -> CIterable[Tuple[T, ...]]:
        return CIterable(combinations_with_replacement(self._iterable, r))

    # itertools-recipes

    def take(self: CIterable[T], n: int) -> CIterable[T]:
        return CIterable(take(n, self._iterable))

    def prepend(self: CIterable[T], value: U) -> CIterable[Union[T, U]]:
        return CIterable(prepend(value, self._iterable))

    @classmethod
    def tabulate(cls: Type[CIterable], func: Callable[[int], T], start: int = 0) -> CIterable[T]:
        return cls(tabulate(func, start=start))

    def tail(self: CIterable[T], n: int) -> CIterable[T]:
        return CIterable(tail(n, self._iterable))

    def consume(self: CIterable[T], n: Optional[int] = None) -> CIterable[T]:
        consume(self._iterable, n=n)
        return self

    def nth(self: CIterable[T], n: int, default: U = None) -> Union[T, U]:
        return nth(self._iterable, n, default=default)

    def all_equal(self: CIterable[Any]) -> bool:
        return all_equal(self._iterable)

    def quantify(self: CIterable[T], pred: Callable[[T], bool] = bool) -> int:
        return quantify(self._iterable, pred=pred)

    def padnone(self: CIterable[T]) -> CIterable[Optional[T]]:
        return CIterable(padnone(self._iterable))

    def ncycles(self: CIterable[T], n: int) -> CIterable[T]:
        return CIterable(ncycles(self._iterable, n))

    def dotproduct(self: CIterable[T], iterable: Iterable[T]) -> T:
        return dotproduct(self._iterable, iterable)

    def flatten(self: CIterable[Iterable[T]]) -> CIterable[T]:
        return CIterable(flatten(self._iterable))

    @classmethod
    def repeatfunc(
        cls: Type[CIterable], func: Callable[..., T], times: Optional[int] = None, *args: Any,
    ) -> CIterable[T]:
        return cls(repeatfunc(func, times=times, *args))

    def pairwise(self: CIterable[T]) -> CIterable[Tuple[T, T]]:
        return CIterable(pairwise(self._iterable))

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

    # multiprocessing

    def pmap(
        self: CIterable[T], func: Callable[[T], U], *, processes: Optional[int] = None,
    ) -> CIterable[U]:
        try:
            with Pool(processes=processes) as pool:
                return CIterable(pool.map(func, self._iterable))
        except AssertionError as error:
            (msg,) = error.args
            if msg == "daemonic processes are not allowed to have children":
                return self.map(func)
            else:
                raise NotImplementedError(msg)

    def pstarmap(
        self: CIterable[Tuple[T, ...]],
        func: Callable[[Tuple[T, ...]], U],
        *,
        processes: Optional[int] = None,
    ) -> CIterable[U]:
        with Pool(processes=processes) as pool:
            return CIterable(pool.starmap(func, self._iterable))

    # pathlib

    @classmethod
    def iterdir(cls: Type[CIterable], path: Union[Path, str]) -> CIterable[Path]:
        return cls(Path(path).iterdir())

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

    def list(self: CFrozenSet[T]) -> CList[T]:  # noqa: A003
        return self.iter().list()

    def map(self: CList[T], func: Callable[..., U], *iterables: Iterable) -> CList[U]:  # noqa: A003
        return self.iter().map(func, *iterables).list()

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
        cls: Type[CList],
        start: int,
        stop: Union[int, Sentinel] = sentinel,
        step: Union[int, Sentinel] = sentinel,
    ) -> CList[int]:
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

    def tuple(self: CList[T]) -> Tuple[T, ...]:  # noqa: A003
        return self.iter().tuple()

    def zip(self: CList[T], *iterables: Iterable[U]) -> CList[Tuple[Union[T, U]]]:  # noqa: A003
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

    accumulate = _accumulate_clist

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

    def starmap(self: CList[Tuple[T, ...]], func: Callable[[Tuple[T, ...]], U]) -> CList[U]:
        return self.iter().starmap(func).list()

    def takewhile(self: CList[T], func: Callable[[T], bool]) -> CList[T]:
        return self.iter().takewhile(func).list()

    def tee(self: CList[T], n: int = 2) -> CList[CList[T]]:
        return self.iter().tee(n=n).list().map(CList)

    def zip_longest(
        self: CList[T], *iterables: List, fillvalue: Any = None,
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
    def repeatfunc(cls: Type[CList], func: Callable[..., T], times: int, *args: Any) -> CList[T]:
        return CIterable.repeatfunc(func, times=times, *args).list()

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

    def list(self: CSet[T]) -> CList[T]:  # noqa: A003
        return self.iter().list()

    def map(self: CSet[T], func: Callable[..., U], *iterables: Iterable) -> CSet[U]:  # noqa: A003
        return self.iter().map(func, *iterables).set()

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
        cls: Type[CSet],
        start: int,
        stop: Union[int, Sentinel] = sentinel,
        step: Union[int, Sentinel] = sentinel,
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

    def tuple(self: CSet[T]) -> Tuple[T, ...]:  # noqa: A003
        return self.iter().tuple()

    def zip(self: CSet[T], *iterables: Iterable[U]) -> CSet[Tuple[Union[T, U]]]:  # noqa: A003
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

    def update(self: CSet[T]) -> NoReturn:  # dead: disable
        raise RuntimeError("Use the 'union' method instead of 'update'")

    def intersection_update(self: CSet[T]) -> NoReturn:  # dead: disable
        raise RuntimeError("Use the 'intersection' method instead of 'intersection_update'")

    def difference_update(self: CSet[T]) -> NoReturn:  # dead: disable
        raise RuntimeError("Use the 'difference' method instead of 'difference_update'")

    def symmetric_difference_update(self: CSet[T]) -> NoReturn:  # dead: disable
        raise RuntimeError(
            "Use the 'symmetric_difference' method instead of 'symmetric_difference_update'",
        )

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

    accumulate = _accumulate_cset

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

    def starmap(self: CSet[Tuple[T, ...]], func: Callable[[Tuple[T, ...]], U]) -> CSet[U]:
        return self.iter().starmap(func).set()

    def takewhile(self: CSet[T], func: Callable[[T], bool]) -> CSet[T]:
        return self.iter().takewhile(func).set()

    def tee(self: CSet[T], n: int = 2) -> CSet[CFrozenSet[T]]:
        return self.iter().tee(n=n).set().map(CFrozenSet)

    # itertools - recipes

    def take(self: CSet[T], n: int) -> CSet[T]:
        return self.iter().take(n).set()

    # multiprocessing

    def pmap(self: CSet[T], func: Callable[[T], U], *, processes: Optional[int] = None) -> CSet[U]:
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

    def list(self: CFrozenSet[T]) -> CList[T]:  # noqa: A003
        return self.iter().list()

    def map(  # noqa: A003
        self: CFrozenSet[T], func: Callable[..., U], *iterables: Iterable,
    ) -> CFrozenSet[U]:
        return self.iter().map(func, *iterables).frozenset()

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
        cls: Type[CFrozenSet],
        start: int,
        stop: Union[int, Sentinel] = sentinel,
        step: Union[int, Sentinel] = sentinel,
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

    def tuple(self: CFrozenSet[T]) -> Tuple[T, ...]:  # noqa: A003
        return self.iter().tuple()

    def zip(  # noqa: A003
        self: CFrozenSet[T], *iterables: Iterable[U],
    ) -> CFrozenSet[Tuple[Union[T, U]]]:
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

    @classmethod
    def repeat(cls: Type[CFrozenSet], x: T, times: int) -> CFrozenSet[T]:
        return cls(CIterable.repeat(x, times=times))

    accumulate = _accumulate_cfrozenset

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

    def starmap(
        self: CFrozenSet[Tuple[T, ...]], func: Callable[[Tuple[T, ...]], U],
    ) -> CFrozenSet[U]:
        return self.iter().starmap(func).frozenset()

    def takewhile(self: CFrozenSet[T], func: Callable[[T], bool]) -> CFrozenSet[T]:
        return self.iter().takewhile(func).frozenset()

    def tee(self: CFrozenSet[T], n: int = 2) -> CFrozenSet[CFrozenSet[T]]:
        return self.iter().tee(n=n).frozenset().map(CFrozenSet)

    # itertools - recipes

    def take(self: CFrozenSet[T], n: int) -> CFrozenSet[T]:
        return self.iter().take(n).frozenset()

    # multiprocessing

    def pmap(
        self: CFrozenSet[T], func: Callable[[T], U], *, processes: Optional[int] = None,
    ) -> CFrozenSet[U]:
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

    # built-in

    def all_keys(self: CDict[Any, Any]) -> bool:  # dead: disable
        return self.keys().all()

    def all_values(self: CDict[Any, Any]) -> bool:  # dead: disable
        return self.values().all()

    def all_items(self: CDict[Any, Any]) -> bool:  # dead: disable
        return self.items().all()

    def any_keys(self: CDict[Any, Any]) -> bool:  # dead: disable
        return self.keys().any()

    def any_values(self: CDict[Any, Any]) -> bool:  # dead: disable
        return self.values().any()

    def any_items(self: CDict[Any, Any]) -> bool:  # dead: disable
        return self.items().any()

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

    def frozenset_keys(self: CDict[T, Any]) -> CFrozenSet[T]:  # dead: disable
        return self.keys().frozenset()

    def frozenset_values(self: CDict[T, Any]) -> CFrozenSet[U]:  # dead: disable
        return self.values().frozenset()

    def frozenset_items(self: CDict[T, Any]) -> CFrozenSet[Tuple[T, U]]:  # dead: disable
        return self.items().frozenset()

    def list_keys(self: CDict[T, Any]) -> CList[T]:  # dead: disable
        return self.keys().list()

    def list_values(self: CDict[Any, U]) -> CList[U]:  # dead: disable
        return self.values().list()

    def list_items(self: CDict[T, U]) -> CList[Tuple[T, U]]:  # dead: disable
        return self.items().list()

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

    def max_keys(  # dead: disable
        self: CDict[T, U],
        *,
        key: MAX_MIN_KEY_ANNOTATION = MAX_MIN_KEY_DEFAULT,
        default: Union[T, Sentinel] = sentinel,
    ) -> T:
        return self.keys().max(key=key, default=default)

    def max_values(  # dead: disable
        self: CDict[T, U],
        *,
        key: MAX_MIN_KEY_ANNOTATION = MAX_MIN_KEY_DEFAULT,
        default: Union[U, Sentinel] = sentinel,
    ) -> U:
        return self.values().max(key=key, default=default)

    def max_items(  # dead: disable
        self: CDict[T, U],
        *,
        key: MAX_MIN_KEY_ANNOTATION = MAX_MIN_KEY_DEFAULT,
        default: Union[T, Sentinel] = sentinel,
    ) -> T:
        _, kwargs = drop_sentinel(key=key, default=default)
        return max(self.items(), **kwargs)

    def set_keys(self: CDict[T, U]) -> CSet[T]:  # dead: disable
        return self.keys().set()

    def set_values(self: CDict[T, U]) -> CSet[U]:  # dead: disable
        return self.values().set()

    def set_items(self: CDict[T, U]) -> CSet[Tuple[T, U]]:  # dead: disable
        return self.items().set()
