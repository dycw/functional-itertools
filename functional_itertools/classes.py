from __future__ import annotations

import builtins
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

from functional_itertools.errors import EmptyIterableError
from functional_itertools.errors import MultipleElementsError
from functional_itertools.errors import StopArgumentMissing
from functional_itertools.errors import UnsupportVersionError
from functional_itertools.methods.base import MethodBuilder
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


def defines_method_factory(doc: str) -> Callable[[str], FunctionType]:
    def decorator(factory):
        def wrapped(name: str) -> FunctionType:
            method = factory(name)
            method.__annotations__ = {
                k: v.strip("'").format(name=name) for k, v in method.__annotations__.items()
            }
            method.__doc__ = doc.format(name=name)
            return method

        return wrapped

    return decorator


@defines_method_factory(
    "Return `True` if all elements of the {name} are true (or if the {name} is empty)."
)
def _build_all(name: str) -> Callable[..., Any]:
    def all(self: "{name}[T]") -> bool:  # noqa: A003
        return builtins.all(self)

    return all


@defines_method_factory(
    "Return `True` if all elements of the {name} are true (or if the {name} is empty)."
)
def _build_any(name: str) -> Callable[..., bool]:
    def any(self: "{name}[T]") -> bool:  # noqa: A003
        return builtins.any(self)

    return any


class DictMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: DictMethodBuilder) -> Callable[..., CDict]:
        def method(self: Template[Tuple[T, U]]) -> CDict[T, U]:
            return CDict(self)

        return method

    _doc = "Create a new CDict from the {0}."


class EnumerateMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[EnumerateMethodBuilder]) -> Callable[..., Any]:
        def method(self: Template[T], start: int = 0) -> Template[CTuple[Union[int, T]]]:
            return type(self)(map(CTuple, enumerate(self, start=start)))

        return method

    _doc = "Return an enumerate object, cast as a {0}."


class FilterMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[FilterMethodBuilder]) -> Callable[..., Any]:
        def method(self: Template[T], func: Optional[Callable[[T], bool]]) -> Template[T]:
            return type(self)(filter(func, self))

        return method

    _doc = "Construct a {0} from those elements of the {0} for which function returns true."


class IterMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: IterMethodBuilder) -> Callable[..., CIterable]:
        def method(self: Template[T]) -> CIterable[T]:
            return CIterable(self)

        return method

    _doc = "Create a new CDict from the {0}."


class FrozenSetMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: FrozenSetMethodBuilder) -> Callable[..., CFrozenSet]:
        def method(self: Template[T]) -> CFrozenSet[T]:
            return CFrozenSet(self)

        return method

    _doc = "Create a new CFrozenSet from the {0}."


class LenMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[LenMethodBuilder]) -> Callable[..., int]:
        def method(self: Template[T]) -> int:
            return len(self)

        return method

    _doc = "Return the length of the {0}."


class ListMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: ListMethodBuilder) -> Callable[..., CList]:
        def method(self: Template[T]) -> CList[T]:
            return CList(self)

        return method

    _doc = "Create a new CList from the {0}."


class MapMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[MapMethodBuilder]) -> Callable[..., Any]:
        def method(self: Template[T], func: Callable[..., U], *iterables: Iterable) -> Template[U]:
            return type(self)(map(func, self, *iterables))

        return method

    _doc = "Construct a {0} by applying `func` to every item of the {0}."


class MaxMinMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(
        cls: Type[MaxMinMethodBuilder], func: Callable[..., Any],
    ) -> Callable[..., Any]:
        if VERSION is Version.py37:

            def method(
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

            def method(
                self: Template[T],
                *,
                key: Optional[Callable[[T], Any]] = None,
                default: U = sentinel,
            ) -> Union[T, U]:
                return func(self, key=key, **({} if default is sentinel else {"default": default}))

        else:
            raise UnsupportVersionError(VERSION)  # pragma: no cover

        return method

    _doc = "Return the maximum/minimum over the {0}."


class RangeMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[RangeMethodBuilder]) -> Callable[..., Any]:
        def method(
            cls: Type[Template[T]],
            start: int,
            stop: Optional[int] = None,
            step: Optional[int] = None,
        ) -> Template[int]:
            if (stop is None) and (step is not None):
                raise StopArgumentMissing()
            else:
                return cls(
                    range(
                        start,
                        *(() if stop is None else (stop,)),
                        *(() if step is None else (step,)),
                    ),
                )

        return method

    _doc = "Return a range of integers as a {0}."


class SetMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: SetMethodBuilder) -> Callable[..., CSet]:
        def method(self: Template[T]) -> CSet[T]:
            return CSet(self)

        return method

    _doc = "Create a new CSet from the {0}."


class SortedMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: MethodBuilder) -> Callable[..., Any]:
        def method(
            self: Template[T], *, key: Optional[Callable[[T], Any]] = None, reverse: bool = False,
        ) -> CList[T]:
            return CList(sorted(self, key=key, reverse=reverse))

        return method

    _doc = "Return a sorted CList from the items in the {0}."


class SumMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[SumMethodBuilder]) -> Callable[..., int]:
        def method(self: Template[T], start: Union[U, Sentinel] = sentinel) -> Union[T, U]:
            return sum(self, *(() if start is sentinel else (start,)))

        return method

    _doc = "Return the sum of the elements in {0}."


class TupleMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: TupleMethodBuilder) -> Callable[..., CTuple]:
        def method(self: Template[T]) -> CTuple[T]:
            return CTuple(self)

        return method

    _doc = "Create a new CTuple from the {0}."


class ZipMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[ZipMethodBuilder]) -> Callable[..., int]:
        def method(self: Template[T], *iterables: Iterable[U]) -> Template[CTuple[Union[T, U]]]:
            return type(self)(map(CTuple, zip(self, *iterables)))

        return method

    _doc = "Return an iterator that aggregates elements from the {0} and the input iterables."


# itertools


class AccumulateMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[AccumulateMethodBuilder]) -> Callable[..., Any]:
        if VERSION is Version.py37:

            def method(self: Template[T], func: Callable[[T, T], T] = add) -> Template[T]:
                return type(self)(accumulate(self, func))

        elif VERSION is Version.py38:

            def method(
                self: Template[T],
                func: Callable[[Union[T, U], Union[T, U]], Union[T, U]] = add,
                *,
                initial: Optional[U] = None,
            ) -> Template[Union[T, U]]:
                return type(self)(accumulate(self, func, initial=initial))

        else:
            raise UnsupportVersionError(VERSION)  # pragma: no cover

        return method

    _doc = "accumulate([1,2,3,4,5]) --> 1 3 6 10 15"


class ChainMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[ChainMethodBuilder]) -> Callable[..., Any]:
        def method(self: Template[T], *iterables: Iterable[U]) -> Template[Union[T, U]]:
            return type(self)(chain(self, *iterables))

        return method

    _doc = "chain('ABC', 'DEF') --> A B C D E F"


class CombinationsMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[CombinationsMethodBuilder]) -> Callable[..., Any]:
        def method(self: Template[T], r: int) -> Template[CTuple[T]]:
            return type(self)(map(CTuple, combinations(self, r)))

        return method

    _doc = "\n".join(
        [
            "combinations('ABCD', 2) --> AB AC AD BC BD CD",
            "combinations(range(4), 3) --> 012 013 023 123",
        ],
    )


class CombinationsWithReplacementMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[CombinationsWithReplacementMethodBuilder]) -> Callable[..., Any]:
        def method(self: Template[T], r: int) -> Template[CTuple[T]]:
            return type(self)(map(CTuple, combinations_with_replacement(self, r)))

        return method

    _doc = "combinations_with_replacement('ABC', 2) --> AA AB AC BB BC CC"


class CompressMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[CompressMethodBuilder]) -> Callable[..., Any]:
        def method(self: Template[T], selectors: Iterable[Any]) -> Template[T]:
            return type(self)(compress(self, selectors))

        return method

    _doc = "compress('ABCDEF', [1,0,1,0,1,1]) --> A C E F"


class CountMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[CountMethodBuilder]) -> Callable[..., Any]:
        def method(cls: Type[Template[T]], start: int = 0, step: int = 1) -> Template[int]:
            return cls(count(start=start, step=step))

        return method

    _doc = "count(10) --> 10 11 12 13 14 ..."


class CycleMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[CycleMethodBuilder]) -> Callable[..., Any]:
        def method(self: Template[T]) -> Template[T]:
            return type(self)(cycle(self))

        return method

    _doc = "cycle('ABCD') --> A B C D A B C D A B C D ..."


class DropWhileMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[DropWhileMethodBuilder]) -> Callable[..., Any]:
        def method(self: Template[T], func: Callable[[T], bool]) -> Template[T]:
            return type(self)(dropwhile(func, self))

        return method

    _doc = "dropwhile(lambda x: x<5, [1,4,6,4,1]) --> 6 4 1"


class FilterFalseMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[FilterFalseMethodBuilder]) -> Callable[..., Any]:
        def method(self: Template[T], func: Callable[[T], bool]) -> Template[T]:
            return type(self)(filterfalse(func, self))

        return method

    _doc = "filterfalse(lambda x: x%2, range(10)) --> 0 2 4 6 8"


class GroupByMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[GroupByMethodBuilder]) -> Callable[..., Any]:
        def method(
            self: Template[T], key: Optional[Callable[[T], U]] = None,
        ) -> Template[Tuple[U, Template[T]]]:
            cls = type(self)
            if cls is CSet:
                inner_cls = CFrozenSet
            else:
                inner_cls = cls
            return cls(((k, inner_cls(v))) for k, v in groupby(self, key=key))

        return method

    _doc = "\n".join(
        [
            "[k for k, g in groupby('AAAABBBCCDAABBB')] --> A B C D A B",
            "[list(g) for k, g in groupby('AAAABBBCCD')] --> AAAA BBB CC D",
        ],
    )


class ISliceMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[ISliceMethodBuilder]) -> Callable[..., Any]:
        def method(
            self: Template[T], start: int, stop: Optional[int] = None, step: Optional[int] = None,
        ) -> Template[T]:
            if (stop is None) and (step is not None):
                raise StopArgumentMissing()
            else:
                return type(self)(
                    islice(
                        self,
                        start,
                        *(() if stop is None else (stop,)),
                        *(() if step is None else (step,)),
                    ),
                )

        return method

    _doc = "\n".join(
        [
            "islice('ABCDEFG', 2) --> A B",
            "islice('ABCDEFG', 2, 4) --> C D",
            "islice('ABCDEFG', 2, None) --> C D E F G",
            "islice('ABCDEFG', 0, None, 2) --> A C E G",
        ],
    )


class PermutationsBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[PermutationsBuilder]) -> Callable[..., Any]:
        def method(self: Template[T], r: Optional[int] = None) -> Template[CTuple[T]]:
            return type(self)(map(CTuple, permutations(self, r=r)))

        return method

    _doc = "\n".join(
        [
            "permutations('ABCD', 2) --> AB AC AD BA BC BD CA CB CD DA DB DC",
            "permutations(range(3)) --> 012 021 102 120 201 210",
        ],
    )


class ProductMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[ProductMethodBuilder]) -> Callable[..., Any]:
        def method(
            self: Template[T], *iterables: Iterable[U], repeat: int = 1,
        ) -> Template[CTuple[T]]:
            return type(self)(map(CTuple, product(self, *iterables, repeat=repeat)))

        return method

    _doc = "Cartesian product of input iterables."


class RepeatMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(
        cls: Type[RepeatMethodBuilder], *, allow_infinite: bool,
    ) -> Callable[..., Any]:
        if allow_infinite:

            def method(cls: Type[Template[T]], x: T, times: Optional[int] = None) -> Template[T]:
                return cls(repeat(x, **({} if times is None else {"times": times})))

        else:

            def method(cls: Type[Template[T]], x: T, times: int) -> Template[T]:
                return cls(repeat(x, times=times))

        return method

    _doc = "repeat(10, 3) --> 10 10 10"


class StarMapMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[StarMapMethodBuilder]) -> Callable[..., Any]:
        def method(
            self: Template[Tuple[T, ...]], func: Callable[[Tuple[T, ...]], U],
        ) -> Template[U]:
            return type(self)(starmap(func, self))

        return method

    _doc = "starmap(pow, [(2,5), (3,2), (10,3)]) --> 32 9 1000"


class TakeWhileMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[TakeWhileMethodBuilder]) -> Callable[..., Any]:
        def method(self: Template[T], func: Callable[[T], bool]) -> Template[T]:
            return type(self)(takewhile(func, self))

        return method

    _doc = "takewhile(lambda x: x<5, [1,4,6,4,1]) --> 1 4"


class TeeMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[TeeMethodBuilder]) -> Callable[..., Any]:
        def method(self: Template[T], n: int = 2) -> Template[Template[T]]:
            cls = type(self)
            if cls is CSet:
                inner_cls = CFrozenSet
            else:
                inner_cls = cls
            return cls(map(inner_cls, tee(self, n)))

        return method

    _doc = "Return n independent iterators from a single iterable."


class ZipLongestMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[ZipLongestMethodBuilder]) -> Callable[..., Any]:
        def method(
            self: Template[T], *iterables: Iterable[U], fillvalue: V = None,
        ) -> Template[CTuple[T]]:
            return type(self)(map(CTuple, zip_longest(self, *iterables, fillvalue=fillvalue)))

        return method

    _doc = "zip_longest('ABCD', 'xy', fillvalue='-') --> Ax By C- D-"


# more-itertools


def _build_chunked(name: str) -> Callable[..., Any]:
    is_citerable = name == _CIterable
    ann = _CIterable if is_citerable else _CList

    def chunked(self: f"{name}[T]", n: int) -> f"{ann}[{ann}[T]]":
        cls = CIterable if is_citerable else CList
        return cls(map(cls, more_itertools.chunked(self, n)))

    return chunked


def _build_distribute(name: str) -> Callable[..., Any]:
    is_citerable = name == _CIterable
    ann = _CIterable if is_citerable else _CList

    def distribute(self: f"{name}[T]", n: int) -> f"{ann}[{ann}[T]]":
        cls = CIterable if is_citerable else CList
        return cls(map(cls, more_itertools.distribute(n, self)))

    return distribute


def _build_divide(name: str) -> Callable[..., Any]:
    is_citerable = name == _CIterable
    ann = _CIterable if is_citerable else _CList

    def divide(self: f"{name}[T]", n: int) -> f"{ann}[{ann}[T]]":
        cls = CIterable if is_citerable else CList
        return cls(map(cls, more_itertools.divide(n, list(self))))

    return divide


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

    # built-ins

    all = _build_all(_CIterable)  # noqa: A003
    any = _build_any(_CIterable)  # noqa: A003
    dict = DictMethodBuilder(_CIterable)  # noqa: A003
    enumerate = EnumerateMethodBuilder(_CIterable)  # noqa: A003
    filter = FilterMethodBuilder(_CIterable)  # noqa: A003
    frozenset = FrozenSetMethodBuilder(_CIterable)  # noqa: A003
    iter = IterMethodBuilder(_CIterable)  # noqa: A003
    list = ListMethodBuilder(_CIterable)  # noqa: A003
    map = MapMethodBuilder(_CIterable)  # noqa: A003
    max = MaxMinMethodBuilder(_CIterable, func=max)  # noqa: A003
    min = MaxMinMethodBuilder(_CIterable, func=min)  # noqa: A003
    range = classmethod(RangeMethodBuilder(_CIterable))  # noqa: A003
    set = SetMethodBuilder(_CIterable)  # noqa: A003
    sorted = SortedMethodBuilder(_CIterable)  # noqa: A003
    sum = SumMethodBuilder(_CIterable)  # noqa: A003
    tuple = TupleMethodBuilder(_CIterable)  # noqa: A003
    zip = ZipMethodBuilder(_CIterable)  # noqa: A003

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

    combinations = CombinationsMethodBuilder(_CIterable)
    combinations_with_replacement = CombinationsWithReplacementMethodBuilder(_CIterable)
    count = classmethod(CountMethodBuilder(_CIterable))
    cycle = CycleMethodBuilder(_CIterable)
    repeat = classmethod(RepeatMethodBuilder(_CIterable, allow_infinite=True))
    accumulate = AccumulateMethodBuilder(_CIterable)
    chain = ChainMethodBuilder(_CIterable)
    compress = CompressMethodBuilder(_CIterable)
    dropwhile = DropWhileMethodBuilder(_CIterable)
    filterfalse = FilterFalseMethodBuilder(_CIterable)
    groupby = GroupByMethodBuilder(_CIterable)
    islice = ISliceMethodBuilder(_CIterable)
    permutations = PermutationsBuilder(_CIterable)
    product = ProductMethodBuilder(_CIterable)
    starmap = StarMapMethodBuilder(_CIterable)
    takewhile = TakeWhileMethodBuilder(_CIterable)
    tee = TeeMethodBuilder(_CIterable)
    zip_longest = ZipLongestMethodBuilder(_CIterable)

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
        iterator = iter(self)
        consume(iterator, n=n)
        return CIterable(iterator)

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
        return cls(repeatfunc(func, times, *args))

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

    # more-itertools

    chunked = _build_chunked(_CIterable)
    distribute = _build_distribute(_CIterable)
    divide = _build_divide(_CIterable)

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
    dict = DictMethodBuilder(_CList)  # noqa: A003
    enumerate = EnumerateMethodBuilder(_CList)  # noqa: A003
    filter = FilterMethodBuilder(_CList)  # noqa: A003
    frozenset = FrozenSetMethodBuilder(_CList)  # noqa: A003
    iter = IterMethodBuilder(_CList)  # noqa: A003
    len = LenMethodBuilder(_CList)  # noqa: A003
    list = ListMethodBuilder(_CList)  # noqa: A003
    map = MapMethodBuilder(_CList)  # noqa: A003
    max = MaxMinMethodBuilder(_CList, func=max)  # noqa: A003
    min = MaxMinMethodBuilder(_CList, func=min)  # noqa: A003
    range = classmethod(RangeMethodBuilder(_CList))  # noqa: A003
    set = SetMethodBuilder(_CList)  # noqa: A003
    sorted = SortedMethodBuilder(_CList)  # noqa: A003
    sum = SumMethodBuilder(_CList)  # noqa: A003
    tuple = TupleMethodBuilder(_CList)  # noqa: A003
    zip = ZipMethodBuilder(_CList)  # noqa: A003

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

    def reduce(
        self: CList[T], func: Callable[[T, T], T], initial: Union[U, Sentinel] = sentinel,
    ) -> Any:
        return self.iter().reduce(func, initial=initial)

    # itertools

    combinations = CombinationsMethodBuilder(_CList)
    combinations_with_replacement = CombinationsWithReplacementMethodBuilder(_CList)
    repeat = classmethod(RepeatMethodBuilder(_CList, allow_infinite=False))
    accumulate = AccumulateMethodBuilder(_CList)
    chain = ChainMethodBuilder(_CList)
    compress = CompressMethodBuilder(_CList)
    dropwhile = DropWhileMethodBuilder(_CList)
    filterfalse = FilterFalseMethodBuilder(_CList)
    groupby = GroupByMethodBuilder(_CList)
    islice = ISliceMethodBuilder(_CList)
    permutations = PermutationsBuilder(_CList)
    product = ProductMethodBuilder(_CList)
    starmap = StarMapMethodBuilder(_CList)
    takewhile = TakeWhileMethodBuilder(_CList)
    tee = TeeMethodBuilder(_CList)
    zip_longest = ZipLongestMethodBuilder(_CList)

    def permutations(self: CList[T], r: Optional[int] = None) -> CList[Tuple[T, ...]]:
        return self.iter().permutations(r=r).list()

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

    # more-itertools

    chunked = _build_chunked(_CList)
    distribute = _build_distribute(_CList)
    divide = _build_divide(_CList)

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

    def unzip(self: CList[Tuple[T, ...]]) -> Tuple[CList[T], ...]:
        return CList(self.iter().unzip()).map(CList)


class CTuple(Tuple[T]):
    """A tuple with chainable methods."""

    # built-ins

    all = _build_all(_CTuple)  # noqa: A003
    any = _build_any(_CTuple)  # noqa: A003
    dict = DictMethodBuilder(_CTuple)  # noqa: A003
    enumerate = EnumerateMethodBuilder(_CTuple)  # noqa: A003
    filter = FilterMethodBuilder(_CTuple)  # noqa: A003
    frozenset = FrozenSetMethodBuilder(_CTuple)  # noqa: A003
    iter = IterMethodBuilder(_CTuple)  # noqa: A003
    len = LenMethodBuilder(_CTuple)  # noqa: A003
    list = ListMethodBuilder(_CTuple)  # noqa: A003
    map = MapMethodBuilder(_CTuple)  # noqa: A003
    max = MaxMinMethodBuilder(_CTuple, func=max)  # noqa: A003
    min = MaxMinMethodBuilder(_CTuple, func=min)  # noqa: A003
    range = classmethod(RangeMethodBuilder(_CTuple))  # noqa: A003
    set = SetMethodBuilder(_CTuple)  # noqa: A003
    sorted = SortedMethodBuilder(_CTuple)  # noqa: A003
    sum = SumMethodBuilder(_CTuple)  # noqa: A003
    tuple = TupleMethodBuilder(_CTuple)  # noqa: A003
    zip = ZipMethodBuilder(_CTuple)  # noqa: A003

    # itertools

    combinations = CombinationsMethodBuilder(_CTuple)
    combinations_with_replacement = CombinationsWithReplacementMethodBuilder(_CTuple)
    repeat = classmethod(RepeatMethodBuilder(_CTuple, allow_infinite=False))
    accumulate = AccumulateMethodBuilder(_CTuple)
    chain = ChainMethodBuilder(_CTuple)
    compress = CompressMethodBuilder(_CTuple)
    dropwhile = DropWhileMethodBuilder(_CTuple)
    filterfalse = FilterFalseMethodBuilder(_CTuple)
    groupby = GroupByMethodBuilder(_CTuple)
    islice = ISliceMethodBuilder(_CTuple)
    permutations = PermutationsBuilder(_CTuple)
    product = ProductMethodBuilder(_CTuple)
    starmap = StarMapMethodBuilder(_CTuple)
    takewhile = TakeWhileMethodBuilder(_CTuple)
    tee = TeeMethodBuilder(_CTuple)
    zip_longest = ZipLongestMethodBuilder(_CTuple)

    # more-itertools

    chunked = _build_chunked(_CTuple)
    distribute = _build_distribute(_CTuple)
    divide = _build_divide(_CTuple)


class CSet(Set[T]):
    """A set with chainable methods."""

    # built-ins

    all = _build_all(_CSet)  # noqa: A003
    any = _build_any(_CSet)  # noqa: A003
    dict = DictMethodBuilder(_CSet)  # noqa: A003
    enumerate = EnumerateMethodBuilder(_CSet)  # noqa: A003
    filter = FilterMethodBuilder(_CSet)  # noqa: A003
    frozenset = FrozenSetMethodBuilder(_CSet)  # noqa: A003
    iter = IterMethodBuilder(_CSet)  # noqa: A003
    len = LenMethodBuilder(_CSet)  # noqa: A003
    list = ListMethodBuilder(_CSet)  # noqa: A003
    map = MapMethodBuilder(_CSet)  # noqa: A003
    max = MaxMinMethodBuilder(_CSet, func=max)  # noqa: A003
    min = MaxMinMethodBuilder(_CSet, func=min)  # noqa: A003
    range = classmethod(RangeMethodBuilder(_CSet))  # noqa: A003
    set = SetMethodBuilder(_CSet)  # noqa: A003
    sorted = SortedMethodBuilder(_CSet)  # noqa: A003
    sum = SumMethodBuilder(_CSet)  # noqa: A003
    tuple = TupleMethodBuilder(_CSet)  # noqa: A003
    zip = ZipMethodBuilder(_CSet)  # noqa: A003

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

    accumulate = AccumulateMethodBuilder(_CSet)
    chain = ChainMethodBuilder(_CSet)
    combinations = CombinationsMethodBuilder(_CSet)
    combinations_with_replacement = CombinationsWithReplacementMethodBuilder(_CSet)
    compress = CompressMethodBuilder(_CSet)
    dropwhile = DropWhileMethodBuilder(_CSet)
    filterfalse = FilterFalseMethodBuilder(_CSet)
    groupby = GroupByMethodBuilder(_CSet)
    islice = ISliceMethodBuilder(_CSet)
    permutations = PermutationsBuilder(_CSet)
    product = ProductMethodBuilder(_CSet)
    repeat = classmethod(RepeatMethodBuilder(_CSet, allow_infinite=False))
    starmap = StarMapMethodBuilder(_CSet)
    takewhile = TakeWhileMethodBuilder(_CSet)
    tee = TeeMethodBuilder(_CSet)
    zip_longest = ZipLongestMethodBuilder(_CSet)

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

    # more-itertools

    chunked = _build_chunked(_CSet)
    distribute = _build_distribute(_CSet)
    divide = _build_divide(_CSet)

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

    # built-ins

    all = _build_all(_CFrozenSet)  # noqa: A003
    any = _build_any(_CFrozenSet)  # noqa: A003
    dict = DictMethodBuilder(_CFrozenSet)  # noqa: A003
    enumerate = EnumerateMethodBuilder(_CFrozenSet)  # noqa: A003
    filter = FilterMethodBuilder(_CFrozenSet)  # noqa: A003
    frozenset = FrozenSetMethodBuilder(_CFrozenSet)  # noqa: A003
    iter = IterMethodBuilder(_CFrozenSet)  # noqa: A003
    len = LenMethodBuilder(_CFrozenSet)  # noqa: A003
    list = ListMethodBuilder(_CFrozenSet)  # noqa: A003
    map = MapMethodBuilder(_CFrozenSet)  # noqa: A003
    max = MaxMinMethodBuilder(_CFrozenSet, func=max)  # noqa: A003
    min = MaxMinMethodBuilder(_CFrozenSet, func=min)  # noqa: A003
    range = classmethod(RangeMethodBuilder(_CFrozenSet))  # noqa: A003
    set = SetMethodBuilder(_CFrozenSet)  # noqa: A003
    sorted = SortedMethodBuilder(_CFrozenSet)  # noqa: A003
    sum = SumMethodBuilder(_CFrozenSet)  # noqa: A003
    tuple = TupleMethodBuilder(_CFrozenSet)  # noqa: A003
    zip = ZipMethodBuilder(_CFrozenSet)  # noqa: A003

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

    accumulate = AccumulateMethodBuilder(_CFrozenSet)
    chain = ChainMethodBuilder(_CFrozenSet)
    combinations = CombinationsMethodBuilder(_CFrozenSet)
    combinations_with_replacement = CombinationsWithReplacementMethodBuilder(_CFrozenSet)
    compress = CompressMethodBuilder(_CFrozenSet)
    dropwhile = DropWhileMethodBuilder(_CFrozenSet)
    filterfalse = FilterFalseMethodBuilder(_CFrozenSet)
    groupby = GroupByMethodBuilder(_CFrozenSet)
    islice = ISliceMethodBuilder(_CFrozenSet)
    permutations = PermutationsBuilder(_CFrozenSet)
    product = ProductMethodBuilder(_CFrozenSet)
    repeat = classmethod(RepeatMethodBuilder(_CFrozenSet, allow_infinite=False))
    starmap = StarMapMethodBuilder(_CFrozenSet)
    takewhile = TakeWhileMethodBuilder(_CFrozenSet)
    tee = TeeMethodBuilder(_CFrozenSet)
    zip_longest = ZipLongestMethodBuilder(_CFrozenSet)

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

    # more-itertools

    chunked = _build_chunked(_CFrozenSet)
    distribute = _build_distribute(_CFrozenSet)
    divide = _build_divide(_CFrozenSet)
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
