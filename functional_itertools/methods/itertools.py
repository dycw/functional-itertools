from __future__ import annotations

from itertools import accumulate
from itertools import chain
from itertools import combinations
from itertools import combinations_with_replacement
from itertools import compress
from itertools import count,groupby ,zip_longest
from itertools import cycle
from itertools import dropwhile
from itertools import filterfalse
from itertools import islice
from itertools import permutations
from itertools import product
from itertools import repeat
from itertools import starmap
from itertools import takewhile
from itertools import tee
from operator import add
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union

from functional_itertools.errors import StopArgumentMissing
from functional_itertools.errors import UnsupportVersionError
from functional_itertools.methods.base import MethodBuilder
from functional_itertools.methods.base import Template
from functional_itertools.utilities import VERSION
from functional_itertools.utilities import Version


T = TypeVar("T")
U = TypeVar("U")


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


class CompressMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[CompressMethodBuilder]) -> Callable[..., Any]:
        def method(self: Template[T], selectors: Iterable[Any]) -> Template[T]:
            return type(self)(compress(self, selectors))

        return method

    _doc = "compress('ABCDEF', [1,0,1,0,1,1]) --> A C E F"


class CountMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[CycleMethodBuilder]) -> Callable[..., Any]:
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
    def _build_method(cls: Type[CompressMethodBuilder]) -> Callable[..., Any]:
        def method(
            self: Template[T], key: Optional[Callable[[T], U]] = None,
        ) -> Template[Tuple[U, Template[T]]]:
            cls = type(self)
            return cls(((k, cls(v))) for k, v in groupby(self, key=key))

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


class ProductBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[ProductBuilder]) -> Callable[..., Any]:
        def method(
            self: Template[T], *iterables: Iterable[U], repeat: int = 1,
        ) -> Template[Template[T]]:
            cls = type(self)
            return cls(product(self, *iterables, repeat=repeat)).map(cls)

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
            return cls(tee(self, n)).map(cls)

        return method

    _doc = "Return n independent iterators from a single iterable."


class ZipLongestMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[ZipLongestMethodBuilder]) -> Callable[..., Any]:
        def method(
            self: Template[T], *iterables: Iterable[U], fillvalue: V = None,
        ) -> Template[Template[T]]:
            cls = type(self)
            return cls(zip_longest(self, *iterables, fillvalue=fillvalue)).map(cls)

        return method

    _doc = "zip_longest('ABCD', 'xy', fillvalue='-') --> Ax By C- D-"


if 0:

    def permutations(self: CList[T], r: Optional[int] = None) -> CList[Tuple[T, ...]]:
        return self.iter().permutations(r=r).list()

    def combinations(self: CList[T], r: int) -> CList[Tuple[T, ...]]:
        return self.iter().combinations(r).list()

    def combinations_with_replacement(self: CList[T], r: int) -> CList[Tuple[T, ...]]:
        return self.iter().combinations_with_replacement(r).list()