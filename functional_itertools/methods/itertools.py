from __future__ import annotations

from itertools import accumulate
from itertools import chain
from itertools import count
from itertools import cycle
from itertools import repeat
from operator import add
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Optional
from typing import Type
from typing import TypeVar
from typing import Union

from functional_itertools.errors import UnsupportVersionError
from functional_itertools.methods.base import MethodBuilder
from functional_itertools.methods.base import Template
from functional_itertools.utilities import VERSION
from functional_itertools.utilities import Version


T = TypeVar("T")
U = TypeVar("U")


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
