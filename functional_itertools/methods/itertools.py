from __future__ import annotations

from itertools import count
from itertools import cycle
from itertools import repeat
from typing import Any
from typing import Callable
from typing import Optional
from typing import Type
from typing import TypeVar

from functional_itertools.methods.base import MethodBuilder
from functional_itertools.methods.base import Template


T = TypeVar("T")


class CountMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: MethodBuilder) -> Callable[..., Any]:
        def method(cls: Type[Template[T]], start: int = 0, step: int = 1) -> Template[int]:
            return cls(count(start=start, step=step))

        return method

    _doc = "count(10) --> 10 11 12 13 14 ..."


class CycleMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: MethodBuilder) -> Callable[..., Any]:
        def method(self: Template[T]) -> Template[T]:
            return type(self)(cycle(self))

        return method

    _doc = "cycle('ABCD') --> A B C D A B C D A B C D ..."


class RepeatMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: MethodBuilder, *, allow_infinite: bool) -> Callable[..., Any]:
        if allow_infinite:

            def method(cls: Type[Template[T]], x: T, times: Optional[int] = None) -> Template[T]:
                return cls(repeat(x, times=times))

        else:

            def method(cls: Type[Template[T]], x: T, times: int) -> Template[T]:
                return cls(repeat(x, times=times))

        return method

    _doc = "repeat(10, 3) --> 10 10 10"
