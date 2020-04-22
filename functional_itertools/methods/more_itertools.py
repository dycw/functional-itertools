from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Type
from typing import TypeVar

from more_itertools import distribute
from more_itertools import divide
from more_itertools import ichunked

from functional_itertools.methods.base import MethodBuilder
from functional_itertools.methods.base import Template


T = TypeVar("T")


class ChunkedMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[ChunkedMethodBuilder]) -> Callable[..., Any]:
        def method(self: Template[T], n: int) -> Template[Template[int]]:
            cls = type(self)
            return cls(ichunked(self, n)).map(cls)

        return method

    _doc = "distribute(3, [1, 2, 3, 4, 5, 6, 7]) --> [[1, 4, 7], [2, 5], [3, 6]]"


class DistributeMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[DistributeMethodBuilder]) -> Callable[..., Any]:
        def method(self: Template[T], n: int) -> Template[Template[T]]:
            cls = type(self)
            return cls(distribute(n, self)).map(cls)

        return method

    _doc = "distribute(3, [1, 2, 3, 4, 5, 6, 7]) --> [[1, 4, 7], [2, 5], [3, 6]]"


class DivideMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[DivideMethodBuilder]) -> Callable[..., Any]:
        def method(self: Template[T], n: int) -> Template[Template[T]]:
            cls = type(self)
            return cls(divide(n, list(self))).map(cls)

        return method

    _doc = "divide(3, [1, 2, 3, 4, 5, 6, 7]) --> [[1, 2, 3], [4, 5], [6, 7]]"
