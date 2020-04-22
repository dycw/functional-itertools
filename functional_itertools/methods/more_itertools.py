from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Type
from typing import TypeVar

from more_itertools import divide

from functional_itertools.methods.base import MethodBuilder
from functional_itertools.methods.base import Template

T = TypeVar("T")
U = TypeVar("U")


class DivideMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[DivideMethodBuilder]) -> Callable[..., Any]:
        def method(self: Template[T], n: int) -> Template[Template[int]]:
            cls = type(self)
            return cls(divide(n, list(self))).map(cls)

        return method

    _doc = "divide(2, [1, 2, 3, 4, 5, 6]) --> [[1, 2, 3], [4, 5, 6]]"
