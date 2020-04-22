from __future__ import annotations

from itertools import count
from typing import Any
from typing import Callable
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
