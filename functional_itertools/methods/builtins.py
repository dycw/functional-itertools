from __future__ import annotations

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
from functional_itertools.utilities import Sentinel
from functional_itertools.utilities import sentinel
from functional_itertools.utilities import VERSION
from functional_itertools.utilities import Version


T = TypeVar("T")
U = TypeVar("U")


class AllMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[AllMethodBuilder]) -> Callable[..., Any]:
        def method(self: Template[T]) -> bool:
            return all(self)

        return method

    _doc = "Return `True` if all elements of the {0} are true (or if the {0} is empty)."


class AnyMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[AnyMethodBuilder]) -> Callable[..., Any]:
        def method(self: Template[T]) -> bool:
            return any(self)

        return method

    _doc = "Return `True` if any element of {0} is true. If the {0} is empty, return `False`."


class EnumerateMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[EnumerateMethodBuilder]) -> Callable[..., Any]:
        def method(self: Template[T], start: int = 0) -> Template[Tuple[int, T]]:
            return type(self)(enumerate(self, start=start))

        return method

    _doc = "Return an enumerate object, cast as a {0}."


class FilterMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[FilterMethodBuilder]) -> Callable[..., Any]:
        def method(self: Template[T], func: Optional[Callable[[T], bool]]) -> Template[T]:
            return type(self)(filter(func, self))

        return method

    _doc = "Construct a {0} from those elements of the {0} for which function returns true."


class LenMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[LenMethodBuilder]) -> Callable[..., int]:
        def method(self: Template[T]) -> int:
            return len(self)

        return method

    _doc = "Return the length of the {0}."


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


class SumMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[SumMethodBuilder]) -> Callable[..., int]:
        def method(self: Template[T], start: Union[U, Sentinel] = sentinel) -> Union[T, U]:
            return sum(self, *(() if start is sentinel else (start,)))

        return method

    _doc = "Return the sum of the elements in {0}."


class ZipMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: Type[ZipMethodBuilder]) -> Callable[..., int]:
        def method(self: Template[T], *iterables: Iterable[U]) -> Template[Tuple[Union[T, U]]]:
            return type(self)(zip(self, *iterables))

        return method

    _doc = "Return an iterator that aggregates elements from the {0} and the input iterables."
