from __future__ import annotations

from types import FunctionType
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import TypeVar


T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")


class Template(Iterable[T]):
    pass


class MethodBuilderMeta(type):
    def __call__(cls: MethodBuilder, cls_name: str) -> Callable[..., Any]:
        method = cls._build_method()
        method.__annotations__ = {
            k: v.replace("Template", cls_name) for k, v in method.__annotations__.items()
        }
        method.__doc__ = cls._doc.format(cls_name)
        return method


class MethodBuilder(metaclass=MethodBuilderMeta):
    @classmethod
    def _build_method(cls: MethodBuilder) -> FunctionType:  # noqa: U100
        raise NotImplementedError

    _doc = NotImplemented


class AllMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: MethodBuilder) -> Callable[..., Any]:
        def method(self: Template[T]) -> bool:
            return all(self)

        return method

    _doc = "Return `True` if all elements of the {0} are true (or if the {0} is empty)."


class AnyMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: MethodBuilder) -> Callable[..., Any]:
        def method(self: Template[T]) -> bool:
            return any(self)

        return method

    _doc = "Return `True` if any element of {0} is true. If the {0} is empty, return `False`."


class EnumerateMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: MethodBuilder) -> Callable[..., Any]:
        def method(self: Template[T], start: int = 0) -> Template[Tuple[int, T]]:
            return type(self)(enumerate(self, start=start))

        return method

    _doc = "Return an enumerate object, cast as a {0}."


class FilterMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: MethodBuilder) -> Callable[..., Any]:
        def method(self: Template[T], func: Optional[Callable[[T], bool]]) -> Template[T]:
            return type(self)(filter(func, self))

        return method

    _doc = "Construct a {0} from those elements of the {0} for which function returns true."


class MapMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: MethodBuilder) -> Callable[..., Any]:
        def method(self: Template[T], func: Callable[..., U], *iterables: Iterable) -> Template[U]:
            return type(self)(map(func, self, *iterables))

        return method

    _doc = "Construct a {0} by applying `func` to every item of the {0}."
