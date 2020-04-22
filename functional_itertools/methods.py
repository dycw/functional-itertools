from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Iterable
from typing import Tuple
from typing import TypeVar


T = TypeVar("T")


class MethodBuilderMeta(type):
    def __call__(cls: MethodBuilder, cls_name: int) -> Callable[..., Any]:
        method = cls._build_method()
        method.__annotations__ = {
            k: v.replace("Iterable", cls_name) for k, v in method.__annotations__.items()
        }
        method.__doc__ = cls._doc.format(cls_name)
        return method


class MethodBuilder(metaclass=MethodBuilderMeta):
    @classmethod
    def _build_method(cls: MethodBuilder) -> Callable[..., T]:  # noqa: U100
        raise NotImplementedError

    _doc = NotImplemented


class AllMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: MethodBuilder) -> Callable[..., T]:
        def method(self: Iterable[T]) -> bool:
            return all(self)

        return method

    _doc = "Return `True` if all elements of the {0} are true (or if the {0} is empty)."


class AnyMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: MethodBuilder) -> Callable[..., T]:
        def method(self: Iterable[T]) -> bool:
            return any(self)

        return method

    _doc = "Return `True` if any element of {0} is true. If the {0} is empty, return `False`."


class EnumerateMethodBuilder(MethodBuilder):
    @classmethod
    def _build_method(cls: MethodBuilder) -> Callable[..., T]:
        def method(self: Iterable[T], start: int = 0) -> Iterable[Tuple[int, T]]:
            return type(self)(enumerate(self, start=start))

        return method

    _doc = "Return an enumerate object, cast as a {0}."
