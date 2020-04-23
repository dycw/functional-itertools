from __future__ import annotations

from types import FunctionType
from typing import Any
from typing import Callable
from typing import Iterable
from typing import TypeVar


T = TypeVar("T")


class Template(Iterable[T]):
    pass


class CIterableOrCList(Iterable[T]):
    pass


class MethodBuilderMeta(type):
    def __call__(cls: MethodBuilder, cls_name: str, **kwargs: Any) -> Callable:
        method = cls._build_method(**kwargs)
        method.__annotations__ = {
            k: v.replace(Template.__name__, cls_name) for k, v in method.__annotations__.items()
        }
        method.__doc__ = cls._doc.format(cls_name)
        return method


class MethodBuilder(metaclass=MethodBuilderMeta):
    @classmethod  # noqa: U100
    def _build_method(cls: MethodBuilder, **kwargs: Any) -> FunctionType:  # noqa: U100
        raise NotImplementedError

    _doc = NotImplemented
