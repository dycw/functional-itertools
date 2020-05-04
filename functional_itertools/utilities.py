from __future__ import annotations

from contextlib import contextmanager
from enum import auto
from enum import Enum
from itertools import chain
from sys import version_info
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generator
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
from warnings import warn

from attr import has

from functional_itertools.errors import UnsupportVersionError


T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")


# dropper


def _drop_object(*args: Any, _obj: Any, **kwargs: Any) -> Tuple[Tuple, Dict[str, Any]]:
    return (
        tuple(x for x in args if x is not _obj),
        {k: v for k, v in kwargs.items() if v is not _obj},
    )


def drop_none(*args: Any, **kwargs: Any) -> Tuple[Tuple, Dict[str, Any]]:
    return _drop_object(*args, _obj=None, **kwargs)


# helper functions


def helper_filter_keys(item: Tuple[T, Any], *, func: Callable[[T], bool]) -> bool:
    key, _ = item
    return func(key)


def helper_filter_values(item: Tuple[Any, U], *, func: Callable[[U], bool]) -> bool:
    _, value = item
    return func(value)


def helper_map_dict(
    x: T, *iterables: U, func: Callable[..., V],
) -> Union[
    Tuple[T, V], Tuple[Union[T, U], V],
]:
    if iterables:
        return tuple(chain([x], iterables)), func(x, *iterables)
    else:
        return x, func(x)


def helper_starfilter(x: Tuple[T, ...], *, func: Callable[[Tuple[T, ...]], bool]) -> bool:
    return func(*x)


def helper_map_keys(item: Tuple[T, U], *, func: Callable[[T], V]) -> Tuple[V, U]:
    key, value = item
    return func(key), value


def helper_map_values(item: Tuple[T, U], *, func: Callable[[U], V]) -> Tuple[T, V]:
    key, value = item
    return key, func(value)


def helper_map_items(item: Tuple[T, U], *, func: Callable[[T, U], Tuple[V, W]]) -> Tuple[V, W]:
    key, value = item
    return func(key, value)


def helper_cattrs_map_1(item: Tuple[str, Any]) -> bool:
    _, value = item
    return has(value)


# multiprocessing


@contextmanager
def suppress_daemonic_processes_with_children(error: AssertionError) -> Generator[None, None, None]:
    (msg,) = error.args
    if msg == "daemonic processes are not allowed to have children":
        yield
    else:
        raise error


# sentinel


class Sentinel:
    def __repr__(self: Sentinel) -> str:
        return "<sentinel>"

    __str__ = __repr__


sentinel = Sentinel()


def drop_sentinel(*args: Any, **kwargs: Any) -> Tuple[Tuple, Dict[str, Any]]:
    return _drop_object(*args, _obj=sentinel, **kwargs)


# version


class Version(Enum):
    py37 = auto()
    py38 = auto()


def _get_version() -> Version:
    major, minor, *_ = version_info
    if major != 3:  # pragma: no cover
        raise RuntimeError(f"Expected Python 3; got {major}")
    mapping = {7: Version.py37, 8: Version.py38}
    try:
        return mapping[minor]
    except KeyError:  # pragma: no cover
        raise UnsupportVersionError(f"Expected Python 3.6-3.8; got 3.{minor}") from None


VERSION = _get_version()


# warn


def warn_non_functional(cls: Type, incorrect: str, suggestion: str) -> None:
    name = cls.__name__
    warn(
        f"{name}.{incorrect} is a non-functional method, did you mean {name}.{suggestion} instead?",
    )
