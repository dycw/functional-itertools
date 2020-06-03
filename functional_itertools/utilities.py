from __future__ import annotations

from contextlib import contextmanager
from itertools import chain
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generator
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
from warnings import warn

from attr import has


T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")


# dropper


def _drop_object(
    *args: Any, _obj: Any, **kwargs: Any,
) -> Tuple[Tuple, Dict[str, Any]]:
    return (
        tuple(x for x in args if x is not _obj),
        {k: v for k, v in kwargs.items() if v is not _obj},
    )


def drop_none(*args: Any, **kwargs: Any) -> Tuple[Tuple, Dict[str, Any]]:
    return _drop_object(*args, _obj=None, **kwargs)


# helper functions


def helper_filter_keys(func: Callable[[T], bool], item: Tuple[T, Any]) -> bool:
    key, _ = item
    return func(key)


def helper_filter_values(
    func: Callable[[U], bool], item: Tuple[Any, U],
) -> bool:
    _, value = item
    return func(value)


def helper_map_dict(
    func: Callable[..., V], x: T, *iterables: U,
) -> Tuple[Union[T, Tuple[Union[T, U], ...]], V]:
    if iterables:
        return tuple(chain([x], iterables)), func(x, *iterables)
    else:
        return x, func(x)


def helper_starfilter(
    func: Callable[[Tuple[T, ...]], bool], x: Tuple[Tuple[T, ...], ...],
) -> bool:
    return func(*x)


def helper_map_keys(func: Callable[[T], V], item: Tuple[T, U]) -> Tuple[V, U]:
    key, value = item
    return func(key), value


def helper_map_values(
    func: Callable[[U], V], item: Tuple[T, U],
) -> Tuple[T, V]:
    key, value = item
    return key, func(value)


def helper_map_items(
    func: Callable[[T, U], Tuple[V, W]], item: Tuple[T, U],
) -> Tuple[V, W]:
    key, value = item
    return func(key, value)


def helper_cattrs_map_1(item: Tuple[str, Any]) -> bool:
    _, value = item
    return has(value)


# multiprocessing


@contextmanager
def suppress_daemonic_processes_with_children(
    error: AssertionError,
) -> Generator[None, None, None]:
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


# warn


def warn_non_functional(
    cls: Type, incorrect: str, *, suggestion: Optional[str] = None,
) -> None:
    name = cls.__name__
    msg = f"{name}.{incorrect} is a non-functional method"
    if suggestion is not None:
        msg += f"; did you mean {name}.{suggestion} instead?"
    warn(msg)
