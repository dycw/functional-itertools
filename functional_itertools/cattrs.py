from __future__ import annotations

from functools import partial
from typing import Any
from typing import Callable
from typing import Generic
from typing import Optional
from typing import Type
from typing import TypeVar

from attr import asdict
from attr import astuple
from attr import Attribute
from attr import evolve
from attr import has

from functional_itertools import CDict
from functional_itertools import CList
from functional_itertools import CTuple


T = TypeVar("T")
U = TypeVar("U")


class CAttrs(Generic[T]):
    """A base class for the attrs package."""

    # built-in

    def dict(  # noqa: A003
        self: CAttrs[T],
        *,
        recurse: bool = True,
        filter: Optional[Callable[[Attribute, Any], bool]] = None,  # noqa: A002
    ) -> CDict[str, T]:
        return helper_cattrs_dict(self, recurse=recurse, filter=filter)

    def list(  # noqa: A003
        self: CAttrs[T],
        *,
        recurse: bool = True,
        filter: Optional[Callable[[Attribute, Any], bool]] = None,  # noqa: A002
    ) -> CList[T]:
        return helper_cattrs_tuple(self, recurse=recurse, filter=filter, tuple_factory=CList)

    def map(  # noqa: A003
        self: CAttrs[T],
        func: Callable[..., U],
        parallel: bool = False,
        processes: Optional[int] = None,
        recurse: bool = True,
        filter: Optional[Callable[[Attribute, Any], bool]] = None,  # noqa: A002
    ) -> CAttrs[U]:
        return helper_cattrs_map(
            self, func=func, parallel=parallel, processes=processes, recurse=recurse, filter=filter,
        )

    def tuple(  # noqa: A003
        self: CAttrs[T],
        *,
        recurse: bool = True,
        filter: Optional[Callable[[Attribute, Any], bool]] = None,  # noqa: A002
    ) -> CTuple[T]:
        return helper_cattrs_tuple(self, recurse=recurse, filter=filter)


def helper_cattrs_dict(
    x: Any,
    *,
    recurse: bool = True,
    filter: Optional[Callable[[Attribute, Any], bool]] = None,  # noqa: A002
) -> Any:
    if isinstance(x, CAttrs):
        res: CDict[T] = asdict(x, recurse=False, filter=filter, dict_factory=CDict)
        if recurse:
            return res.map_values(partial(helper_cattrs_dict, recurse=recurse, filter=filter))
        else:
            return res
    else:
        return x


def helper_cattrs_map(
    x: Any,
    *,
    func: Callable[..., U],
    parallel: bool = False,
    processes: Optional[int] = None,
    recurse: bool = True,
    filter: Optional[Callable[[Attribute, Any], bool]] = None,  # noqa: A002
) -> Any:
    if isinstance(x, CAttrs):
        not_attr_items, is_attr_items = (
            x.dict(recurse=False, filter=filter)
            .items()
            .partition(lambda x: has(x[1]))
            .map(lambda x: x.dict())
        )
        if recurse:
            extra = is_attr_items.map_values(
                partial(
                    helper_cattrs_map,
                    func=func,
                    parallel=parallel,
                    processes=processes,
                    recurse=recurse,
                    filter=filter,
                ),
            )
        else:
            extra = is_attr_items
        return evolve(x, **not_attr_items.map_values(func), **extra)
    else:
        return func(x)


def helper_cattrs_tuple(
    x: Any,
    *,
    recurse: bool = True,
    filter: Optional[Callable[[Attribute, Any], bool]] = None,  # noqa: A002
    tuple_factory: Type = CTuple,
) -> Any:
    if isinstance(x, CAttrs):
        res: CTuple[T] = astuple(x, recurse=False, filter=filter, tuple_factory=tuple_factory)
        if recurse:
            return res.map(
                partial(
                    helper_cattrs_tuple,
                    recurse=recurse,
                    filter=filter,
                    tuple_factory=tuple_factory,
                ),
            )
        else:
            return res
    else:
        return x
