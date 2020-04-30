from __future__ import annotations

from functools import partial
from typing import Any
from typing import Callable
from typing import Generic
from typing import Optional
from typing import TypeVar

from attr import asdict
from attr import Attribute
from attr import evolve
from attr import has

from functional_itertools.classes import CDict


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
        return asdict(self, recurse=recurse, filter=filter, dict_factory=CDict)

    def map(  # noqa: A003
        self: CAttrs[T],
        func: Callable[..., U],
        parallel: bool = False,
        processes: Optional[int] = None,
        recurse: bool = True,
    ) -> CAttrs[U]:
        return helper_cattrs_map(
            self, func, parallel=parallel, processes=processes, recurse=recurse,
        )


def helper_cattrs_map(
    value: Any,
    func: Callable[..., U],
    *,
    parallel: bool = False,
    processes: Optional[int] = None,
    recurse: bool = True,
) -> Any:
    if isinstance(value, CAttrs):
        not_attr_items, is_attr_items = (
            value.dict(recurse=False).items().partition(lambda x: has(x[1])).map(lambda x: x.dict())
        )
        if recurse:
            extra = is_attr_items.map_values(
                partial(
                    helper_cattrs_map,
                    func=func,
                    parallel=parallel,
                    processes=processes,
                    recurse=recurse,
                ),
            )
        else:
            extra = is_attr_items
        return evolve(value, **not_attr_items.map_values(func), **extra)
    else:
        return func(value)
