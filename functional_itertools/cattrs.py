from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Generic
from typing import TypeVar
from typing import Union

from attr import asdict
from attr import evolve
from attr.exceptions import NotAnAttrsClassError

from functional_itertools.classes import CDict
from functional_itertools.classes import CIterable
from functional_itertools.classes import CList


T = TypeVar("T")
U = TypeVar("U")


class CAttrs(Generic[T]):
    def dict(self: CAttrs[T], *, recurse: bool = True) -> CDict[str, T]:  # noqa: A003
        mapping: CDict[str, T] = asdict(
            self, recurse=False, dict_factory=CDict,
        )
        if recurse:
            for key, value in mapping.items():
                try:
                    v_dict = value.dict()
                except AttributeError:
                    pass
                else:
                    mapping[key] = v_dict()
        return mapping

    def map_values(
        self: CAttrs[T], func: Callable[..., U], *attrs: CAttrs[U], recurse: bool = True,
    ) -> CAttrs[Union[T, U]]:
        return self._map_values(func, self, *attrs, recurse=recurse)

    def _map_values(
        self: Any, func: Callable[..., U], value: Any, *values: Any, recurse: bool,
    ) -> Any:
        try:
            asdict(value)
        except NotAnAttrsClassError:
            return func(value, *values)
        else:
            if CIterable(values).map(lambda x: isinstance(x, type(self))).all():
                mappings = CList([value]).chain(*values).map(lambda x: asdict(x, recurse=True))
                mapping = (
                    mappings.unique_everseen()
                    .map(lambda x: (x, mappings.map(lambda y: y[x])))
                    .dict()
                )
                if recurse:
                    kwargs = mapping.map_values(
                        lambda x: self._map_values(func, *x, recurse=recurse),
                    )
                else:
                    kwargs = mapping.map_values(lambda x: func(*x))
                return evolve(value, **kwargs)
            else:
                return func(value, *values)
