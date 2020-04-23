from __future__ import annotations

from functools import reduce
from operator import add
from operator import or_
from re import escape
from typing import Any
from typing import Callable
from typing import NoReturn
from typing import Tuple
from typing import Type
from typing import Union

from hypothesis import given
from hypothesis.strategies import data
from hypothesis.strategies import DataObject
from hypothesis.strategies import integers
from hypothesis.strategies import just
from hypothesis.strategies import tuples
from pytest import mark
from pytest import raises

from functional_itertools import CFrozenSet
from functional_itertools import CIterable
from functional_itertools import CList
from functional_itertools import CSet
from functional_itertools import CTuple
from functional_itertools import EmptyIterableError
from functional_itertools.utilities import Sentinel
from functional_itertools.utilities import sentinel
from tests.strategies import CLASSES
from tests.strategies import nested_siterables
from tests.strategies import siterables


@mark.parametrize("cls", CLASSES)
@given(data=data(), initial=integers() | just(sentinel))
def test_reduce(cls: Type, data: DataObject, initial: Union[int, Sentinel]) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    args = () if initial is sentinel else (initial,)
    try:
        y = cls(x).reduce(add, initial=initial)
    except EmptyIterableError:
        with raises(
            TypeError, match=escape("reduce() of empty sequence with no initial value"),
        ):
            reduce(add, x, *args)
    else:
        assert isinstance(y, int)
        assert y == reduce(add, x, *args)


@given(x=tuples(integers(), integers()))
def test_reduce_does_not_suppress_type_errors(x: Tuple[int, int]) -> None:
    def func(x: Any, y: Any) -> NoReturn:
        raise TypeError("Always fail")

    with raises(TypeError, match="Always fail"):
        CIterable(x).reduce(func)


@mark.parametrize(
    "cls, cls_base, func",
    [(CList, list, add), (CTuple, tuple, add), (CSet, set, or_), (CFrozenSet, frozenset, or_)],
)
@given(data=data())
def test_reduce_returning_c_classes(
    cls: Type, data: DataObject, cls_base: Type, func: Callable[[Any, Any], Any],
) -> None:
    x, cast = data.draw(nested_siterables(cls, integers(), min_size=1))
    assert isinstance(CIterable(x).map(cls_base).reduce(func), cls)
