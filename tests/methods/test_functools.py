from __future__ import annotations

from functools import reduce
from operator import add
from operator import or_
from re import escape
from typing import Any
from typing import Callable
from typing import List
from typing import NoReturn
from typing import Tuple
from typing import Type
from typing import Union

from hypothesis.strategies import integers
from hypothesis.strategies import just
from hypothesis.strategies import lists
from hypothesis.strategies import tuples
from pytest import raises

from functional_itertools import CFrozenSet
from functional_itertools import CIterable
from functional_itertools import CList
from functional_itertools import CSet
from functional_itertools import CTuple
from functional_itertools import EmptyIterableError
from functional_itertools.utilities import drop_sentinel
from functional_itertools.utilities import Sentinel
from functional_itertools.utilities import sentinel
from tests import given
from tests import parametrize
from tests.strategies import Case
from tests.strategies import CASES


@given(x=lists(integers()), initial=integers() | just(sentinel))
@parametrize("case", CASES)
def test_reduce(
    x: List[int], initial: Union[int, Sentinel], case: Case,
) -> None:
    args, _ = drop_sentinel(initial)
    try:
        y = case.cls(x).reduce(add, initial=initial)
    except EmptyIterableError:
        with raises(
            TypeError,
            match=escape("reduce() of empty sequence with no initial value"),
        ):
            reduce(add, x, *args)
    else:
        assert isinstance(y, int)
        assert y == reduce(add, case.cast(x), *args)


@given(x=tuples(integers(), integers()))
def test_reduce_does_not_suppress_type_errors(x: Tuple[int, int]) -> None:
    def func(x: Any, y: Any) -> NoReturn:
        raise TypeError("Always fail")

    with raises(TypeError, match="Always fail"):
        CIterable(x).reduce(func)


@given(x=lists(lists(integers(), min_size=1), min_size=1))
@parametrize(
    "cls, cls_base, func",
    [
        (CList, list, add),
        (CTuple, tuple, add),
        (CSet, set, or_),
        (CFrozenSet, frozenset, or_),
    ],
)
def test_reduce_returning_c_classes(
    x: List[List[int]],
    cls: Type,
    cls_base: Type,
    func: Callable[[Any, Any], Any],
) -> None:
    assert isinstance(CIterable(x).map(cls_base).reduce(func), cls)
