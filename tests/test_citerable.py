from __future__ import annotations

from functools import reduce
from itertools import chain
from itertools import permutations
from operator import add
from operator import or_
from pathlib import Path
from re import escape
from string import ascii_lowercase
from sys import maxsize
from tempfile import TemporaryDirectory
from typing import Any
from typing import Callable
from typing import Iterable
from typing import List
from typing import NoReturn
from typing import Tuple
from typing import Type
from typing import Union

from hypothesis import example
from hypothesis import given
from hypothesis.strategies import data
from hypothesis.strategies import DataObject
from hypothesis.strategies import integers
from hypothesis.strategies import iterables
from hypothesis.strategies import just
from hypothesis.strategies import lists
from hypothesis.strategies import text
from hypothesis.strategies import tuples
from pytest import mark
from pytest import raises

from functional_itertools import CFrozenSet
from functional_itertools import CIterable
from functional_itertools import CList
from functional_itertools import CSet
from functional_itertools import EmptyIterableError
from functional_itertools import MultipleElementsError
from functional_itertools.utilities import drop_sentinel
from functional_itertools.utilities import Sentinel
from functional_itertools.utilities import sentinel
from tests.strategies import nested_siterables
from tests.strategies import siterables


@given(x=integers() | lists(integers()))
def test_init(x: Union[int, List[int]]) -> None:
    if isinstance(x, int):
        with raises(
            TypeError, match="CIterable expected an iterable, but 'int' object is not iterable",
        ):
            CIterable(x)  # type: ignore
    else:
        assert isinstance(CIterable(iter(x)), CIterable)


@given(x=lists(integers()), index=integers())
@example(x=[], index=-1)
@example(x=[], index=maxsize + 1)
@example(x=[], index=0.0)
def test_get_item(x: List[int], index: Union[int, float]) -> None:
    y = CIterable(x)
    if isinstance(index, int):
        num_ints = len(x)
        if index < 0:
            with raises(
                IndexError, match=f"Expected a non-negative index; got {index}",
            ):
                y[index]
        elif 0 <= index < num_ints:
            z = y[index]
            assert isinstance(z, int)
            assert z == x[index]
        elif num_ints <= index <= maxsize:
            with raises(IndexError, match="CIterable index out of range"):
                y[index]
        else:
            with raises(
                IndexError, match=f"Expected an index at most {maxsize}; got {index}",
            ):
                y[index]
    else:
        with raises(
            TypeError, match=escape("Expected an int or slice; got a(n) float"),
        ):
            y[index]


@given(x=lists(integers()))
def test_dunder_iter(x: List[int]) -> None:
    assert list(CIterable(x)) == x


# repr and str


@given(x=iterables(integers()))
def test_repr(x: Iterable[int]) -> None:
    assert repr(CIterable(x)) == f"CIterable({x!r})"


@given(x=iterables(integers()))
def test_str(x: Iterable[int]) -> None:
    assert str(CIterable(x)) == f"CIterable({x})"


# functools


@mark.parametrize("case", [CIterable, CList, CSet, CFrozenSet])
@given(data=data(), initial=integers() | just(sentinel))
def test_reduce(case: Case, data: DataObject, initial: Union[int, Sentinel]) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    args, _ = drop_sentinel(initial)
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
    "case, cls_base, func", [(CList, list, add), (CSet, set, or_), (CFrozenSet, frozenset, or_)],
)
@given(data=data())
def test_reduce_returning_c_classes(
    case: Case, data: DataObject, cls_base: Type, func: Callable[[Any, Any], Any],
) -> None:
    x, cast = data.draw(nested_siterables(cls, integers(), min_size=1))
    assert isinstance(CIterable(x).map(cls_base).reduce(func), cls)


# pathlib


@mark.parametrize("case", [CIterable, CList, CSet, CFrozenSet])
@mark.parametrize("use_path", [True, False])
@given(data=data())
def test_iterdir(case: Case, data: DataObject, use_path: bool) -> None:
    x, cast = data.draw(siterables(cls, text(alphabet=ascii_lowercase, min_size=1), unique=True))
    with TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        for i in x:
            temp_dir.joinpath(i).touch()
        if use_path:
            y = cls.iterdir(temp_dir)
        else:
            y = cls.iterdir(temp_dir_str)
        assert isinstance(y, case.cls)
        assert set(y) == {temp_dir.joinpath(i) for i in x}


# extra public


@given(x=lists(integers()), value=integers())
def test_append(x: List[int], value: int) -> None:
    y = CIterable(x).append(value)
    assert isinstance(y, CIterable)
    assert list(y) == list(chain(x, [value]))


@given(x=lists(integers()))
@mark.parametrize("method_name, index", [("first", 0), ("last", -1)])
def test_first_and_last(x: List[int], method_name: str, index: int) -> None:
    method = getattr(CIterable(x), method_name)
    if x:
        assert method() == x[index]
    else:
        with raises(EmptyIterableError):
            method()


@given(x=lists(integers()))
def test_one(x: List[int]) -> None:
    length = len(x)
    if length == 0:
        with raises(EmptyIterableError):
            CIterable(x).one()
    elif length == 1:
        assert CIterable(x).one() == x[0]
    else:
        with raises(MultipleElementsError, match=f"{x[0]}, {x[1]}"):
            CIterable(x).one()


@given(x=lists(integers()))
def test_pipe(x: List[int]) -> None:
    y = CIterable(x).pipe(permutations, r=2)
    assert isinstance(y, CIterable)
    assert list(y) == list(permutations(x, r=2))


@mark.parametrize("case", [CIterable, CList])
@given(x=lists(integers(), min_size=1))
def test_unzip(case: Case, x: List[int]) -> None:
    indices, ints = cls(x).enumerate().unzip()
    assert isinstance(indices, case.cls)
    assert list(indices) == list(range(len(x)))
    assert isinstance(ints, case.cls)
    assert list(ints) == x
