from __future__ import annotations

from functools import reduce
from itertools import chain
from itertools import permutations
from itertools import starmap
from operator import add
from operator import neg
from operator import or_
from pathlib import Path
from re import escape
from string import ascii_lowercase
from sys import maxsize
from tempfile import TemporaryDirectory
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import NoReturn
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

from hypothesis import assume
from hypothesis import example
from hypothesis import given
from hypothesis.strategies import booleans
from hypothesis.strategies import data
from hypothesis.strategies import DataObject
from hypothesis.strategies import fixed_dictionaries
from hypothesis.strategies import integers
from hypothesis.strategies import iterables
from hypothesis.strategies import just
from hypothesis.strategies import lists
from hypothesis.strategies import none
from hypothesis.strategies import text
from hypothesis.strategies import tuples
from pytest import mark
from pytest import raises

from functional_itertools import CDict
from functional_itertools import CFrozenSet
from functional_itertools import CIterable
from functional_itertools import CList
from functional_itertools import CSet
from functional_itertools import CTuple
from functional_itertools import EmptyIterableError
from functional_itertools import MultipleElementsError
from functional_itertools.utilities import drop_sentinel
from functional_itertools.utilities import Sentinel
from functional_itertools.utilities import sentinel
from functional_itertools.utilities import VERSION
from functional_itertools.utilities import Version
from tests.strategies import islice_ints
from tests.strategies import nested_siterables
from tests.strategies import siterables
from tests.test_utilities import is_even


CLASSES = [CIterable, CList, CTuple, CSet, CFrozenSet]


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


# built-ins


@mark.parametrize("cls", CLASSES)
@given(data=data())
def test_all(cls: Type, data: DataObject) -> None:
    x, _ = data.draw(siterables(cls, booleans()))
    y = cls(x).all()
    assert isinstance(y, bool)
    assert y == all(x)


@mark.parametrize("cls", CLASSES)
@given(data=data())
def test_any(cls: Type, data: DataObject) -> None:
    x, _ = data.draw(siterables(cls, booleans()))
    y = cls(x).any()
    assert isinstance(y, bool)
    assert y == any(x)


@mark.parametrize("cls", CLASSES)
@given(data=data())
def test_dict(cls: Type, data: DataObject) -> None:
    x, _ = data.draw(siterables(cls, tuples(integers(), integers())))
    assert isinstance(cls(x).dict(), CDict)


@mark.parametrize("cls", CLASSES)
@given(data=data(), start=integers())
def test_enumerate(cls: Type, data: DataObject, start: int) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    y = cls(x).enumerate(start=start)
    assert isinstance(y, cls)
    if cls in {CIterable, CList, CTuple}:
        assert cast(y) == cast(enumerate(x, start=start))


@mark.parametrize("cls", CLASSES)
@given(data=data())
def test_filter(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    y = cls(x).filter(is_even)
    assert isinstance(y, cls)
    assert cast(y) == cast(filter(is_even, x))


@mark.parametrize("cls", CLASSES)
@given(data=data())
def test_frozenset(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(siterables(cls, tuples(integers(), integers())))
    y = cls(x).frozenset()
    assert isinstance(y, CFrozenSet)
    assert cast(y) == cast(frozenset(x))


@mark.parametrize("cls", CLASSES)
@given(data=data())
def test_iter(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(siterables(cls, tuples(integers(), integers())))
    y = cls(x).iter()
    assert isinstance(y, CIterable)
    assert cast(y) == cast(iter(x))


@mark.parametrize("cls", [CList, CTuple, CSet, CFrozenSet])
@given(data=data())
def test_len(cls: Type, data: DataObject) -> None:
    x, _ = data.draw(siterables(cls, integers()))
    y = cls(x).len()
    assert isinstance(y, int)
    assert y == len(x)


@mark.parametrize("cls", CLASSES)
@given(data=data())
def test_list(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    y = cls(x).list()
    assert isinstance(y, CList)
    assert cast(y) == cast(list(x))


@mark.parametrize("cls", CLASSES)
@given(data=data())
def test_map(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    y = cls(x).map(neg)
    assert isinstance(y, cls)
    assert cast(y) == cast(map(neg, x))


@mark.parametrize("cls", CLASSES)
@mark.parametrize("func", [max, min])
@given(
    data=data(),
    key_kwargs=just({})
    | (
        just({"key": neg})
        if VERSION is Version.py37
        else fixed_dictionaries({"key": none() | just(neg)})
    ),
    default_kwargs=just({}) | fixed_dictionaries({"default": integers()}),
)
def test_max_and_min(
    cls: Type,
    func: Callable[..., int],
    data: DataObject,
    key_kwargs: Dict[str, int],
    default_kwargs: Dict[str, int],
) -> None:
    x, _ = data.draw(siterables(cls, integers()))
    try:
        y = getattr(cls(x), func.__name__)(**key_kwargs, **default_kwargs)
    except ValueError:
        with raises(
            ValueError, match=escape(f"{func.__name__}() arg is an empty sequence"),
        ):
            func(x, **key_kwargs, **default_kwargs)
    else:
        assert isinstance(y, int)
        assert y == func(x, **key_kwargs, **default_kwargs)


@mark.parametrize("cls", CLASSES)
@given(
    data=data(), start=islice_ints, stop=none() | islice_ints, step=none() | islice_ints,
)
def test_range(
    cls: Type, data: DataObject, start: int, stop: Optional[int], step: Optional[int],
) -> None:
    if step is not None:
        assume((stop is not None) and (step != 0))
    x = cls.range(start, stop, step)
    assert isinstance(x, cls)
    _, cast = data.draw(siterables(cls, none()))
    assert cast(x) == cast(
        range(start, *(() if stop is None else (stop,)), *(() if step is None else (step,))),
    )


@mark.parametrize("cls", CLASSES)
@given(data=data())
def test_set(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    y = cls(x).set()
    assert isinstance(y, CSet)
    assert cast(y) == cast(set(x))


@mark.parametrize("cls", CLASSES)
@given(data=data(), key=none() | just(neg), reverse=booleans())
def test_sorted(
    cls: Type, data: DataObject, key: Optional[Callable[[int], int]], reverse: bool,
) -> None:
    x, _ = data.draw(siterables(cls, integers()))
    y = cls(x).sorted(key=key, reverse=reverse)
    assert isinstance(y, CList)
    assert y == sorted(x, key=key, reverse=reverse)


@mark.parametrize("cls", CLASSES)
@given(data=data(), start=integers() | just(sentinel))
def test_sum(cls: Type, data: DataObject, start: Union[int, Sentinel]) -> None:
    x, _ = data.draw(siterables(cls, integers()))
    y = cls(x).sum(start=start)
    assert isinstance(y, int)
    assert y == sum(x, *(() if start is sentinel else (start,)))


@mark.parametrize("cls", CLASSES)
@given(data=data())
def test_tuple(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    y = cls(x).tuple()
    assert isinstance(y, CTuple)
    assert cast(y) == cast(tuple(x))


@mark.parametrize("cls", CLASSES)
@given(data=data())
def test_zip(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    xs, _ = data.draw(nested_siterables(cls, integers()))
    y = cls(x).zip(*xs)
    assert isinstance(y, cls)
    if cls in {CIterable, CList, CTuple}:
        assert cast(y) == cast(zip(x, *xs))


# functools


@mark.parametrize("cls", [CIterable, CList, CSet, CFrozenSet])
@given(data=data(), initial=integers() | just(sentinel))
def test_reduce(cls: Type, data: DataObject, initial: Union[int, Sentinel]) -> None:
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
    "cls, cls_base, func", [(CList, list, add), (CSet, set, or_), (CFrozenSet, frozenset, or_)],
)
@given(data=data())
def test_reduce_returning_c_classes(
    cls: Type, data: DataObject, cls_base: Type, func: Callable[[Any, Any], Any],
) -> None:
    x, cast = data.draw(nested_siterables(cls, integers(), min_size=1))
    assert isinstance(CIterable(x).map(cls_base).reduce(func), cls)


# itertools


# itertools-recipes


# multiprocessing


@mark.parametrize("cls", [CIterable, CList, CSet, CFrozenSet])
@given(data=data())
def test_pmap(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    y = cls(x).pmap(neg, processes=1)
    assert isinstance(y, cls)
    assert cast(y) == cast(map(neg, x))


@mark.parametrize("cls", [CIterable, CList, CSet, CFrozenSet])
@given(data=data())
def test_pmap_nested(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(nested_siterables(cls, integers(), min_size=1))
    y = cls(x).pmap(_pmap_neg, processes=1)
    assert isinstance(y, cls)
    assert cast(y) == cast(max(map(neg, x_i)) for x_i in x)


def _pmap_neg(x: Iterable[int]) -> int:
    return CIterable(x).pmap(neg, processes=1).max()


@mark.parametrize("cls", [CIterable, CList, CSet, CFrozenSet])
@given(data=data())
def test_pstarmap(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(siterables(cls, tuples(integers(), integers())))
    y = cls(x).pstarmap(max, processes=1)
    assert isinstance(y, cls)
    assert cast(y) == cast(starmap(max, x))


# pathlib


@mark.parametrize("cls", [CIterable, CList, CSet, CFrozenSet])
@mark.parametrize("use_path", [True, False])
@given(data=data())
def test_iterdir(cls: Type, data: DataObject, use_path: bool) -> None:
    x, cast = data.draw(siterables(cls, text(alphabet=ascii_lowercase, min_size=1), unique=True))
    with TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        for i in x:
            temp_dir.joinpath(i).touch()
        if use_path:
            y = cls.iterdir(temp_dir)
        else:
            y = cls.iterdir(temp_dir_str)
        assert isinstance(y, cls)
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


@mark.parametrize("cls", [CIterable, CList])
@given(x=lists(integers(), min_size=1))
def test_unzip(cls: Type, x: List[int]) -> None:
    indices, ints = cls(x).enumerate().unzip()
    assert isinstance(indices, cls)
    assert list(indices) == list(range(len(x)))
    assert isinstance(ints, cls)
    assert list(ints) == x
