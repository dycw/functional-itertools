from functools import reduce
from itertools import accumulate
from itertools import chain
from itertools import combinations
from itertools import combinations_with_replacement
from itertools import compress
from itertools import count
from itertools import cycle
from itertools import dropwhile
from itertools import filterfalse
from itertools import groupby
from itertools import islice
from itertools import permutations
from itertools import product
from itertools import repeat
from itertools import starmap
from itertools import takewhile
from itertools import tee
from itertools import zip_longest
from operator import neg
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
from hypothesis import settings
from hypothesis.strategies import booleans
from hypothesis.strategies import data
from hypothesis.strategies import DataObject
from hypothesis.strategies import fixed_dictionaries
from hypothesis.strategies import floats
from hypothesis.strategies import integers
from hypothesis.strategies import iterables
from hypothesis.strategies import just
from hypothesis.strategies import lists
from hypothesis.strategies import none
from hypothesis.strategies import text
from hypothesis.strategies import tuples
from more_itertools.recipes import flatten
from more_itertools.recipes import prepend
from more_itertools.recipes import tabulate
from more_itertools.recipes import take
from pytest import mark
from pytest import param
from pytest import raises

from functional_itertools.classes import CDict
from functional_itertools.classes import CFrozenSet
from functional_itertools.classes import CIterable
from functional_itertools.classes import CList
from functional_itertools.classes import CSet
from functional_itertools.errors import EmptyIterableError
from functional_itertools.errors import MultipleElementsError
from functional_itertools.utilities import drop_sentinel
from functional_itertools.utilities import Sentinel
from functional_itertools.utilities import sentinel
from functional_itertools.utilities import VERSION
from functional_itertools.utilities import Version
from tests.test_utilities import is_even
from tests.test_utilities import lists_or_sets
from tests.test_utilities import small_ints


@given(x=integers() | lists(integers()))
def test_init(x: Union[int, List[int]]) -> None:
    if isinstance(x, int):
        with raises(
            TypeError, match="CIterable expected an iterable, " "but 'int' object is not iterable",
        ):
            CIterable(x)  # type: ignore
    else:
        assert isinstance(CIterable(iter(x)), CIterable)


@given(x=lists(integers()), index=integers() | floats())
@example(x=[], index=-1)
@example(x=[], index=maxsize + 1)
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


@mark.parametrize("cls", [CIterable, CList, CSet, CFrozenSet])
@given(data=data())
def test_all(cls: Type, data: DataObject) -> None:
    x, _ = data.draw(lists_or_sets(cls, booleans()))
    y = cls(x).all()
    assert isinstance(y, bool)
    assert y == all(x)


@mark.parametrize("cls", [CIterable, CList, CSet, CFrozenSet])
@given(data=data())
def test_any(cls: Type, data: DataObject) -> None:
    x, _ = data.draw(lists_or_sets(cls, booleans()))
    y = cls(x).any()
    assert isinstance(y, bool)
    assert y == any(x)


@mark.parametrize("cls", [CIterable, CList, CSet, CFrozenSet])
@given(data=data())
def test_dict(cls: Type, data: DataObject) -> None:
    x, _ = data.draw(lists_or_sets(cls, tuples(integers(), integers())))
    assert isinstance(cls(x).dict(), CDict)


@mark.parametrize("cls", [CIterable, CList, CSet, CFrozenSet])
@given(data=data(), start=integers())
def test_enumerate(cls: Type, data: DataObject, start: int) -> None:
    x, cast = data.draw(lists_or_sets(cls, integers()))
    y = cls(x).enumerate(start=start)
    assert isinstance(y, cls)
    if cls in {CIterable, CList}:
        assert cast(y) == list(enumerate(x, start=start))


@mark.parametrize("cls", [CIterable, CList, CSet, CFrozenSet])
@given(data=data())
def test_filter(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(lists_or_sets(cls, integers()))
    y = cls(x).filter(is_even)
    assert isinstance(y, cls)
    assert cast(y) == cast(filter(is_even, x))


@mark.parametrize("cls", [CIterable, CList, CSet, CFrozenSet])
@given(data=data())
def test_frozenset(cls: Type, data: DataObject) -> None:
    x, _ = data.draw(lists_or_sets(cls, tuples(integers(), integers())))
    assert isinstance(cls(x).frozenset(), CFrozenSet)


@mark.parametrize("cls", [CIterable, CList, CSet, CFrozenSet])
@given(data=data())
def test_iter(cls: Type, data: DataObject) -> None:
    x, _ = data.draw(lists_or_sets(cls, tuples(integers(), integers())))
    assert isinstance(cls(x).iter(), CIterable)


@mark.parametrize("cls", [CIterable, CList, CSet, CFrozenSet])
@given(data=data())
def test_list(cls: Type, data: DataObject) -> None:
    x, _ = data.draw(lists_or_sets(cls, integers()))
    assert isinstance(cls(x).list(), CList)


@mark.parametrize("cls", [CIterable, CList, CSet, CFrozenSet])
@given(data=data())
def test_map(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(lists_or_sets(cls, integers()))
    y = cls(x).map(neg)
    assert isinstance(y, cls)
    assert cast(y) == cast(map(neg, x))


@mark.parametrize("cls", [CIterable, CList, CSet, CFrozenSet])
@mark.parametrize("func", [max, min])
@given(
    data=data(),
    key_kwargs=just({})
    | (
        just({"key": neg})
        if VERSION in {Version.py36, Version.py37}
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
    x, _ = data.draw(lists_or_sets(cls, integers()))
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


@mark.parametrize("cls", [CIterable, CList, CSet, CFrozenSet])
@given(
    data=data(),
    start=small_ints,
    stop=small_ints | just(sentinel),
    step=small_ints | just(sentinel),
)
def test_range(
    cls: Type, data: DataObject, start: int, stop: Union[int, Sentinel], step: Union[int, Sentinel],
) -> None:
    if step is sentinel:
        assume(stop is not sentinel)
    else:
        assume(step != 0)
    args, _ = drop_sentinel(stop, step)
    x = cls.range(start, *args)
    assert isinstance(x, cls)
    _, cast = data.draw(lists_or_sets(cls, none()))
    assert cast(x) == cast(range(start, *args))


@mark.parametrize("cls", [CIterable, CList, CSet, CFrozenSet])
@given(data=data())
def test_set(cls: Type, data: DataObject) -> None:
    x, _ = data.draw(lists_or_sets(cls, integers()))
    assert isinstance(cls(x).set(), CSet)


@mark.parametrize("cls", [CIterable, CList, CSet, CFrozenSet])
@given(data=data(), key=none() | just(neg), reverse=booleans())
def test_sorted(
    cls: Type, data: DataObject, key: Optional[Callable[[int], int]], reverse: bool,
) -> None:
    x, _ = data.draw(lists_or_sets(cls, integers()))
    y = cls(x).sorted(key=key, reverse=reverse)
    assert isinstance(y, CList)
    assert y == sorted(x, key=key, reverse=reverse)


@mark.parametrize("cls", [CIterable, CList, CSet, CFrozenSet])
@given(data=data(), start=integers() | just(sentinel))
def test_sum(cls: Type, data: DataObject, start: Union[int, Sentinel]) -> None:
    x, _ = data.draw(lists_or_sets(cls, integers()))
    y = cls(x).sum(start=start)
    assert isinstance(y, int)
    args, _ = drop_sentinel(start)
    assert y == sum(x, *args)


@mark.parametrize("cls", [CIterable, CList, CSet, CFrozenSet])
@given(data=data())
def test_tuple(cls: Type, data: DataObject) -> None:
    x, _ = data.draw(lists_or_sets(cls, integers()))
    y = cls(x).tuple()
    assert isinstance(y, tuple)
    if cls in {CIterable, CList}:
        assert y == tuple(x)


@mark.parametrize("cls", [CIterable, CList, CSet, CFrozenSet])
@given(
    data=data(), iterables=lists(lists(integers(), max_size=1000), max_size=10),
)
def test_zip(cls: Type, data: DataObject, iterables: List[List[int]]) -> None:
    x, cast = data.draw(lists_or_sets(cls, integers(), max_size=1000))
    y = cls(x).zip(*iterables)
    assert isinstance(y, cls)
    if cls in {CIterable, CList}:
        assert list(y) == list(zip(x, *iterables))


# functools


@mark.parametrize("cls", [CIterable, CList, CSet, CFrozenSet])
@given(data=data(), initial=integers() | just(sentinel))
def test_reduce(cls: Type, data: DataObject, initial: Union[int, Sentinel]) -> None:
    x, cast = data.draw(lists_or_sets(cls, integers()))
    args, _ = drop_sentinel(initial)
    try:
        y = cls(x).reduce(max, initial=initial)
    except EmptyIterableError:
        with raises(
            TypeError, match=escape("reduce() of empty sequence with no initial value"),
        ):
            reduce(max, x, *args)
    else:
        assert isinstance(y, int)
        assert y == reduce(max, x, *args)


@given(x=tuples(integers(), integers()))
def test_reduce_propagating_type_error(x: Tuple[int, int]) -> None:
    def func(*args: Any) -> NoReturn:
        raise TypeError("Always fail")

    with raises(TypeError, match="Always fail"):
        CIterable(x).reduce(func)


@mark.xfail
@given(
    x=lists(lists(integers())), initial=lists(integers()) | just(sentinel), n=small_ints,
)
def test_reduce_as_iterable(
    x: List[List[int]], initial: Union[List[int], Sentinel], n: int,
) -> None:
    args, _ = drop_sentinel(initial)
    try:
        y = CIterable(x).reduce_as_iterable(lambda x, y: list(chain(x, y)), initial=initial)
    except EmptyIterableError:
        assume(False)
    else:
        assert isinstance(y, CIterable)
        assert list(y[:n]) == list(islice(flatten(chain(x, args)), n))


# itertools


@given(start=integers(), step=integers(), n=small_ints)
def test_count(start: int, step: int, n: int) -> None:
    x = CIterable.count(start=start, step=step)
    assert isinstance(x, CIterable)
    assert list(x[:n]) == list(islice(count(start=start, step=step), n))


@given(x=lists(integers()), n=small_ints)
def test_cycle(x: List[int], n: int) -> None:
    y = CIterable(x).cycle()
    assert isinstance(y, CIterable)
    assert list(y[:n]) == list(islice(cycle(x), n))


@mark.parametrize("cls", [CIterable, CList, CSet, CFrozenSet])
@given(data=data(), x=integers())
def test_repeat(cls: Type, data: DataObject, x: int) -> None:
    if cls is CIterable:
        times = data.draw(small_ints | just(sentinel))
    else:
        times = data.draw(small_ints)
    try:
        y = cls.repeat(x, times=times)
    except OverflowError:
        assume(False)
    else:
        assert isinstance(y, cls)
        _, cast = data.draw(lists_or_sets(cls, none()))
        args, _ = drop_sentinel(times)
        z = repeat(x, *args)
        if (cls is CIterable) and (times is sentinel):
            n = data.draw(small_ints)
            assert cast(y[:n]) == cast(islice(z, n))
        else:
            assert cast(y) == cast(z)


@mark.parametrize("cls", [CIterable, CList, CSet, CFrozenSet])
@given(
    data=data(),
    initial=just({})
    if VERSION in {Version.py36, Version.py37}
    else fixed_dictionaries({"initial": none() | integers()}),
)
def test_accumulate(cls: Type, data: DataObject, initial: Dict[str, Any]) -> None:
    x, cast = data.draw(lists_or_sets(cls, integers()))
    y = cls(x).accumulate(max, **initial)
    assert isinstance(y, cls)
    if cls in {CIterable, CList}:
        assert cast(y) == cast(accumulate(x, max, **initial))


@mark.parametrize("cls", [CIterable, CList, CSet, CFrozenSet])
@given(data=data(), xs=lists(lists(integers(), max_size=1000), max_size=10))
def test_chain(cls: Type, data: DataObject, xs: List[List[int]]) -> None:
    x, cast = data.draw(lists_or_sets(cls, integers(), max_size=1000))
    y = cls(x).chain(*xs)
    assert isinstance(y, cls)
    assert cast(y) == cast(chain(x, *xs))


@mark.parametrize("cls", [CIterable, CList, CSet, CFrozenSet])
@given(data=data())
def test_compress(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(lists_or_sets(cls, integers()))
    selectors = data.draw(lists(booleans(), min_size=len(x), max_size=len(x)))
    y = cls(x).compress(selectors)
    assert isinstance(y, cls)
    if cls in {CIterable, CList}:
        assert cast(y) == cast(compress(x, selectors))


@mark.parametrize("cls", [CIterable, CList, CSet, CFrozenSet])
@given(data=data())
def test_dropwhile(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(lists_or_sets(cls, integers()))
    y = cls(x).dropwhile(is_even)
    assert isinstance(y, cls)
    if cls in {CIterable, CList}:
        assert cast(y) == cast(dropwhile(is_even, x))


@mark.parametrize("cls", [CIterable, CList, CSet, CFrozenSet])
@given(data=data())
def test_filterfalse(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(lists_or_sets(cls, integers()))
    y = cls(x).filterfalse(is_even)
    assert isinstance(y, cls)
    if cls in {CIterable, CList}:
        assert cast(y) == cast(filterfalse(is_even, x))


@mark.parametrize(
    "cls", [CIterable, param(CList, marks=mark.xfail), CSet, CFrozenSet],
)
@given(data=data(), key=none() | just(neg))
def test_groupby(cls: Type, data: DataObject, key: Optional[Callable[[int], int]]) -> None:
    x, cast = data.draw(lists_or_sets(cls, integers()))
    y = cls(x).groupby(key=key)
    assert isinstance(y, cls)
    y = cast(y)
    z = cast(groupby(x, key=key))
    assert len(y) == len(z)
    if cls in {CIterable, CList}:
        for (key_y, group_y), (key_z, group_z) in zip(y, z):
            assert key_y == key_z
            assert isinstance(key_y, int)
            assert isinstance(group_y, cls)
            assert list(group_y) == list(group_z)


@given(
    x=lists(integers()),
    start=small_ints,
    stop=small_ints | just(sentinel),
    step=small_ints | just(sentinel),
)
def test_islice(
    x: List[int], start: int, stop: Union[int, Sentinel], step: Union[int, Sentinel],
) -> None:
    if step is sentinel:
        assume(stop is not sentinel)
    else:
        assume(step != 0)
    args, _ = drop_sentinel(stop, step)
    y = CIterable(x).islice(start, *args)
    assert isinstance(y, CIterable)
    assert list(y) == list(islice(x, start, *args))


@mark.parametrize("cls", [CIterable, CList, CSet, CFrozenSet])
@given(data=data())
def test_starmap(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(lists_or_sets(cls, tuples(integers(), integers())))
    y = cls(x).starmap(max)
    assert isinstance(y, cls)
    assert cast(y) == cast(starmap(max, x))


@mark.parametrize("cls", [CIterable, CList, CSet, CFrozenSet])
@given(data=data())
def test_takewhile(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(lists_or_sets(cls, integers()))
    y = cls(x).takewhile(is_even)
    assert isinstance(y, cls)
    if cls in {CIterable, CList}:
        assert cast(y) == cast(takewhile(is_even, x))


@mark.parametrize("cls", [CIterable, CList, CSet, CFrozenSet])
@given(data=data(), n=small_ints)
def test_tee(cls: Type, data: DataObject, n: int) -> None:
    x, cast = data.draw(lists_or_sets(cls, integers()))
    y = cls(x).tee(n=n)
    assert isinstance(y, cls)
    y = cast(y)
    z = cast(tee(x, n))
    if cls in {CIterable, CList}:
        assert len(y) == len(z)
    for y_i, z_i in zip(y, z):
        if cls is CSet:
            assert isinstance(y_i, CFrozenSet)
        else:
            assert isinstance(y_i, cls)
        assert cast(y_i) == cast(z_i)


@given(
    x=lists(integers()),
    iterables=lists(lists(integers())),
    fillvalue=none() | integers(),
    n=small_ints,
)
def test_zip_longest(
    x: List[int], iterables: List[List[int]], fillvalue: Optional[int], n: int,
) -> None:
    y = CIterable(x).zip_longest(*iterables, fillvalue=fillvalue)
    assert isinstance(y, CIterable)
    assert list(y[:n]) == list(islice(zip_longest(x, *iterables, fillvalue=fillvalue), n))


@given(
    x=lists(integers()), iterables=lists(lists(integers())), n=small_ints,
)
def test_product(x: List[int], iterables: List[List[int]], n: int) -> None:
    y = CIterable(x).product(*iterables)
    assert isinstance(y, CIterable)
    assert list(y[:n]) == list(islice(product(x, *iterables), n))


@given(
    x=lists(integers()), r=none() | small_ints, n=small_ints,
)
def test_permutations(x: List[int], r: Optional[int], n: int) -> None:
    y = CIterable(x).permutations(r=r)
    assert isinstance(y, CIterable)
    assert list(y[:n]) == list(islice(permutations(x, r=r), n))


@given(x=lists(integers()), r=small_ints, n=small_ints)
def test_combinations(x: List[int], r: int, n: int) -> None:
    y = CIterable(x).combinations(r)
    assert isinstance(y, CIterable)
    assert list(y[:n]) == list(islice(combinations(x, r), n))


@given(x=lists(integers()), r=small_ints, n=small_ints)
def test_combinations_with_replacement(x: List[int], r: int, n: int) -> None:
    y = CIterable(x).combinations_with_replacement(r)
    assert isinstance(y, CIterable)
    assert list(y[:n]) == list(islice(combinations_with_replacement(x, r), n))


# itertools-recipes


@mark.parametrize("cls", [CIterable, CList, CSet, CFrozenSet])
@given(data=data(), n=integers(0, maxsize))
def test_take(cls: Type, data: DataObject, n: int) -> None:
    x, cast = data.draw(lists_or_sets(cls, integers()))
    y = cls(x).take(n)
    assert isinstance(y, cls)
    if cls in {CIterable, CList}:
        assert cast(y) == cast(take(n, x))


@given(x=lists(integers()), value=integers())
def test_prepend(x: List[int], value: int) -> None:
    y = CIterable(x).prepend(value)
    assert isinstance(y, CIterable)
    assert list(y) == list(prepend(value, x))


@given(start=integers(), n=small_ints)
def test_tabulate(start: int, n: int) -> None:
    x = CIterable.tabulate(neg, start=start)
    assert isinstance(x, CIterable)
    assert list(islice(x, n)) == list(islice(tabulate(neg, start=start), n))


# multiprocessing


@mark.parametrize("cls", [CIterable, CList, CSet, CFrozenSet])
@given(data=data())
@settings(deadline=None)
def test_pmap(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(lists_or_sets(cls, integers()))
    y = cls(x).pmap(neg, processes=1)
    assert isinstance(y, cls)
    assert cast(y) == cast(map(neg, x))


# pathlib


@mark.parametrize("cls", [CIterable, CList, CSet, CFrozenSet])
@given(data=data(), use_path=booleans())
def test_iterdir(cls: Type, data: DataObject, use_path: bool) -> None:
    x, func = data.draw(
        lists_or_sets(cls, text(alphabet=ascii_lowercase, min_size=1), unique=True),
    )
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
