from __future__ import annotations

from functools import partial
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
from itertools import zip_longest
from operator import add
from operator import neg
from sys import maxsize
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Type

from hypothesis import assume
from hypothesis import given
from hypothesis.strategies import booleans
from hypothesis.strategies import data
from hypothesis.strategies import DataObject
from hypothesis.strategies import fixed_dictionaries
from hypothesis.strategies import integers
from hypothesis.strategies import just
from hypothesis.strategies import lists
from hypothesis.strategies import none
from hypothesis.strategies import tuples
from more_itertools import all_equal
from more_itertools import consume
from more_itertools import dotproduct
from more_itertools import flatten
from more_itertools import ncycles
from more_itertools import nth
from more_itertools import padnone
from more_itertools import pairwise
from more_itertools import prepend
from more_itertools import quantify
from more_itertools import repeatfunc
from more_itertools import tabulate
from more_itertools import tail
from more_itertools import take
from pytest import mark

from functional_itertools import CIterable
from functional_itertools import CList
from functional_itertools import CTuple
from functional_itertools.utilities import VERSION
from functional_itertools.utilities import Version
from tests.strategies import CLASSES
from tests.strategies import islice_ints
from tests.strategies import nested_siterables
from tests.strategies import siterables
from tests.strategies import slists
from tests.strategies import small_ints
from tests.test_utilities import is_even


@given(start=integers(), step=integers(), n=islice_ints)
def test_count(start: int, step: int, n: int) -> None:
    x = CIterable.count(start=start, step=step)
    assert isinstance(x, CIterable)
    assert list(x[:n]) == list(islice(count(start=start, step=step), n))


@given(x=slists(integers()), n=islice_ints)
def test_cycle(x: List[int], n: int) -> None:
    y = CIterable(x).cycle()
    assert isinstance(y, CIterable)
    assert list(y[:n]) == list(islice(cycle(x), n))


@mark.parametrize("cls", CLASSES)
@given(data=data(), x=integers(), n=islice_ints)
def test_repeat(cls: Type, data: DataObject, x: int, n: int) -> None:
    if cls is CIterable:
        times = data.draw(none() | small_ints)
    else:
        times = data.draw(small_ints)
    y = cls.repeat(x, times=times)
    assert isinstance(y, cls)
    _, cast = data.draw(siterables(cls, none()))
    z = repeat(x, *(() if times is None else (times,)))
    if (cls is CIterable) and (times is None):
        assert cast(y[:n]) == cast(islice(z, n))
    else:
        assert cast(y) == cast(z)


@mark.parametrize("cls", CLASSES)
@given(
    data=data(),
    initial=just({})
    if VERSION is Version.py37
    else fixed_dictionaries({"initial": none() | integers()}),
)
def test_accumulate(cls: Type, data: DataObject, initial: Dict[str, Any]) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    y = cls(x).accumulate(add, **initial)
    assert isinstance(y, cls)
    if cls in {CIterable, CList, CTuple}:
        assert cast(y) == cast(accumulate(x, add, **initial))


@mark.parametrize("cls", CLASSES)
@given(data=data())
def test_chain(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    xs, _ = data.draw(nested_siterables(cls, integers()))
    y = cls(x).chain(*xs)
    assert isinstance(y, cls)
    assert cast(y) == cast(chain(x, *xs))


@mark.parametrize("cls", CLASSES)
@given(data=data())
def test_compress(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    selectors = data.draw(lists(booleans(), min_size=len(x), max_size=len(x)))
    y = cls(x).compress(selectors)
    assert isinstance(y, cls)
    if cls in {CIterable, CList, CTuple}:
        assert cast(y) == cast(compress(x, selectors))


@mark.parametrize("cls", CLASSES)
@given(data=data())
def test_dropwhile(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    y = cls(x).dropwhile(is_even)
    assert isinstance(y, cls)
    if cls in {CIterable, CList, CTuple}:
        assert cast(y) == cast(dropwhile(is_even, x))


@mark.parametrize("cls", CLASSES)
@given(data=data())
def test_filterfalse(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    y = cls(x).filterfalse(is_even)
    assert isinstance(y, cls)
    if cls in {CIterable, CList, CTuple}:
        assert cast(y) == cast(filterfalse(is_even, x))


@mark.parametrize("cls", [CIterable, CList, CTuple])
@given(data=data(), key=none() | just(neg))
def test_groupby(cls: Type, data: DataObject, key: Optional[Callable[[int], int]]) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    y1, y2 = [cls(x).groupby(key=key) for _ in range(2)]
    assert isinstance(y1, cls)
    z1, z2 = [groupby(x, key=key) for _ in range(2)]
    assert len(cast(y1)) == len(cast(z1))
    for y_i, (kz_i, vz_i) in zip(y2, z2):
        assert isinstance(y_i, CTuple)
        ky_i, vy_i = y_i
        assert isinstance(ky_i, int)
        assert ky_i == kz_i
        assert isinstance(vy_i, cls)
        assert cast(vy_i) == cast(vz_i)


@mark.parametrize("cls", CLASSES)
@given(
    data=data(), start=islice_ints, stop=none() | islice_ints, step=none() | islice_ints,
)
def test_islice(
    cls: Type, data: DataObject, start: int, stop: Optional[int], step: Optional[int],
) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    if step is not None:
        assume((stop is not None) and (step != 0))
    y = cls(x).islice(start, stop, step)
    assert isinstance(y, cls)
    if cls in {CIterable, CList, CTuple}:
        assert cast(y) == cast(
            islice(
                x, start, *(() if stop is None else (stop,)), *(() if step is None else (step,)),
            ),
        )


@mark.parametrize("cls", CLASSES)
@given(data=data())
def test_starmap(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(siterables(cls, tuples(integers(), integers())))
    y = cls(x).starmap(max)
    assert isinstance(y, cls)
    assert cast(y) == cast(starmap(max, x))


@mark.parametrize("cls", CLASSES)
@given(data=data())
def test_takewhile(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    y = cls(x).takewhile(is_even)
    assert isinstance(y, cls)
    if cls in {CIterable, CList, CTuple}:
        assert cast(y) == cast(takewhile(is_even, x))


@mark.parametrize("cls", [CIterable, CList, CTuple])
@given(data=data(), n=small_ints)
def test_tee(cls: Type, data: DataObject, n: int) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    y = cls(x).tee(n=n)
    assert isinstance(y, cls)
    for y_i in y:
        assert isinstance(y_i, cls)
        assert cast(y_i) == cast(x)


@mark.parametrize("cls", CLASSES)
@given(data=data(), fillvalue=none() | integers())
def test_zip_longest(cls: Type, data: DataObject, fillvalue: Optional[int]) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    xs, _ = data.draw(nested_siterables(cls, integers()))
    y = cls(x).zip_longest(*xs, fillvalue=fillvalue)
    assert isinstance(y, cls)
    if cls in {CIterable, CList, CTuple}:
        assert cast(y) == cast(zip_longest(x, *xs, fillvalue=fillvalue))


@mark.parametrize("cls", CLASSES)
@given(data=data(), repeat=integers(1, 3))
def test_product(cls: Type, data: DataObject, repeat: int) -> None:
    x, cast = data.draw(siterables(cls, integers(), max_size=3))
    xs, _ = data.draw(nested_siterables(cls, integers(), max_size=3))
    y = cls(x).product(*xs, repeat=repeat)
    assert isinstance(y, cls)
    assert cast(y) == cast(product(x, *xs, repeat=repeat))


@mark.parametrize("cls", CLASSES)
@given(data=data(), r=none() | small_ints)
def test_permutations(cls: Type, data: DataObject, r: Optional[int]) -> None:
    x, cast = data.draw(siterables(cls, integers(), max_size=3))
    y = cls(x).permutations(r=r)
    assert isinstance(y, cls)
    assert cast(y) == cast(permutations(x, r=r))


@mark.parametrize("cls", CLASSES)
@given(data=data(), r=small_ints)
def test_combinations(cls: Type, data: DataObject, r: int) -> None:
    x, cast = data.draw(siterables(cls, integers(), max_size=3))
    y = cls(x).combinations(r)
    assert isinstance(y, cls)
    assert cast(y) == cast(combinations(x, r))


@mark.parametrize("cls", CLASSES)
@given(data=data(), r=small_ints)
def test_combinations_with_replacement(cls: Type, data: DataObject, r: int) -> None:
    x, cast = data.draw(siterables(cls, integers(), max_size=3))
    y = cls(x).combinations_with_replacement(r)
    assert isinstance(y, cls)
    if cls in {CIterable, CList, CTuple}:
        assert cast(y) == cast(combinations_with_replacement(x, r))


@mark.parametrize("cls", CLASSES)
@given(data=data(), n=integers(0, maxsize))
def test_take(cls: Type, data: DataObject, n: int) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    y = cls(x).take(n)
    assert isinstance(y, cls)
    if cls in {CIterable, CList, CTuple}:
        assert cast(y) == cast(take(n, x))


@mark.parametrize("cls", CLASSES)
@given(data=data(), value=integers())
def test_prepend(cls: Type, data: DataObject, value: int) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    y = cls(x).prepend(value)
    assert isinstance(y, cls)
    assert cast(y) == cast(prepend(value, x))


@given(start=integers(), n=islice_ints)
def test_tabulate(start: int, n: int) -> None:
    x = CIterable.tabulate(neg, start=start)
    assert isinstance(x, CIterable)
    assert list(islice(x, n)) == list(islice(tabulate(neg, start=start), n))


@mark.parametrize("cls", CLASSES)
@given(data=data(), n=small_ints)
def test_tail(cls: Type, data: DataObject, n: int) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    y = cls(x).tail(n)
    assert isinstance(y, cls)
    if cls in {CIterable, CList, CTuple}:
        assert cast(y) == cast(tail(n, x))


@mark.parametrize("cls", CLASSES)
@given(data=data(), n=none() | small_ints)
def test_consume(cls: Type, data: DataObject, n: Optional[int]) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    y = cls(x).consume(n=n)
    assert isinstance(y, cls)
    if cls in {CIterable, CList, CTuple}:
        iter_x = iter(x)
        consume(iter_x, n=n)
        assert cast(y) == cast(iter_x)


@mark.parametrize("cls", CLASSES)
@given(data=data(), n=small_ints, default=none() | small_ints)
def test_nth(cls: Type, data: DataObject, n: int, default: Optional[int]) -> None:
    x, _ = data.draw(siterables(cls, integers()))
    y = cls(x).nth(n, default=default)
    assert isinstance(y, int) or (y is None)
    if cls in {CIterable, CList, CTuple}:
        assert y == nth(x, n, default=default)


@mark.parametrize("cls", CLASSES)
@given(data=data())
def test_all_equal(cls: Type, data: DataObject) -> None:
    x, _ = data.draw(siterables(cls, integers()))
    y = cls(x).all_equal()
    assert isinstance(y, bool)
    assert y == all_equal(x)


@mark.parametrize("cls", CLASSES)
@given(data=data())
def test_quantify(cls: Type, data: DataObject) -> None:
    x, _ = data.draw(siterables(cls, integers()))
    y = cls(x).quantify(pred=is_even)
    assert isinstance(y, int)
    assert y == quantify(x, pred=is_even)


@given(x=slists(integers()), n=islice_ints)
def test_padnone(x: List[int], n: int) -> None:
    y = CIterable(x).padnone()
    assert isinstance(y, CIterable)
    assert list(y[:n]) == list(islice(padnone(x), n))


@mark.parametrize("cls", CLASSES)
@given(data=data(), n=small_ints)
def test_ncycles(cls: Type, data: DataObject, n: int) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    y = cls(x).ncycles(n)
    assert isinstance(y, cls)
    assert cast(y) == cast(ncycles(x, n))


@mark.parametrize("cls", CLASSES)
@given(data=data())
def test_dotproduct(cls: Type, data: DataObject) -> None:
    x, _ = data.draw(siterables(cls, integers()))
    y, _ = data.draw(siterables(cls, integers(), min_size=len(x), max_size=len(x)))
    z = cls(x).dotproduct(y)
    assert isinstance(z, int)
    if cls in {CIterable, CList, CTuple}:
        assert z == dotproduct(x, y)


@mark.parametrize("cls", CLASSES)
@given(data=data())
def test_flatten(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(nested_siterables(cls, integers()))
    y = cls(x).flatten()
    assert isinstance(y, cls)
    assert cast(y) == cast(flatten(x))


@mark.parametrize("cls", CLASSES)
@given(data=data(), n=islice_ints)
def test_repeatfunc(cls: Type, data: DataObject, n: int) -> None:
    add1 = partial(add, 1)
    if cls is CIterable:
        times = data.draw(none() | small_ints)
    else:
        times = data.draw(small_ints)

    y = cls.repeatfunc(add1, times, 0)
    assert isinstance(y, cls)
    _, cast = data.draw(siterables(cls, none()))
    z = repeatfunc(add1, times, 0)
    if (cls is CIterable) and (times is None):
        assert cast(y[:n]) == cast(islice(z, n))
    else:
        assert cast(y) == cast(z)


@mark.parametrize("cls", CLASSES)
@given(data=data())
def test_pairwise(cls: Type, data: DataObject) -> None:
    x, cast = data.draw(siterables(cls, integers()))
    y = cls(x).pairwise()
    assert isinstance(y, cls)
    if cls in {CIterable, CList, CTuple}:
        assert cast(y) == cast(pairwise(x))
