from __future__ import annotations

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
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import Type

from hypothesis import given
from hypothesis.strategies import booleans
from hypothesis.strategies import data
from hypothesis.strategies import DataObject
from hypothesis.strategies import fixed_dictionaries
from hypothesis.strategies import integers
from hypothesis.strategies import just
from hypothesis.strategies import none
from hypothesis.strategies import tuples
from pytest import mark

from functional_itertools import CIterable
from functional_itertools import CList
from functional_itertools import CTuple
from functional_itertools.utilities import VERSION
from functional_itertools.utilities import Version
from tests.strategies import CLASSES
from tests.strategies import get_cast
from tests.strategies import islice_ints
from tests.strategies import ORDERED_CLASSES
from tests.strategies import range_args
from tests.strategies import real_iterables
from tests.test_utilities import is_even


@mark.parametrize("cls", CLASSES)
@given(
    x=real_iterables(integers()),
    initial=just({})
    if VERSION is Version.py37
    else fixed_dictionaries({"initial": none() | integers()}),
)
def test_accumulate(cls: Type, x: Iterable[int], initial: Dict[str, Any]) -> None:
    y = cls(x).accumulate(add, **initial)
    assert isinstance(y, CIterable if cls is CIterable else CList)
    if cls in ORDERED_CLASSES:
        assert list(y) == list(accumulate(x, add, **initial))


@mark.parametrize("cls", CLASSES)
@given(x=real_iterables(integers()), xs=real_iterables(real_iterables(integers())))
def test_chain(cls: Type, x: Iterable[int], xs: Iterable[Iterable[int]]) -> None:
    y = cls(x).chain(*xs)
    assert isinstance(y, CIterable if cls is CIterable else CList)
    if cls in ORDERED_CLASSES:
        assert list(y) == list(chain(x, *xs))


@mark.parametrize("cls", CLASSES)
@given(x=real_iterables(integers(), max_size=100), r=integers(0, 10))
def test_combinations(cls: Type, x: Iterable[int], r: int) -> None:
    y = cls(x).combinations(r)
    assert isinstance(y, CIterable if cls is CIterable else CList)
    z = list(y)
    for zi in z:
        assert isinstance(zi, CTuple)
    if cls in ORDERED_CLASSES:
        assert z == list(combinations(x, r))


@mark.parametrize("cls", CLASSES)
@given(x=real_iterables(integers(), max_size=100), r=integers(0, 5))
def test_combinations_with_replacement(cls: Type, x: Iterable[int], r: int) -> None:
    y = cls(x).combinations_with_replacement(r)
    assert isinstance(y, CIterable if cls is CIterable else CList)
    z = list(y)
    for zi in z:
        assert isinstance(zi, CTuple)
    if cls in ORDERED_CLASSES:
        assert z == list(combinations_with_replacement(x, r))


@mark.parametrize("cls", CLASSES)
@given(pairs=real_iterables(tuples(integers(), booleans()), min_size=1))
def test_compress(cls: Type, pairs: Iterable[Tuple[int, bool]]) -> None:
    x, selectors = zip(*pairs)
    y = cls(x).compress(selectors)
    assert isinstance(y, CIterable if cls is CIterable else CList)
    if cls in ORDERED_CLASSES:
        assert list(y) == list(compress(x, selectors))


@given(start=integers(), step=integers(), n=islice_ints)
def test_count(start: int, step: int, n: int) -> None:
    x = CIterable.count(start=start, step=step)
    assert isinstance(x, CIterable)
    assert list(x[:n]) == list(islice(count(start=start, step=step), n))


@given(x=real_iterables(integers()), n=islice_ints)
def test_cycle(x: Iterable[int], n: int) -> None:
    y = CIterable(x).cycle()
    assert isinstance(y, CIterable)
    assert list(y[:n]) == list(islice(cycle(x), n))


@mark.parametrize("cls", CLASSES)
@given(x=real_iterables(integers()))
def test_dropwhile(cls: Type, x: Iterable[int]) -> None:
    y = cls(x).dropwhile(is_even)
    assert isinstance(y, CIterable if cls is CIterable else CList)
    if cls in ORDERED_CLASSES:
        assert list(y) == list(dropwhile(is_even, x))


@mark.parametrize("cls", CLASSES)
@given(x=real_iterables(integers()))
def test_filterfalse(cls: Type, x: Iterable[int]) -> None:
    y = cls(x).filterfalse(is_even)
    assert isinstance(y, CIterable if cls is CIterable else CList)
    if cls in ORDERED_CLASSES:
        assert list(y) == list(filterfalse(is_even, x))


@mark.parametrize("cls", CLASSES)
@given(x=real_iterables(integers()), key=none() | just(neg))
def test_groupby(cls: Type, x: Iterable[int], key: Optional[Callable[[int], int]]) -> None:
    y = cls(x).groupby(key=key)
    assert isinstance(y, CIterable if cls is CIterable else CList)
    for yi in y:
        assert isinstance(yi, tuple)
        k, v = yi
        assert isinstance(k, int)
        assert isinstance(v, CIterable if cls is CIterable else CList)
    if cls in ORDERED_CLASSES:
        assert [(k, list(v)) for k, v in cls(x).groupby(key=key)] == [
            (k, list(v)) for k, v in groupby(x, key=key)
        ]


@mark.parametrize("cls", CLASSES)
@given(x=real_iterables(integers()), args=range_args)
def test_islice(
    cls: Type, x: Iterable[int], args: Tuple[int, Optional[int], Optional[int]],
) -> None:
    y = cls(x).islice(*args)
    assert isinstance(y, CIterable)
    if cls in ORDERED_CLASSES:
        start, stop, step = args
        assert list(y) == list(
            islice(
                x, start, *(() if stop is None else (stop,)), *(() if step is None else (step,)),
            ),
        )


@mark.parametrize("cls", CLASSES)
@given(x=real_iterables(integers(), max_size=5), r=none() | integers(0, 3))
def test_permutations(cls: Type, x: Iterable[int], r: Optional[int]) -> None:
    y = cls(x).permutations(r=r)
    assert isinstance(y, CIterable if cls is CIterable else CList)
    z = list(y)
    for zi in z:
        assert isinstance(zi, CTuple)
    if cls in ORDERED_CLASSES:
        assert z == list(permutations(x, r=r))


@mark.parametrize("cls", CLASSES)
@given(
    x=real_iterables(integers(), max_size=3),
    xs=real_iterables(real_iterables(integers(), max_size=3), max_size=3),
    repeat=integers(0, 3),
)
def test_product(cls: Type, x: Iterable[int], xs: Iterable[Iterable[int]], repeat: int) -> None:
    y = cls(x).product(*xs, repeat=repeat)
    assert isinstance(y, CIterable if cls is CIterable else CList)
    z = list(y)
    for zi in z:
        assert isinstance(zi, CTuple)
    if cls in ORDERED_CLASSES:
        assert z == list(product(x, *xs, repeat=repeat))


@mark.parametrize("cls", CLASSES)
@given(data=data(), x=integers(), n=islice_ints)
def test_repeat(cls: Type, data: DataObject, x: int, n: int) -> None:
    if cls is CIterable:
        times = data.draw(none() | integers(0, 10))
    else:
        times = data.draw(integers(0, 10))
    y = cls.repeat(x, times=times)
    assert isinstance(y, CIterable if cls is CIterable else CList)
    z = repeat(x, *(() if times is None else (times,)))
    if (cls is CIterable) and (times is None):
        assert list(y[:n]) == list(islice(z, n))
    else:
        assert list(y) == list(z)


@mark.parametrize("cls", CLASSES)
@given(x=real_iterables(tuples(integers(), integers())))
def test_starmap(cls: Type, x: Iterable[Tuple[int, int]]) -> None:
    y = cls(x).starmap(max)
    assert isinstance(y, cls)
    cast = get_cast(cls)
    assert cast(y) == cast(starmap(max, x))


@mark.parametrize("cls", CLASSES)
@given(x=real_iterables(integers()))
def test_takewhile(cls: Type, x: Iterable[int]) -> None:
    y = cls(x).takewhile(is_even)
    assert isinstance(y, CIterable if cls is CIterable else CList)
    if cls in ORDERED_CLASSES:
        assert list(y) == list(takewhile(is_even, x))


@mark.parametrize("cls", CLASSES)
@given(x=real_iterables(integers()), n=integers(0, 10))
def test_tee(cls: Type, x: Iterable[int], n: int) -> None:
    y = cls(x).tee(n=n)
    assert isinstance(y, CIterable)
    for yi in y:
        assert isinstance(yi, CIterable)


@mark.parametrize("cls", CLASSES)
@given(
    x=real_iterables(integers()),
    xs=real_iterables(real_iterables(integers())),
    fillvalue=none() | integers(),
)
def test_zip_longest(
    cls: Type, x: Iterable[int], xs: Iterable[Iterable[int]], fillvalue: Optional[int],
) -> None:
    y = cls(x).zip_longest(*xs, fillvalue=fillvalue)
    assert isinstance(y, CIterable if cls is CIterable else CList)
    z = list(y)
    for zi in z:
        assert isinstance(zi, CTuple)
    if cls in ORDERED_CLASSES:
        assert z == list(zip_longest(x, *xs, fillvalue=fillvalue))
