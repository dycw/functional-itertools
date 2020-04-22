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

from functional_itertools import CFrozenSet
from functional_itertools import CIterable
from functional_itertools import CList
from functional_itertools import CSet
from functional_itertools import CTuple
from functional_itertools.utilities import VERSION
from functional_itertools.utilities import Version
from tests.strategies import Case
from tests.strategies import CASES
from tests.strategies import islice_ints
from tests.strategies import nested_siterables
from tests.strategies import range_args
from tests.strategies import real_iterables
from tests.strategies import siterables
from tests.strategies import small_ints
from tests.test_utilities import is_even


@mark.parametrize("case", CASES)
@given(
    x=real_iterables(integers()),
    initial=just({})
    if VERSION is Version.py37
    else fixed_dictionaries({"initial": none() | integers()}),
)
def test_accumulate(case: Case, x: Iterable[int], initial: Dict[str, Any]) -> None:
    y = case.cls(x).accumulate(add, **initial)
    assert isinstance(y, case.cls)
    if case.cls in {CIterable, CList, CTuple}:
        assert case.cast(y) == case.cast(accumulate(x, add, **initial))


@mark.parametrize("case", CASES)
@given(x=real_iterables(integers()), xs=real_iterables(real_iterables(integers())))
def test_chain(case: Case, x: Iterable[int], xs: Iterable[Iterable[int]]) -> None:
    y = case.cls(x).chain(*xs)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast(chain(x, *xs))


@mark.parametrize("case", CASES)
@given(x=real_iterables(integers(), max_size=100), r=integers(0, 10))
def test_combinations(case: Case, x: Iterable[int], r: int) -> None:
    y = case.cls(x).combinations(r)
    assert isinstance(y, case.cls)
    z = case.cast(y)
    for zi in z:
        assert isinstance(zi, CTuple)
    if case.cls in {CIterable, CList, CTuple}:
        assert z == case.cast(combinations(x, r))


@mark.parametrize("case", CASES)
@given(x=real_iterables(integers(), max_size=100), r=integers(0, 5))
def test_combinations_with_replacement(case: Case, x: Iterable[int], r: int) -> None:
    y = case.cls(x).combinations_with_replacement(r)
    assert isinstance(y, case.cls)
    z = case.cast(y)
    for zi in z:
        assert isinstance(zi, CTuple)
    if case.cls in {CIterable, CList, CTuple}:
        assert z == case.cast(combinations_with_replacement(x, r))


@mark.parametrize("case", CASES)
@given(pairs=real_iterables(tuples(integers(), booleans()), min_size=1))
def test_compress(case: Case, pairs: Iterable[Tuple[int, bool]]) -> None:
    x, selectors = zip(*pairs)
    y = case.cls(x).compress(selectors)
    assert isinstance(y, case.cls)
    if case.cls in {CIterable, CList, CTuple}:
        assert case.cast(y) == case.cast(compress(x, selectors))


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


@mark.parametrize("case", CASES)
@given(x=real_iterables(integers()))
def test_dropwhile(case: Case, x: Iterable[int]) -> None:
    y = case.cls(x).dropwhile(is_even)
    assert isinstance(y, case.cls)
    if case.cls in {CIterable, CList, CTuple}:
        assert case.cast(y) == case.cast(dropwhile(is_even, x))


@mark.parametrize("case", CASES)
@given(x=real_iterables(integers()))
def test_filterfalse(case: Case, x: Iterable[int]) -> None:
    y = case.cls(x).filterfalse(is_even)
    assert isinstance(y, case.cls)
    if case.cls in {CIterable, CList, CTuple}:
        assert case.cast(y) == case.cast(filterfalse(is_even, x))


@mark.parametrize("case", CASES)
@given(x=real_iterables(integers()), key=none() | just(neg))
def test_groupby(case: Case, x: Iterable[int], key: Optional[Callable[[int], int]]) -> None:
    y = case.cls(x).groupby(key=key)
    assert isinstance(y, case.cls)
    for zi in case.cast(y):
        assert isinstance(zi, tuple)
        zi0, zi1 = zi
        assert isinstance(zi0, int)
        if case.cls is CSet:
            assert isinstance(zi1, CFrozenSet)
        else:
            assert isinstance(zi1, case.cls)
    if case.cls in {CIterable, CList, CTuple}:
        for (zi0, zi1), (wi0, wi1) in zip(case.cls(x).groupby(key=key), groupby(x, key=key)):
            assert zi0 == wi0
            assert case.cast(zi1) == case.cast(wi1)


@mark.parametrize("case", CASES)
@given(x=real_iterables(integers()), args=range_args)
def test_islice(
    case: Case, x: Iterable[int], args: Tuple[int, Optional[int], Optional[int]],
) -> None:
    y = case.cls(x).islice(*args)
    assert isinstance(y, case.cls)
    if case.cls in {CIterable, CList, CTuple}:
        start, stop, step = args
        assert case.cast(y) == case.cast(
            islice(
                x, start, *(() if stop is None else (stop,)), *(() if step is None else (step,)),
            ),
        )


@mark.parametrize("case", CASES)
@given(x=real_iterables(integers(), max_size=5), r=none() | integers(0, 3))
def test_permutations(case: Case, x: Iterable[int], r: Optional[int]) -> None:
    y = case.cls(x).permutations(r=r)
    assert isinstance(y, case.cls)
    z = case.cast(y)
    for zi in z:
        assert isinstance(zi, CTuple)
    if case.cls in {CIterable, CList, CTuple}:
        assert z == case.cast(permutations(x, r=r))


@mark.parametrize("case", CASES)
@given(
    x=real_iterables(integers(), max_size=3),
    xs=real_iterables(real_iterables(integers(), max_size=3), max_size=3),
    repeat=integers(0, 3),
)
def test_product(case: Case, x: Iterable[int], xs: Iterable[Iterable[int]], repeat: int) -> None:
    y = case.cls(x).product(*xs, repeat=repeat)
    assert isinstance(y, case.cls)
    z = case.cast(y)
    for zi in z:
        assert isinstance(zi, CTuple)
    if case.cls in {CIterable, CList, CTuple}:
        assert z == case.cast(product(x, *xs, repeat=repeat))


@mark.parametrize("case", CASES)
@given(data=data(), x=integers(), n=islice_ints)
def test_repeat(case: Case, data: DataObject, x: int, n: int) -> None:
    if case.cls is CIterable:
        times = data.draw(none() | small_ints)
    else:
        times = data.draw(small_ints)
    y = case.cls.repeat(x, times=times)
    assert isinstance(y, case.cls)
    _, case.cast = data.draw(siterables(case.cls, none()))
    z = repeat(x, *(() if times is None else (times,)))
    if (case.cls is CIterable) and (times is None):
        assert case.cast(y[:n]) == case.cast(islice(z, n))
    else:
        assert case.cast(y) == case.cast(z)


@mark.parametrize("case", CASES)
@given(data=data())
def test_starmap(case: Case, data: DataObject) -> None:
    x, case.cast = data.draw(siterables(case.cls, tuples(integers(), integers())))
    y = case.cls(x).starmap(max)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast(starmap(max, x))


@mark.parametrize("case", CASES)
@given(data=data())
def test_takewhile(case: Case, data: DataObject) -> None:
    x, case.cast = data.draw(siterables(case.cls, integers()))
    y = case.cls(x).takewhile(is_even)
    assert isinstance(y, case.cls)
    if case.cls in {CIterable, CList, CTuple}:
        assert case.cast(y) == case.cast(takewhile(is_even, x))


@mark.parametrize("case", [CIterable, CList, CTuple])
@given(data=data(), n=small_ints)
def test_tee(case: Case, data: DataObject, n: int) -> None:
    x, case.cast = data.draw(siterables(case.cls, integers()))
    y = case.cls(x).tee(n=n)
    assert isinstance(y, case.cls)
    for y_i in y:
        assert isinstance(y_i, case.cls)
        assert case.cast(y_i) == case.cast(x)


@mark.parametrize("case", [CIterable, CList, CTuple])
@given(data=data(), fillvalue=none() | integers())
def test_zip_longest(case: Case, data: DataObject, fillvalue: Optional[int]) -> None:
    x, case.cast = data.draw(siterables(case.cls, integers()))
    xs, _ = data.draw(nested_siterables(case.cls, integers()))
    y1, y2 = [cls(x).zip_longest(*xs, fillvalue=fillvalue) for _ in range(2)]
    assert isinstance(y1, case.cls)
    z1, z2 = [zip_longest(x, *xs, fillvalue=fillvalue) for _ in range(2)]
    assert len(case.cast(y1)) == len(case.cast(z1))
    for y_i, z_i in zip(y2, z2):
        assert isinstance(y_i, case.cls)
        if case.cls in {CIterable, CList, CTuple}:
            assert case.cast(y_i) == case.cast(z_i)
