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
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

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
from pytest import mark

from functional_itertools import CIterable
from functional_itertools import CTuple
from functional_itertools.utilities import drop_none
from functional_itertools.utilities import drop_sentinel
from functional_itertools.utilities import Sentinel
from functional_itertools.utilities import sentinel
from functional_itertools.utilities import VERSION
from functional_itertools.utilities import Version
from tests.strategies import Case
from tests.strategies import CASES
from tests.strategies import combinations_r
from tests.strategies import combinations_x
from tests.strategies import islice_ints
from tests.strategies import MAX_SIZE
from tests.strategies import permutations_r
from tests.strategies import permutations_x
from tests.strategies import product_repeat
from tests.strategies import product_x
from tests.strategies import product_xs
from tests.test_utilities import is_even


@mark.parametrize("case", CASES)
@given(
    x=lists(integers()),
    initial=just({})
    if VERSION is Version.py37
    else fixed_dictionaries({"initial": none() | integers()}),
)
def test_accumulate(case: Case, x: List[int], initial: Dict[str, Any]) -> None:
    y = case.cls(x).accumulate(add, **initial)
    assert isinstance(y, case.cls)
    _, kwargs = drop_sentinel(**initial)
    assert case.cast(y) == case.cast(accumulate(case.cast(x), add, **kwargs))


@mark.parametrize("case", CASES)
@given(x=lists(integers()), xs=lists(lists(integers())))
def test_chain(case: Case, x: List[int], xs: List[List[int]]) -> None:
    y = case.cls(x).chain(*xs)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast(chain(case.cast(x), *xs))


@mark.parametrize("case", CASES)
@given(x=combinations_x, r=combinations_r)
def test_combinations(case: Case, x: List[int], r: int) -> None:
    y = case.cls(x).combinations(r)
    assert isinstance(y, case.cls)
    z = list(y)
    for zi in z:
        assert isinstance(zi, CTuple)
    assert case.cast(map(case.cast, z)) == case.cast(map(case.cast, combinations(case.cast(x), r)))


@mark.parametrize("case", CASES)
@given(x=combinations_x, r=combinations_r)
def test_combinations_with_replacement(case: Case, x: List[int], r: int) -> None:
    y = case.cls(x).combinations_with_replacement(r)
    assert isinstance(y, case.cls)
    z = list(y)
    for zi in z:
        assert isinstance(zi, CTuple)
    assert case.cast(map(case.cast, z)) == case.cast(
        map(case.cast, combinations_with_replacement(case.cast(x), r)),
    )


@mark.parametrize("case", CASES)
@given(pairs=lists(tuples(integers(), booleans()), min_size=1))
def test_compress(case: Case, pairs: List[Tuple[int, bool]]) -> None:
    x, selectors = zip(*pairs)
    y = case.cls(x).compress(selectors)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast(compress(case.cast(x), selectors))


@given(start=integers(), step=integers(), n=islice_ints)
def test_count(start: int, step: int, n: int) -> None:
    x = CIterable.count(start=start, step=step)
    assert isinstance(x, CIterable)
    assert list(x[:n]) == list(islice(count(start=start, step=step), n))


@given(x=lists(integers()), n=islice_ints)
def test_cycle(x: List[int], n: int) -> None:
    y = CIterable(x).cycle()
    assert isinstance(y, CIterable)
    assert list(y[:n]) == list(islice(cycle(x), n))


@mark.parametrize("case", CASES)
@given(x=lists(integers()))
def test_dropwhile(case: Case, x: List[int]) -> None:
    y = case.cls(x).dropwhile(is_even)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast(dropwhile(is_even, case.cast(x)))


@mark.parametrize("case", CASES)
@given(x=lists(integers()))
def test_filterfalse(case: Case, x: List[int]) -> None:
    y = case.cls(x).filterfalse(is_even)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast(filterfalse(is_even, case.cast(x)))


@mark.parametrize("case", CASES)
@given(x=lists(integers()), key=none() | just(neg))
def test_groupby(case: Case, x: List[int], key: Optional[Callable[[int], int]]) -> None:
    y = case.cls(x).groupby(key=key)
    assert isinstance(y, case.cls)
    z = list(y)
    for zi in z:
        assert isinstance(zi, tuple)
        k, v = zi
        assert isinstance(k, int)
        assert isinstance(v, CTuple)
    assert case.cast(z) == case.cast((k, CTuple(v)) for k, v in groupby(case.cast(x), key=key))


@mark.parametrize("case", CASES)
@given(
    x=lists(integers()),
    start=integers(0, MAX_SIZE),
    stop=none() | integers(0, MAX_SIZE) | just(sentinel),
    step=none() | integers(1, 10) | just(sentinel),
)
def test_islice(
    case: Case,
    x: List[int],
    start: int,
    stop: Union[Optional[int], Sentinel],
    step: Union[Optional[int], Sentinel],
) -> None:
    if step is not sentinel:
        assume(stop is not sentinel)
    y = case.cls(x).islice(start, stop=stop, step=step)
    assert isinstance(y, case.cls)
    args, _ = drop_sentinel(stop, step)
    assert case.cast(y) == case.cast(islice(case.cast(x), start, *args))


@mark.parametrize(
    "args, expected",
    [
        ((2,), ["A", "B"]),
        ((2, 4), ["C", "D"]),
        ((2, None), ["C", "D", "E", "F", "G"]),
        ((0, None, 2), ["A", "C", "E", "G"]),
    ],
)
def test_islice_deterministic(args: Tuple[Optional[int], ...], expected: List[str]) -> None:
    assert CIterable("ABCDEFG").islice(*args).list() == expected


@mark.parametrize("case", CASES)
@given(x=permutations_x, r=permutations_r)
def test_permutations(case: Case, x: List[int], r: Optional[int]) -> None:
    y = case.cls(x).permutations(r=r)
    assert isinstance(y, case.cls)
    z = list(y)
    for zi in z:
        assert isinstance(zi, CTuple)
    assert case.cast(z) == case.cast(permutations(case.cast(x), r=r))


@mark.parametrize("case", CASES)
@given(
    x=product_x, xs=product_xs, repeat=product_repeat,
)
def test_product(case: Case, x: List[int], xs: List[List[int]], repeat: int) -> None:
    y = case.cls(x).product(*xs, repeat=repeat)
    assert isinstance(y, case.cls)
    z = list(y)
    for zi in z:
        assert isinstance(zi, CTuple)
    assert case.cast(z) == case.cast(product(x, *xs, repeat=repeat))


@mark.parametrize("case", CASES)
@given(data=data(), x=integers(), n=islice_ints)
def test_repeat(case: Case, data: DataObject, x: int, n: int) -> None:
    if case.cls is CIterable:
        times = data.draw(none() | integers(0, 10))
    else:
        times = data.draw(integers(0, 10))
    y = case.cls.repeat(x, times=times)
    assert isinstance(y, case.cls)
    args, _ = drop_none(times)
    z = repeat(x, *args)
    if (case.cls is CIterable) and (times is None):
        assert case.cast(y[:n]) == case.cast(islice(z, n))
    else:
        assert case.cast(y) == case.cast(z)


@mark.parametrize("case", CASES)
@mark.parametrize("kwargs", [{}, {"parallel": True, "processes": 1}])
@given(x=lists(tuples(integers(), integers())))
def test_starmap(case: Case, x: List[Tuple[int, int]], kwargs: Dict[str, Any]) -> None:
    y = case.cls(x).starmap(max, **kwargs)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast(starmap(max, x))


@mark.parametrize("case", CASES)
@given(x=lists(integers()))
def test_takewhile(case: Case, x: List[int]) -> None:
    y = case.cls(x).takewhile(is_even)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast(takewhile(is_even, case.cast(x)))


@mark.parametrize("case", CASES)
@given(x=lists(integers()), n=integers(0, 10))
def test_tee(case: Case, x: List[int], n: int) -> None:
    y = case.cls(x).tee(n=n)
    assert isinstance(y, CIterable)
    for yi in y:
        assert isinstance(yi, CIterable)


@mark.parametrize("case", CASES)
@given(
    x=lists(integers()), xs=lists(lists(integers())), fillvalue=none() | integers(),
)
def test_zip_longest(
    case: Case, x: List[int], xs: List[List[int]], fillvalue: Optional[int],
) -> None:
    y = case.cls(x).zip_longest(*xs, fillvalue=fillvalue)
    assert isinstance(y, case.cls)
    z = list(y)
    for zi in z:
        assert isinstance(zi, CTuple)
    assert case.cast(z) == case.cast(zip_longest(case.cast(x), *xs, fillvalue=fillvalue))
