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
from hypothesis.strategies import booleans
from hypothesis.strategies import data
from hypothesis.strategies import DataObject
from hypothesis.strategies import integers
from hypothesis.strategies import just
from hypothesis.strategies import lists
from hypothesis.strategies import none
from hypothesis.strategies import tuples

from functional_itertools import CIterable
from functional_itertools import CTuple
from functional_itertools.utilities import drop_none
from functional_itertools.utilities import drop_sentinel
from functional_itertools.utilities import Sentinel
from functional_itertools.utilities import sentinel
from tests import given
from tests import parametrize
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


@given(
    x=lists(integers()), initial=none() | integers(),
)
@parametrize("case", CASES)
def test_accumulate(x: List[int], initial: Optional[int], case: Case) -> None:
    y = case.cls(x).accumulate(add, initial=initial)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast(
        accumulate(case.cast(x), add, initial=initial),
    )


@given(x=lists(integers()), xs=lists(lists(integers())))
@parametrize("case", CASES)
def test_chain(x: List[int], xs: List[List[int]], case: Case) -> None:
    y = case.cls(x).chain(*xs)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast(chain(case.cast(x), *xs))


@given(x=combinations_x, r=combinations_r)
@parametrize("case", CASES)
def test_combinations(x: List[int], r: int, case: Case) -> None:
    y = case.cls(x).combinations(r)
    assert isinstance(y, case.cls)
    z = list(y)
    for zi in z:
        assert isinstance(zi, CTuple)
    assert case.cast(map(case.cast, z)) == case.cast(
        map(case.cast, combinations(case.cast(x), r)),
    )


@given(x=combinations_x, r=combinations_r)
@parametrize("case", CASES)
def test_combinations_with_replacement(
    x: List[int], r: int, case: Case,
) -> None:
    y = case.cls(x).combinations_with_replacement(r)
    assert isinstance(y, case.cls)
    z = list(y)
    for zi in z:
        assert isinstance(zi, CTuple)
    assert case.cast(map(case.cast, z)) == case.cast(
        map(case.cast, combinations_with_replacement(case.cast(x), r)),
    )


@given(pairs=lists(tuples(integers(), booleans()), min_size=1))
@parametrize("case", CASES)
def test_compress(pairs: List[Tuple[int, bool]], case: Case) -> None:
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


@given(x=lists(integers()))
@parametrize("case", CASES)
def test_dropwhile(x: List[int], case: Case) -> None:
    y = case.cls(x).dropwhile(is_even)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast(dropwhile(is_even, case.cast(x)))


@given(x=lists(integers()))
@parametrize("case", CASES)
def test_filterfalse(x: List[int], case: Case) -> None:
    y = case.cls(x).filterfalse(is_even)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast(filterfalse(is_even, case.cast(x)))


@given(x=lists(integers()), key=none() | just(neg))
@parametrize("case", CASES)
def test_groupby(
    x: List[int], key: Optional[Callable[[int], int]], case: Case,
) -> None:
    y = case.cls(x).groupby(key=key)
    assert isinstance(y, case.cls)
    z = list(y)
    for zi in z:
        assert isinstance(zi, tuple)
        k, v = zi
        assert isinstance(k, int)
        assert isinstance(v, CTuple)
    assert case.cast(z) == case.cast(
        (k, CTuple(v)) for k, v in groupby(case.cast(x), key=key)
    )


@given(
    x=lists(integers()),
    start=integers(0, MAX_SIZE),
    stop=none() | integers(0, MAX_SIZE) | just(sentinel),
    step=none() | integers(1, 10) | just(sentinel),
)
@parametrize("case", CASES)
def test_islice(
    x: List[int],
    start: int,
    stop: Union[Optional[int], Sentinel],
    step: Union[Optional[int], Sentinel],
    case: Case,
) -> None:
    if step is not sentinel:
        assume(stop is not sentinel)
    y = case.cls(x).islice(start, stop=stop, step=step)
    assert isinstance(y, case.cls)
    args, _ = drop_sentinel(stop, step)
    assert case.cast(y) == case.cast(islice(case.cast(x), start, *args))


@parametrize(
    "args, expected",
    [
        ((2,), ["A", "B"]),
        ((2, 4), ["C", "D"]),
        ((2, None), ["C", "D", "E", "F", "G"]),
        ((0, None, 2), ["A", "C", "E", "G"]),
    ],
)
def test_islice_deterministic(
    args: Tuple[Optional[int], ...], expected: List[str],
) -> None:
    assert CIterable("ABCDEFG").islice(*args).list() == expected


@given(x=permutations_x, r=permutations_r)
@parametrize("case", CASES)
def test_permutations(x: List[int], r: Optional[int], case: Case) -> None:
    y = case.cls(x).permutations(r=r)
    assert isinstance(y, case.cls)
    z = list(y)
    for zi in z:
        assert isinstance(zi, CTuple)
    assert case.cast(z) == case.cast(permutations(case.cast(x), r=r))


@given(
    x=product_x, xs=product_xs, repeat=product_repeat,
)
@parametrize("case", CASES)
def test_product(
    x: List[int], xs: List[List[int]], repeat: int, case: Case,
) -> None:
    y = case.cls(x).product(*xs, repeat=repeat)
    assert isinstance(y, case.cls)
    z = list(y)
    for zi in z:
        assert isinstance(zi, CTuple)
    assert case.cast(z) == case.cast(product(x, *xs, repeat=repeat))


@given(data=data(), x=integers(), n=islice_ints)
@parametrize("case", CASES)
def test_repeat(data: DataObject, x: int, n: int, case: Case) -> None:
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


@given(x=lists(tuples(integers(), integers())))
@parametrize("case", CASES)
@parametrize("kwargs", [{}, {"parallel": True, "processes": 1}])
def test_starmap(
    x: List[Tuple[int, int]], case: Case, kwargs: Dict[str, Any],
) -> None:
    y = case.cls(x).starmap(max, **kwargs)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast(starmap(max, x))


@given(x=lists(integers()))
@parametrize("case", CASES)
def test_takewhile(x: List[int], case: Case) -> None:
    y = case.cls(x).takewhile(is_even)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast(takewhile(is_even, case.cast(x)))


@given(x=lists(integers()), n=integers(0, 10))
@parametrize("case", CASES)
def test_tee(x: List[int], n: int, case: Case) -> None:
    y = case.cls(x).tee(n=n)
    assert isinstance(y, CIterable)
    for yi in y:
        assert isinstance(yi, CIterable)


@given(
    x=lists(integers()),
    xs=lists(lists(integers())),
    fillvalue=none() | integers(),
)
@parametrize("case", CASES)
def test_zip_longest(
    x: List[int], xs: List[List[int]], fillvalue: Optional[int], case: Case,
) -> None:
    y = case.cls(x).zip_longest(*xs, fillvalue=fillvalue)
    assert isinstance(y, case.cls)
    z = list(y)
    for zi in z:
        assert isinstance(zi, CTuple)
    assert case.cast(z) == case.cast(
        zip_longest(case.cast(x), *xs, fillvalue=fillvalue),
    )
