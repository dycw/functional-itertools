from __future__ import annotations

from itertools import islice
from operator import le
from operator import neg
from re import escape
from sys import maxsize
from typing import Callable
from typing import cast
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from hypothesis.strategies import fixed_dictionaries
from hypothesis.strategies import integers
from hypothesis.strategies import just
from hypothesis.strategies import lists
from hypothesis.strategies import none
from more_itertools import chunked
from more_itertools import distribute
from more_itertools import divide
from more_itertools import filter_except
from more_itertools import first
from more_itertools import interleave
from more_itertools import interleave_longest
from more_itertools import intersperse
from more_itertools import iterate
from more_itertools import last
from more_itertools import lstrip
from more_itertools import map_except
from more_itertools import nth_or_last
from more_itertools import one
from more_itertools import only
from more_itertools import rstrip
from more_itertools import split_after
from more_itertools import split_at
from more_itertools import split_before
from more_itertools import split_into
from more_itertools import split_when
from more_itertools import strip
from more_itertools import unzip
from pytest import raises

from functional_itertools import CIterable
from functional_itertools import CTuple
from functional_itertools import EmptyIterableError
from functional_itertools import MultipleElementsError
from functional_itertools.utilities import drop_sentinel
from functional_itertools.utilities import Sentinel
from functional_itertools.utilities import sentinel
from tests import given
from tests import parametrize
from tests.strategies import Case
from tests.strategies import CASES
from tests.strategies import islice_ints
from tests.test_utilities import is_even


@given(x=lists(integers(), max_size=1000), n=integers(0, 10))
@parametrize("case", CASES)
def test_chunked(x: List[int], n: int, case: Case) -> None:
    y = case.cls(x).chunked(n)
    assert isinstance(y, case.cls)
    z = list(y)
    for zi in z:
        assert isinstance(zi, CTuple)
    assert case.cast(map(case.cast, z)) == case.cast(
        map(case.cast, chunked(case.cast(x), n)),
    )


@given(x=lists(integers(), max_size=1000), n=integers(1, 10))
@parametrize("case", CASES)
def test_distribute(x: List[int], n: int, case: Case) -> None:
    y = case.cls(x).distribute(n)
    assert isinstance(y, case.cls)
    z = list(y)
    for zi in z:
        assert isinstance(zi, CTuple)
    assert case.cast(map(case.cast, z)) == case.cast(
        map(case.cast, distribute(n, case.cast(x))),
    )


@given(x=lists(integers(), max_size=1000), n=integers(1, 10))
@parametrize("case", CASES)
def test_divide(x: List[int], n: int, case: Case) -> None:
    y = case.cls(x).divide(n)
    assert isinstance(y, case.cls)
    z = list(y)
    for zi in z:
        assert isinstance(zi, CTuple)
    assert case.cast(map(case.cast, z)) == case.cast(
        map(case.cast, divide(n, case.cast(x))),
    )


@given(x=lists(integers()))
@parametrize("case", CASES)
def test_filter_except(x: List[int], case: Case) -> None:
    def func(n: int) -> int:
        if is_even(n):
            return True
        else:
            raise ValueError("'n' must be even")

    y = case.cls(x).filter_except(func, ValueError)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast(filter_except(func, x, ValueError))


@given(
    x=lists(integers()),
    default=just({}) | fixed_dictionaries({"default": integers()}),
)
@parametrize("case", CASES)
@parametrize("func", [first, last])
def test_first_and_last(
    default: Dict[str, int], case: Case, func: Callable[..., int], x: List[int],
) -> None:
    name = func.__name__
    try:
        y = getattr(case.cls(x), name)(**default)
    except EmptyIterableError:
        with raises(
            ValueError,
            match=escape(
                f"{name}() was called on an empty iterable, and no default "
                f"value was provided.",
            ),
        ):
            func(case.cast(x), **default)
    else:
        assert isinstance(y, int)
        assert y == func(case.cast(x), **default)


@given(x=lists(integers()), xs=lists(lists(integers())))
@parametrize("case", CASES)
def test_interleave(x: List[int], xs: List[List[int]], case: Case) -> None:
    y = case.cls(x).interleave(*xs)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast(interleave(case.cast(x), *xs))


@given(x=lists(integers()), xs=lists(lists(integers())))
@parametrize("case", CASES)
def test_interleave_longest(
    x: List[int], xs: List[List[int]], case: Case,
) -> None:
    y = case.cls(x).interleave_longest(*xs)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast(interleave_longest(case.cast(x), *xs))


@given(e=integers(), x=lists(integers()), n=integers(1, maxsize))
@parametrize("case", CASES)
def test_intersperse(e: int, x: List[int], n: int, case: Case) -> None:
    y = case.cls(x).intersperse(e, n=n)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast(intersperse(e, case.cast(x), n=n))


@given(start=integers(), n=islice_ints)
def test_iterate(start: int, n: int) -> None:
    y = CIterable.iterate(neg, start)
    assert isinstance(y, CIterable)
    assert list(y[:n]) == list(
        cast(Iterable[int], islice(iterate(neg, start), n)),
    )


@given(x=lists(integers()))
@parametrize("case", CASES)
def test_lstrip(x: List[int], case: Case) -> None:
    y = case.cls(x).lstrip(is_even)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast(lstrip(case.cast(x), is_even))


@given(x=lists(integers()))
@parametrize("case", CASES)
def test_map_except(x: List[int], case: Case) -> None:
    def func(n: int) -> int:
        if n % 2 == 0:
            return neg(n)
        else:
            raise ValueError("'n' must be even")

    y = case.cls(x).map_except(func, ValueError)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast(map_except(func, x, ValueError))


@given(
    x=lists(integers()),
    n=integers(0, maxsize),
    default=integers() | just(sentinel),
)
@parametrize("case", CASES)
def test_nth_or_last(
    x: List[int], n: int, default: Union[int, Sentinel], case: Case,
) -> None:
    _, kwargs = drop_sentinel(default=default)
    try:
        y = case.cls(x).nth_or_last(n, default=default)
    except ValueError:
        with raises(
            ValueError,
            match=escape(
                "last() was called on an empty iterable, and no default value "
                "was provided.",
            ),
        ):
            nth_or_last(x, n, **kwargs)
    else:
        assert isinstance(y, int)
        assert y == nth_or_last(case.cast(x), n, default=default)


@given(x=lists(integers()))
@parametrize("case", CASES)
def test_one(x: List[int], case: Case) -> None:
    try:
        y = case.cls(x).one()
    except EmptyIterableError:
        with raises(
            ValueError, match=escape("too few items in iterable (expected 1)"),
        ):
            one(case.cast(x))
    except MultipleElementsError:
        with raises(
            ValueError,
            match=r"Expected exactly one item in iterable, but got -?\d+, "
            r"-?\d+, and perhaps more",
        ):
            one(case.cast(x))
    else:
        assert isinstance(y, int)
        assert y == one(case.cast(x))


@given(
    x=lists(integers()), default=none() | integers(),
)
@parametrize("case", CASES)
def test_only(x: List[int], default: Optional[int], case: Case) -> None:
    try:
        y = case.cls(x).only(default=default)
    except EmptyIterableError:
        with raises(
            ValueError, match=escape("too few items in iterable (expected 1)"),
        ):
            only(case.cast(x), default=default)
    except MultipleElementsError:
        with raises(
            ValueError,
            match=r"Expected exactly one item in iterable, but got -?\d+, "
            r"-?\d+, and perhaps more",
        ):
            only(case.cast(x), default=default)
    else:
        assert isinstance(y, int) or (y is None)
        assert y == only(case.cast(x), default=default)


@given(x=lists(integers()))
@parametrize("case", CASES)
def test_rstrip(x: List[int], case: Case) -> None:
    y = case.cls(x).rstrip(is_even)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast(rstrip(case.cast(x), is_even))


@given(x=lists(integers()))
@parametrize("case", CASES)
def test_split_after(x: List[int], case: Case) -> None:
    y = case.cls(x).split_after(neg)
    assert isinstance(y, case.cls)
    z = list(y)
    for zi in z:
        assert isinstance(zi, CTuple)
    assert case.cast(z) == case.cast(
        map(CTuple, split_after(case.cast(x), neg)),
    )


@given(x=lists(integers()))
@parametrize("case", CASES)
def test_split_at(x: List[int], case: Case) -> None:
    y = case.cls(x).split_at(neg)
    assert isinstance(y, case.cls)
    z = list(y)
    for zi in z:
        assert isinstance(zi, CTuple)
    assert case.cast(z) == case.cast(map(CTuple, split_at(case.cast(x), neg)))


@given(x=lists(integers()))
@parametrize("case", CASES)
def test_split_before(x: List[int], case: Case) -> None:
    y = case.cls(x).split_before(neg)
    assert isinstance(y, case.cls)
    z = list(y)
    for zi in z:
        assert isinstance(zi, CTuple)
    assert case.cast(z) == case.cast(
        map(CTuple, split_before(case.cast(x), neg)),
    )


@given(x=lists(integers()), sizes=lists(integers(0, maxsize)))
@parametrize("case", CASES)
def test_split_into(x: List[int], sizes: List[int], case: Case) -> None:
    y = case.cls(x).split_into(sizes)
    assert isinstance(y, case.cls)
    z = list(y)
    for zi in z:
        assert isinstance(zi, CTuple)
    assert case.cast(z) == case.cast(
        map(CTuple, split_into(case.cast(x), sizes)),
    )


@given(x=lists(integers()))
@parametrize("case", CASES)
def test_split_when(x: List[int], case: Case) -> None:
    y = case.cls(x).split_when(le)
    assert isinstance(y, case.cls)
    z = list(y)
    for zi in z:
        assert isinstance(zi, CTuple)
    assert case.cast(z) == case.cast(map(CTuple, split_when(case.cast(x), le)))


@given(x=lists(integers()))
@parametrize("case", CASES)
def test_strip(x: List[int], case: Case) -> None:
    y = case.cls(x).strip(is_even)
    assert isinstance(y, case.cls)
    assert case.cast(y) == case.cast(strip(case.cast(x), is_even))


@given(
    x=integers(0, 10).flatmap(
        lambda x: lists(lists(integers(), min_size=x, max_size=x).map(tuple)),
    ),
)
@parametrize("case", CASES)
def test_unzip(x: List[Tuple[int, ...]], case: Case) -> None:
    y = case.cls(x).unzip()
    assert isinstance(y, case.cls)
    z = list(y)
    for zi in z:
        assert isinstance(zi, CTuple)
    assert case.cast(z) == case.cast(map(CTuple, unzip(case.cast(x))))
