from __future__ import annotations

from typing import Iterable
from typing import Type

import more_itertools
from hypothesis import given
from hypothesis.strategies import integers
from pytest import mark

from functional_itertools import CIterable
from functional_itertools import CList
from tests.strategies import CLASSES
from tests.strategies import real_iterables


@mark.parametrize("cls", CLASSES)
@given(x=real_iterables(integers(), max_size=1000), n=integers(0, 10))
def test_chunked(cls: Type, x: Iterable[int], n: int) -> None:
    y = cls(x).chunked(n)
    expected = CIterable if cls is CIterable else CList
    assert isinstance(y, expected)
    for yi, zi in zip(y, more_itertools.chunked(cast(x), n)):
        assert isinstance(yi, expected)
        assert list(yi) == list(zi)


@mark.parametrize("cls", CLASSES)
@mark.parametrize("name", ["distribute", "divide"])
@given(x=real_iterables(integers(), max_size=1000), n=integers(1, 10))
def test_distribute_and_divide(cls: Type, name: str, x: Iterable[int], n: int) -> None:
    y = getattr(cls(x), name)(n)
    expected = CIterable if cls is CIterable else CList
    assert isinstance(y, expected)
    for yi, zi in zip(y, getattr(more_itertools, name)(n, cast(x))):
        assert isinstance(yi, expected)
        assert list(yi) == list(zi)
