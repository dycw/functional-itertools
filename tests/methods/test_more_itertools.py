from __future__ import annotations

from functools import partial
from operator import le

from hypothesis import given
from hypothesis.strategies import data
from hypothesis.strategies import DataObject
from hypothesis.strategies import integers
from more_itertools import distribute
from more_itertools import divide
from more_itertools import ichunked
from pytest import mark

from functional_itertools import CIterable
from functional_itertools import CList
from functional_itertools import CTuple
from tests.strategies import nested_siterables
from tests.strategies import small_ints


@mark.parametrize("case", [CIterable, CList, CTuple])
@given(data=data(), n=small_ints.filter(partial(le, 1)))
def test_chunked(case: Case, data: DataObject, n: int) -> None:
    x, cast = data.draw(nested_siterables(cls, integers()))
    y1, y2 = [cls(x).chunked(n) for _ in range(2)]
    assert isinstance(y1, case.cls)
    z1, z2 = [ichunked(x, n) for _ in range(2)]
    assert len(cast(y1)) == len(cast(z1))
    for y_i, z_i in zip(y2, z2):
        assert isinstance(y_i, case.cls)
        assert cast(y_i) == cast(z_i)


@mark.parametrize("case", [CIterable, CList, CTuple])
@given(data=data(), n=small_ints.filter(partial(le, 1)))
def test_distribute(case: Case, data: DataObject, n: int) -> None:
    x, cast = data.draw(nested_siterables(cls, integers()))
    y1, y2 = [cls(x).distribute(n) for _ in range(2)]
    assert isinstance(y1, case.cls)
    z1, z2 = [distribute(n, x) for _ in range(2)]
    assert len(cast(y1)) == len(cast(z1))
    for y_i, z_i in zip(y2, z2):
        assert isinstance(y_i, case.cls)
        assert cast(y_i) == cast(z_i)


@mark.parametrize("case", [CIterable, CList, CTuple])
@given(data=data(), n=small_ints.filter(partial(le, 1)))
def test_divide(case: Case, data: DataObject, n: int) -> None:
    x, cast = data.draw(nested_siterables(cls, integers()))
    y1, y2 = [cls(x).divide(n) for _ in range(2)]
    assert isinstance(y1, case.cls)
    z1, z2 = [divide(n, x) for _ in range(2)]
    assert len(cast(y1)) == len(cast(z1))
    for y_i, z_i in zip(y2, z2):
        assert isinstance(y_i, case.cls)
        assert cast(y_i) == cast(z_i)
