from __future__ import annotations

from typing import Callable
from typing import cast
from typing import TypeVar

import hypothesis
from pytest import mark


TestLike = TypeVar("TestLike", bound=Callable[..., None])
example = cast(
    Callable[..., Callable[[TestLike], TestLike]], hypothesis.example,
)
given = cast(Callable[..., Callable[[TestLike], TestLike]], hypothesis.given)
parametrize = cast(
    Callable[..., Callable[[TestLike], TestLike]], mark.parametrize,
)
