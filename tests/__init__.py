from __future__ import annotations

from typing import Callable
from typing import TypeVar

import hypothesis
from pytest import mark


TestLike = TypeVar("TestLike", bound=Callable[..., None])
given: Callable[
    ..., Callable[[TestLike], TestLike],
] = hypothesis.given  # type: ignore
parametrize: Callable[..., Callable[[TestLike], TestLike]] = mark.parametrize
