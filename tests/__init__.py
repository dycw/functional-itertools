from __future__ import annotations

from typing import Callable
from typing import TypeVar

import hypothesis
from pytest import mark


TestLike = TypeVar("TestLike", bound=Callable[..., None])
example: Callable[..., Callable[[TestLike], TestLike]] = hypothesis.example
given: Callable[..., Callable[[TestLike], TestLike]] = hypothesis.given
parametrize: Callable[..., Callable[[TestLike], TestLike]] = mark.parametrize
