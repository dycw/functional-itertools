from __future__ import annotations

from enum import auto
from enum import Enum
from sys import version_info
from typing import Type
from warnings import warn

from functional_itertools.errors import UnsupportVersionError


# sentinel


class Sentinel:
    def __repr__(self: Sentinel) -> str:
        return "<sentinel>"

    __str__ = __repr__


sentinel = Sentinel()


# version


class Version(Enum):
    py37 = auto()
    py38 = auto()


def _get_version() -> Version:
    major, minor, *_ = version_info
    if major != 3:  # pragma: no cover
        raise RuntimeError(f"Expected Python 3; got {major}")
    mapping = {7: Version.py37, 8: Version.py38}
    try:
        return mapping[minor]
    except KeyError:  # pragma: no cover
        raise UnsupportVersionError(f"Expected Python 3.6-3.8; got 3.{minor}") from None


VERSION = _get_version()


# warn


def warn_non_functional(cls: Type, incorrect: str, suggestion: str) -> None:
    name = cls.__name__
    warn(
        f"{name}.{incorrect} is a non-functional method, did you mean {name}.{suggestion} instead?",
    )
