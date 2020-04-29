from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Optional
from typing import TypeVar
from typing import Union

from functional_itertools.errors import UnsupportVersionError
from functional_itertools.utilities import Sentinel
from functional_itertools.utilities import sentinel
from functional_itertools.utilities import VERSION
from functional_itertools.utilities import Version


T = TypeVar("T")


if VERSION is Version.py37:
    MAX_MIN_KEY_ANNOTATION = Union[Callable[[T], Any], Sentinel]
    MAX_MIN_KEY_DEFAULT = sentinel
elif VERSION is Version.py38:
    MAX_MIN_KEY_ANNOTATION = Optional[Callable[[T], Any]]
    MAX_MIN_KEY_DEFAULT = None
else:
    raise UnsupportVersionError(VERSION)  # pragma: no cover
