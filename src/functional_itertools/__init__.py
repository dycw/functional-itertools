"""Python collections in a functional-programming style."""
from __future__ import annotations

from functional_itertools.cattrs import CAttrs
from functional_itertools.classes import CDict
from functional_itertools.classes import CFrozenSet
from functional_itertools.classes import CIterable
from functional_itertools.classes import CList
from functional_itertools.classes import CSet
from functional_itertools.errors import EmptyIterableError
from functional_itertools.errors import MultipleElementsError

from ._version import get_versions


__all__ = [
    "CAttrs",
    "CDict",
    "CFrozenSet",
    "CIterable",
    "CList",
    "CSet",
    "EmptyIterableError",
    "MultipleElementsError",
]
__version__ = get_versions()["version"]
del get_versions
