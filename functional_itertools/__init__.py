"""Python collections in a functional-programming style."""
from functional_itertools.cattrs import CAttrs
from functional_itertools.classes import CDict
from functional_itertools.classes import CFrozenSet
from functional_itertools.classes import CIterable
from functional_itertools.classes import CList
from functional_itertools.classes import CSet
from functional_itertools.errors import EmptyIterableError
from functional_itertools.errors import MultipleElementsError


__version__ = "0.6.0"
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
