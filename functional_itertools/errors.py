from __future__ import annotations


class EmptyIterableError(ValueError):
    """Raised when an Iterable is unexpectedly empty."""


class MultipleElementsError(ValueError):
    """Raised when an Iterable unexpectedly contains more than 1 element."""


class StopArgumentMissing(ValueError):
    """Raised when the 'stop' is missing."""


class UnsupportVersionError(RuntimeError):
    """Raised when the version of Python is unsupported."""
