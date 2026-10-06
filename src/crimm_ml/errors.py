"""Exceptions crimm_ml raises on purpose.

Everything derives from `CrimmMLError`, so a caller -- or the web service --
can catch the library's own failures in one place and tell them apart from
bugs. `ContigError` is also a `ValueError`, because a bad contig is a bad
argument, and existing `except ValueError` handlers keep working.
"""

__all__ = ["CrimmMLError", "ContigError", "AssetError"]


class CrimmMLError(Exception):
    """Base class for errors raised by crimm_ml."""


class ContigError(CrimmMLError, ValueError):
    """The contig string is malformed or cannot be applied to the structure."""


class AssetError(CrimmMLError):
    """A model or schedule file is missing, or fails its checksum."""
