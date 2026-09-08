"""Deliberately safe errors which may be shown in the local UI."""


class ImageAPIError(RuntimeError):
    """Never construct this with unfiltered HTTP/SDK exception text."""
