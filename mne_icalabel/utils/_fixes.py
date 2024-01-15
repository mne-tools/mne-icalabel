"""Temporary bug-fixes awaiting an upstream fix."""

from __future__ import annotations  # c.f. PEP 563, PEP 649

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


# https://github.com/sphinx-gallery/sphinx-gallery/issues/1112
class WrapStdOut:
    """Dynamically wrap to sys.stdout.

    This makes packages that monkey-patch sys.stdout (e.g.doctest, sphinx-gallery) work
    properly.
    """

    def __getattr__(self, name: str) -> Any:  # noqa: D105
        # Even more ridiculous than this class, this must be sys.stdout (not
        # just stdout) in order for this to work (tested on OSX and Linux).
        if hasattr(sys.stdout, name):
            return getattr(sys.stdout, name)
        else:
            raise AttributeError(f"'file' object has not attribute '{name}'.")
