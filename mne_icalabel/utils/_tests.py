from __future__ import annotations  # c.f. PEP 563, PEP 649

from importlib import import_module
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from typing import Callable


def requires_module(name: str):  # pragma: no cover
    """Skip a test if package is not available (decorator)."""
    try:
        import_module(name)
        skip = False
    except ImportError:
        skip = True

    def decorator(function: Callable):
        return pytest.mark.skipif(
            skip, reason=f"Test {function.__name__} skipped, requires {name}."
        )(function)

    return decorator
