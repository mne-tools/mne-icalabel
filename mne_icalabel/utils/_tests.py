from functools import partial
from importlib import import_module
from typing import Callable

import pytest


def _requires_module(function: Callable, name: str):
    """Skip a test if package is not available (decorator)."""
    try:
        import_module(name)
        skip = False
    except ImportError:
        skip = True
    reason = f"Test {function.__name__} skipped, requires {name}."
    return pytest.mark.skipif(skip, reason=reason)(function)


requires_torch = partial(_requires_module, name="torch")
requires_onnx = partial(_requires_module, name="onnxruntime")
