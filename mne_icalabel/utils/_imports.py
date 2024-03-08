"""Handle optional dependency imports.

Inspired from pandas: https://pandas.pydata.org/
"""

from __future__ import annotations  # c.f. PEP 563, PEP 649

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType
    from typing import Optional

# A mapping from import name to package name (on PyPI) when the package name
# is different.
_INSTALL_MAPPING: dict[str, str] = {
    "codespell_lib": "codespell",
    "cv2": "opencv-python",
    "parallel": "pyparallel",
    "pytest_cov": "pytest-cov",
    "serial": "pyserial",
    "sklearn": "scikit-learn",
    "sksparse": "scikit-sparse",
}


def import_optional_dependency(
    name: str,
    extra: str = "",
    raise_error: bool = True,
) -> Optional[ModuleType]:
    """Import an optional dependency.

    By default, if a dependency is missing an ImportError with a nice message will be
    raised.

    Parameters
    ----------
    name : str
        The module name.
    extra : str
        Additional text to include in the ImportError message.
    raise_error : bool
        What to do when a dependency is not found.
        * True : Raise an ImportError.
        * False: Return None.

    Returns
    -------
    module : Module | None
        The imported module when found.
        None is returned when the package is not found and raise_error is False.
    """
    package_name = _INSTALL_MAPPING.get(name)
    install_name = package_name if package_name is not None else name

    try:
        module = importlib.import_module(name)
    except ImportError:
        if raise_error:
            raise ImportError(
                f"Missing optional dependency '{install_name}'. {extra} Use pip or "
                f"conda to install {install_name}."
            )
        else:
            return None

    return module
