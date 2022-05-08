"""
Optional dependency import.
Inspired from pandas: https://pandas.pydata.org/
"""
import importlib

# A mapping from import name to package name (on PyPI) when the package name
# is different, e.g. {"serial": "pyserial"}.
INSTALL_MAPPING = {
    "picard": "python-picard",
}


def import_optional_dependency(
    name: str, extra: str = "", raise_error: bool = True
):
    """
    Import an optional dependency.
    By default, if a dependency is missing an ImportError with a nice message
    will be raised.

    Parameters
    ----------
    name : str
        The module name.
    extra : str
        Additional text to include in the ImportError message.
    raise_error : bool
        What to do when a dependency is not found.
        * True : Raise an ImportError.
        * False: If the module is not installed, return None, otherwise, return
        the module.

    Returns
    -------
    maybe_module : Optional[ModuleType]
        The imported module when found.
        None is returned when the package is not found and raise_error is
        False.
    """
    package_name = INSTALL_MAPPING.get(name)
    install_name = package_name if package_name is not None else name

    try:
        module = importlib.import_module(name)
    except ImportError:
        if raise_error:
            raise ImportError(
                f"Missing optional dependency '{install_name}'. {extra} "
                f"Use pip or conda to install {install_name}."
            )
        else:
            return None

    return module
