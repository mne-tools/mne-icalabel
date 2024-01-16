from types import ModuleType as ModuleType
from typing import Optional

_INSTALL_MAPPING: dict[str, str]

def import_optional_dependency(name: str, extra: str='', raise_error: bool=True) -> Optional[ModuleType]:
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