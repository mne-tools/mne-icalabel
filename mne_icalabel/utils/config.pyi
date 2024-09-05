from typing import IO, Callable

from packaging.requirements import Requirement

def sys_info(fid: IO | None = None, developer: bool = False):
    """Print the system information for debugging.

    Parameters
    ----------
    fid : file-like | None
        The file to write to, passed to :func:`print`. Can be None to use
        :data:`sys.stdout`.
    developer : bool
        If True, display information about optional dependencies.
    """

def _list_dependencies_info(
    out: Callable, ljust: int, package: str, dependencies: list[Requirement]
) -> None:
    """List dependencies names and versions."""

def _get_gpu_info() -> tuple[str | None, str | None]:
    """Get the GPU information."""
