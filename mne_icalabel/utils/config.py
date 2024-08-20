from __future__ import annotations

import platform
import sys
from functools import lru_cache, partial
from importlib.metadata import metadata, requires, version
from typing import TYPE_CHECKING

import psutil
from mne.utils import _validate_type
from packaging.requirements import Requirement

if TYPE_CHECKING:
    from typing import IO, Callable, Optional


def sys_info(fid: Optional[IO] = None, developer: bool = False):
    """Print the system information for debugging.

    Parameters
    ----------
    fid : file-like | None
        The file to write to, passed to :func:`print`. Can be None to use
        :data:`sys.stdout`.
    developer : bool
        If True, display information about optional dependencies.
    """
    _validate_type(developer, bool, "developer")

    ljust = 26
    out = partial(print, end="", file=fid)
    package = __package__.split(".")[0]
    package = metadata(package).get("Name", package)

    # OS information - requires python 3.8 or above
    out("Platform:".ljust(ljust) + platform.platform() + "\n")
    # python information
    out("Python:".ljust(ljust) + sys.version.replace("\n", " ") + "\n")
    out("Executable:".ljust(ljust) + sys.executable + "\n")
    # CPU information
    out("CPU:".ljust(ljust) + platform.processor() + "\n")
    out("Physical cores:".ljust(ljust) + str(psutil.cpu_count(False)) + "\n")
    out("Logical cores:".ljust(ljust) + str(psutil.cpu_count(True)) + "\n")
    # memory information
    out("RAM:".ljust(ljust))
    out(f"{psutil.virtual_memory().total / float(2 ** 30):0.1f} GB\n")
    out("SWAP:".ljust(ljust))
    out(f"{psutil.swap_memory().total / float(2 ** 30):0.1f} GB\n")
    # package information
    out(f"{package}:".ljust(ljust) + version(package) + "\n")

    # dependencies
    out("\nCore dependencies\n")
    dependencies = [Requirement(elt) for elt in requires(package)]
    core_dependencies = [dep for dep in dependencies if "extra" not in str(dep.marker)]
    _list_dependencies_info(out, ljust, package, core_dependencies)

    # extras
    if developer:
        keys = sorted(
            [
                elt
                for elt in metadata(package).get_all("Provides-Extra")
                if elt not in ("all", "full")
            ]
        )
        for key in keys:
            extra_dependencies = [
                dep
                for dep in dependencies
                if all(elt in str(dep.marker) for elt in ("extra", key))
            ]
            if len(extra_dependencies) == 0:
                continue
            out(f"\nOptional '{key}' dependencies\n")
            _list_dependencies_info(out, ljust, package, extra_dependencies)


def _list_dependencies_info(
    out: Callable, ljust: int, package: str, dependencies: list[Requirement]
) -> None:
    """List dependencies names and versions."""
    unicode = sys.stdout.encoding.lower().startswith("utf")
    if unicode:
        ljust += 1

    not_found: list[Requirement] = list()
    for dep in dependencies:
        if dep.name == package:
            continue
        try:
            version_ = version(dep.name)
        except Exception:
            not_found.append(dep)
            continue

        # build the output string step by step
        output = f"✔︎ {dep.name}" if unicode else dep.name
        # handle version specifiers
        if len(dep.specifier) != 0:
            output += f" ({str(dep.specifier)})"
        output += ":"
        output = output.ljust(ljust) + version_

        # handle special dependencies with backends, C dep, ..
        if dep.name in ("matplotlib", "seaborn") and version_ != "Not found.":
            try:
                from matplotlib import pyplot as plt

                backend = plt.get_backend()
            except Exception:
                backend = "Not found"

            output += f" (backend: {backend})"
        if dep.name == "pyvista":
            version_, renderer = _get_gpu_info()
            if version_ is None:
                output += " (OpenGL unavailable)"
            else:
                output += f" (OpenGL {version_} via {renderer})"
        out(output + "\n")

    if len(not_found) != 0:
        not_found = [
            f"{dep.name} ({str(dep.specifier)})"
            if len(dep.specifier) != 0
            else dep.name
            for dep in not_found
        ]
        if unicode:
            out(f"✘ Not installed: {', '.join(not_found)}\n")
        else:
            out(f"Not installed: {', '.join(not_found)}\n")


@lru_cache(maxsize=1)
def _get_gpu_info() -> tuple[Optional[str], Optional[str]]:
    """Get the GPU information."""
    try:
        from pyvista import GPUInfo

        gi = GPUInfo()
        return gi.version, gi.renderer
    except Exception:
        return None, None
