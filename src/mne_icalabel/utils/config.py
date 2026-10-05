from __future__ import annotations

import platform
import sys
from functools import lru_cache, partial
from importlib.metadata import metadata, requires, version
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING

import psutil
from mne.utils import _validate_type
from packaging.requirements import Requirement

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import IO


def sys_info(fid: IO | None = None, *, extra: bool = False, developer: bool = False):
    """Print the system information for debugging.

    Parameters
    ----------
    fid : file-like | None
        The file to write to, passed to :func:`print`. Can be None to use
        :data:`sys.stdout`.
    extra : bool
        If True, display information about optional dependencies.
    developer : bool
        If True, display information about developer dependencies. Only available for
        the package installed in editable mode.
    """
    _validate_type(extra, bool, "extra")
    _validate_type(developer, bool, "developer")

    ljust = 26
    out = partial(print, end="", file=fid)
    module = __package__.split(".")[0]
    package = metadata(module).get("Name", module)

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
    out(f"{psutil.virtual_memory().total / float(2**30):0.1f} GB\n")
    out("SWAP:".ljust(ljust))
    out(f"{psutil.swap_memory().total / float(2**30):0.1f} GB\n")
    # package information
    out(f"{package}:".ljust(ljust) + version(package) + "\n")

    # dependencies
    out("\nCore dependencies\n")
    dependencies = [Requirement(elt) for elt in requires(package)]
    core_dependencies = [dep for dep in dependencies if "extra" not in str(dep.marker)]
    _list_dependencies_info(out, ljust, package, core_dependencies)

    if extra:
        for key in sorted(metadata(package).get_all("Provides-Extra") or []):
            extra_dependencies = [
                dep
                for dep in dependencies
                if all(elt in str(dep.marker) for elt in ("extra", key))
            ]
            if len(extra_dependencies) == 0:
                continue
            out(f"\nOptional '{key}' dependencies\n")
            _list_dependencies_info(out, ljust, package, extra_dependencies)

    if developer:
        import tomllib  # requires Python 3.11+

        # following PEP 735, dependency-groups are intentionally omitted from metadata,
        # thus we need to parse the pyproject.toml file directly.
        origin = Path(find_spec(module).origin)
        pyproject = origin.parents[2] / "pyproject.toml"
        if not pyproject.exists():
            raise RuntimeError(
                f"The pyproject.toml file for the package {package} could not be "
                "found. To retrieve developer dependencies, please install the package "
                "from source in an editable install, e.g. using 'uv sync'."
            )
        pyproject_data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        dependency_groups = pyproject_data.get("dependency-groups", {})
        for key in sorted(dependency_groups):
            dependencies = [Requirement(dep) for dep in dependency_groups[key]]
            out(f"\nDeveloper '{key}' dependencies\n")
            _list_dependencies_info(out, ljust, package, dependencies)


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
def _get_gpu_info() -> tuple[str | None, str | None]:
    """Get the GPU information."""
    try:
        from pyvista import GPUInfo

        gi = GPUInfo()
        return gi.version, gi.renderer
    except Exception:
        return None, None
