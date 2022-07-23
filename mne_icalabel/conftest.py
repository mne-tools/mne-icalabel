# -*- coding: utf-8 -*-
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause
import warnings
from contextlib import contextmanager

import pytest
from mne.utils import _check_qt_version

# most of this adapted from MNE-Python


def pytest_configure(config):
    """Configure pytest options."""
    # Markers
    for marker in ("examples",):
        config.addinivalue_line("markers", marker)
    for fixture in ("matplotlib_config", "close_all"):
        config.addinivalue_line("usefixtures", fixture)

    warning_lines = r"""
    error::
    ignore:.*Setting non-standard config type.*:
    always::ResourceWarning
    """  # noqa: E501
    for warning_line in warning_lines.split("\n"):
        warning_line = warning_line.strip()
        if warning_line and not warning_line.startswith("#"):
            config.addinivalue_line("filterwarnings", warning_line)


def _check_skip_backend(name):
    from mne.viz.backends.tests._utils import (
        has_imageio_ffmpeg,
        has_pyvista,
        has_pyvistaqt,
    )

    if name in ("pyvistaqt", "notebook"):
        if not has_pyvista():
            pytest.skip("Test skipped, requires pyvista.")
        if not has_imageio_ffmpeg():
            pytest.skip("Test skipped, requires imageio-ffmpeg")
    if name == "pyvistaqt" and not _check_qt_version():
        pytest.skip("Test skipped, requires Qt.")
    if name == "pyvistaqt" and not has_pyvistaqt():
        pytest.skip("Test skipped, requires pyvistaqt")


@contextmanager
def _use_backend(backend_name, interactive):
    from mne.viz.backends.renderer import _use_test_3d_backend

    _check_skip_backend(backend_name)
    with _use_test_3d_backend(backend_name, interactive=interactive):
        from mne.viz.backends import renderer

        try:
            yield renderer
        finally:
            renderer.backend._close_all()


@pytest.fixture(scope="module", params=["pyvistaqt"])
def renderer_interactive_pyvistaqt(request, options_3d):
    """Yield the interactive PyVista backend."""
    with _use_backend(request.param, interactive=True) as renderer:
        yield renderer


@pytest.fixture(scope="session")
def matplotlib_config():
    """Configure matplotlib for viz tests."""
    from matplotlib import cbook, use

    want = "agg"  # don't pop up windows
    with warnings.catch_warnings(record=True):  # ignore warning
        warnings.filterwarnings("ignore")
        use(want, force=True)
    import matplotlib.pyplot as plt

    assert plt.get_backend() == want
    # overwrite some params that can horribly slow down tests that
    # users might have changed locally (but should not otherwise affect
    # functionality)
    plt.ioff()
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["figure.max_open_warning"] = 100

    # Make sure that we always reraise exceptions in handlers
    orig = cbook.CallbackRegistry

    class CallbackRegistryReraise(orig):
        def __init__(self, exception_handler=None, signals=None):
            super(CallbackRegistryReraise, self).__init__(exception_handler)

    cbook.CallbackRegistry = CallbackRegistryReraise


@pytest.fixture(autouse=True)
def close_all():
    """Close all matplotlib plots, regardless of test status."""
    # This adds < 1 µS in local testing, and we have ~2500 tests, so ~2 ms max
    import matplotlib.pyplot as plt

    yield
    plt.close("all")
