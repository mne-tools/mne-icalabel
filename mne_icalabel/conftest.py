import numpy as np
import pytest
from mne import set_log_level


def pytest_configure(config):
    """Configure pytest options."""
    warnings_lines = r"""
    error::
    # Until MNE 1.6 is the minimum supported version
    # c.f. https://github.com/mne-tools/mne-python/pull/12143
    ignore:Setting non-standard config type.*:RuntimeWarning
    # pandas 3.0 will require pyarrow
    ignore:\n*Pyarrow will become a required dependency of pandas.*:DeprecationWarning
    # Python 3.12+ gives a deprecation warning if TarFile.extraction_filter is None
    ignore:Python 3\.14 will, by default, filter extracted tar.*:DeprecationWarning
    # onnxruntime on windows runners
    ignore:Unsupported Windows version.*:UserWarning
    # Matplotlib deprecation issued in VSCode test debugger
    ignore:.*interactive_bk.*:matplotlib._api.deprecation.MatplotlibDeprecationWarning
    """
    for warning_line in warnings_lines.split("\n"):
        warning_line = warning_line.strip()
        if warning_line and not warning_line.startswith("#"):
            config.addinivalue_line("filterwarnings", warning_line)
    set_log_level("WARNING")


@pytest.fixture(scope="session")
def rng():
    """Return a numpy random generator."""
    return np.random.default_rng()
