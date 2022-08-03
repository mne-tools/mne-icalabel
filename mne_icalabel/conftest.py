# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause


def pytest_configure(config):
    """Configure pytest options."""
    warning_lines = r"""
    error::
    ignore:.*Setting non-standard config type.*:
    always::ResourceWarning
    """  # noqa: E501
    for warning_line in warning_lines.split("\n"):
        warning_line = warning_line.strip()
        if warning_line and not warning_line.startswith("#"):
            config.addinivalue_line("filterwarnings", warning_line)
