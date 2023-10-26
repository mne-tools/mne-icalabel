from mne import set_log_level


def pytest_configure(config):
    """Configure pytest options."""
    warnings_lines = r"""
    error::
    # Until MNE 1.6 is the minimum supported version
    # c.f. https://github.com/mne-tools/mne-python/pull/12143
    ignore:Setting non-standard config type.*:RuntimeWarning
    # mne_bids <-> numpy 2.0
    # c.f. https://github.com/mne-tools/mne-bids/pull/1184
    ignore:`in1d` is deprecated. Use `np.isin` instead.:DeprecationWarning
    """
    for warning_line in warnings_lines.split("\n"):
        warning_line = warning_line.strip()
        if warning_line and not warning_line.startswith("#"):
            config.addinivalue_line("filterwarnings", warning_line)
    set_log_level("WARNING")
