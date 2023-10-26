from mne import set_log_level


def pytest_configure(config):
    """Configure pytest options."""
    warnings_lines = r"""
    error
    """
    for warning_line in warnings_lines.split("\n"):
        warning_line = warning_line.strip()
        if warning_line and not warning_line.startswith("#"):
            config.addinivalue_line("filterwarnings", warning_line)
    set_log_level("WARNING")
