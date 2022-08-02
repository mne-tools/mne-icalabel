from mne.utils import check_version


# TODO: Not needed after MNE version req. is bumped to 1.1
# mne.utils._testing.requires_version can be used instead.
# c.f. mne-tools/mne-python #10958
def requires_version(library, min_version="0.0"):
    """Check for a library version."""
    import pytest

    reason = f"Requires {library}"
    if min_version != "0.0":
        reason += f" version >= {min_version}"
    return pytest.mark.skipif(not check_version(library, min_version), reason=reason)
