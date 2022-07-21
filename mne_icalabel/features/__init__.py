"""Features for the ICLabel"""

from mne.utils import check_version, warn

if check_version("mne", "1.1"):
    # Interpolation has been fixed in mne-tools/mne-python #10579 and #10617
    from .topomap import get_topomap, get_topomaps  # noqa: F401
else:

    def get_topomap(*args, **kwargs):
        warn("Topographic feature is only supported with MNE >= 1.1.0")
        pass

    def get_topomaps(*args, **kwargs):
        warn("Topographic feature is only supported with MNE >= 1.1.0")
        pass
