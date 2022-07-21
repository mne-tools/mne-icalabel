"""
Fill docstrings to avoid redundant docstrings in multiple files.

Inspired from mne: https://mne.tools/stable/index.html
Inspired from mne.utils.docs.py by Eric Larson <larson.eric.d@gmail.com>
"""

from mne.utils.docs import docdict as docdict_mne

# ------------------------- Documentation dictionary -------------------------
docdict = {}

# ---- Documentation to inc. from MNE ----
keys = (
    "border_topomap",
    "extrapolate_topomap",
    "outlines_topomap",
    "picks_ica",
)

for key in keys:
    docdict[key] = docdict_mne[key]
