from .iclabel import iclabel_label_components

ICALABEL_METHODS = {
    "iclabel": iclabel_label_components,
    "manual": None,
}

# map ICLabel labels to MNE str format
ICLABEL_LABELS_TO_MNE = {
    "Brain": "brain",
    "Muscle": "muscle",
    "Eye": "eog",
    "Heart": "ecg",
    "Line Noise": "line_noise",
    "Channel Noise": "ch_noise",
    "Other": "other",
}
