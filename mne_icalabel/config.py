from .iclabel import iclabel_label_components

ICALABEL_METHODS = {
    "iclabel": iclabel_label_components,
    "manual": None,
}

# map ICLabel labels to MNE str format
ICLABEL_LABELS_TO_MNE = {
    "Brain": "brain",
    "Eye": "eog",
    "Heart": "ecg",
    "Muscle": "muscle",
    "Channel Noise": "ch_noise",
    "Line Noise": "line_noise",
    "Other": "other",
}
