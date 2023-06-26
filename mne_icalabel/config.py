from .iclabel import iclabel_label_components

ICALABEL_METHODS = {
    "iclabel": iclabel_label_components,
    "manual": None,
}

# map labels to the equivalent str format in MNE
ICA_LABELS_TO_MNE = {
    "Brain": "brain",
    "Muscle": "muscle",
    "Eye": "eog",
    "Heart": "ecg",
    "Line Noise": "line_noise",
    "Channel Noise": "ch_noise",
    "Other": "other",
}
