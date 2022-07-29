import argparse

from qtpy.QtWidgets import QApplication

from mne_icalabel.gui._label_components import ICAComponentLabeler


def main():
    """Entry point for mne_gui_ic_annotation."""
    parser = argparse.ArgumentParser(prog="mne-icalabel", description="IC annotation GUI")
    parser.add_argument("--dev", help="loads a sample dataset.", action="store_true")
    args = parser.parse_args()

    if not args.dev:
        raise NotImplementedError
    else:
        from mne.datasets import sample
        from mne.io import read_raw
        from mne.preprocessing import ICA

        directory = sample.data_path() / "MEG" / "sample"
        raw = read_raw(directory / "sample_audvis_raw.fif", preload=False)
        raw.crop(0, 10).pick_types(eeg=True, exclude="bads")
        raw.load_data()
        # preprocess
        raw.filter(l_freq=1.0, h_freq=100.0)
        raw.set_eeg_reference("average")

        n_components = 15
        ica = ICA(n_components=n_components, method="picard")
        ica.fit(raw)

    app = QApplication([])
    window = ICAComponentLabeler(inst=raw, ica=ica)
    window.show()
    app.exec()
