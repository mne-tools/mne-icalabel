import argparse

from mne import read_epochs
from mne.io import read_raw
from mne.preprocessing import read_ica
from qtpy.QtWidgets import QApplication

from mne_icalabel.gui._label_components import ICAComponentLabeler


def main():
    """Entry point for mne_gui_ic_annotation."""
    parser = argparse.ArgumentParser(
        prog="mne-icalabel", description="IC annotation GUI"
    )
    parser.add_argument(
        "fname_raw_epo",
        type=str,
        help="path to the file with raw or epochs.",
    )
    parser.add_argument(
        "fname_ica",
        type=str,
        help="path to the file with the ICA decomposition.",
    )
    parser.add_argument("--dev", help="loads a sample dataset.", action="store_true")
    args = parser.parse_args()

    if not args.dev:
        for func in (read_raw, read_epochs):
            try:
                inst = func(args.fname_raw_epo, preload=True)
                break
            except Exception:
                pass
        else:
            raise RuntimeError(f"Could not load the file {args.fname_raw_epo}.")
        ica = read_ica(args.fname_ica)
    else:
        from mne.datasets import sample
        from mne.preprocessing import ICA

        directory = sample.data_path() / "MEG" / "sample"
        inst = read_raw(directory / "sample_audvis_raw.fif", preload=False)
        inst.crop(0, 10).pick_types(eeg=True, exclude="bads")
        inst.load_data()
        # preprocess
        inst.filter(l_freq=1.0, h_freq=100.0)
        inst.set_eeg_reference("average")

        n_components = 15
        ica = ICA(n_components=n_components, method="picard")
        ica.fit(inst)

    app = QApplication([])
    window = ICAComponentLabeler(inst=inst, ica=ica)
    window.show()
    app.exec()
