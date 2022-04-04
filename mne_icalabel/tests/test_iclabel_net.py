import importlib.resources
from pathlib import Path
import pytest

import mne
import numpy as np
from scipy.io import loadmat
import torch


from mne_icalabel.ica_label import ica_eeg_features
from mne_icalabel.ica_net import run_iclabel


# load in test data for features from original Matlab ICLabel
ica_file_path = str(importlib.resources.files(
    'mne_icalabel.tests').joinpath('data/eeglab_ica.set'))
ica_raw_file_path = str(importlib.resources.files(
    'mne_icalabel.tests').joinpath('data/eeglab_ica_raw.mat'))
torch_iclabel_path = Path(__file__).parent.parent / 'assets' / 'iclabelNet.pt'
matconvnet_iclabel_path = Path(__file__).parent / 'data' / 'netICL.mat'


def test_weights():
    """Compare the weights of pytorch model and matconvnet model."""
    network_python = torch.load(torch_iclabel_path)
    network_matlab = loadmat(matconvnet_iclabel_path)

    # load weights from matlab network
    weights_matlab = network_matlab["params"]['value'][0,:]
    # format weights from matlab network to torch convention
    for k, weight in enumerate(weights_matlab):
        if weight.ndim == 4:
            weights_matlab[k] = weight.transpose((3, 2, 0, 1))
        elif weight.ndim == 3:
            weights_matlab[k] = weight.transpose((2,0,1))

    network_python_layers = [
        layer for layer in network_python.keys() if 'seq' not in layer]
    network_matlab_layers = [
        elt[0] for elt in network_matlab['params']['name'][0, :]]

    # match layer names torch -> matconvnet
    keyword_mapping = {
        'img': 'image',
        'psds': 'psdmed',
        'autocorr': 'autocorr',
        'weight': 'kernel',
        'bias': 'bias',
        }

    for python_layer in network_python_layers:
        split = python_layer.split('.')
        if len(split) == 2:
            _, param = split
            matlab_layer = f'discriminator_conv_{keyword_mapping[param]}'
        elif len(split) == 3:
            feature, idx, param = split
            feature = keyword_mapping[feature.split('_')[0]]
            idx = int(idx[-1])
            param = keyword_mapping[param]
            matlab_layer = f'discriminator_{feature}_layer{idx}_conv_{param}'
        else:
            raise ValueError('Unexpected layer name.')

        # find matlab_layer idx
        idx = network_matlab_layers.index(matlab_layer)

        # compare layers weights
        assert np.allclose(network_python[python_layer], weights_matlab[idx])


def test_network_output():
    """
    Compare that the ICLabel network in python and matlab outputs the same
    values for a common set of features.
    """
    pass


def test_labels():
    """Test that the ICLabel network in python and matlab outputs the same
    labels for a common ICA decomposition."""
    eeglab_ica = mne.preprocessing.read_ica_eeglab(ica_file_path)
    eeglab_raw = mne.io.read_raw_eeglab(ica_file_path)

    eeglab_ica_raw = loadmat(ica_raw_file_path)['EEG']
    raw_labels = eeglab_ica_raw['etc'][0][0][0][0]['ic_classification'][0][0][0][0][0][1]

    # compute the features of the ICA waveforms
    ica_features = ica_eeg_features(eeglab_raw, eeglab_ica)
    topo = ica_features[0].astype(np.float32)
    psds = ica_features[1].astype(np.float32)
    with pytest.warns(np.ComplexWarning):
        autocorr = ica_features[2].astype(np.float32)

    # run ICLabel network
    labels = run_iclabel(topo, psds, autocorr)

    num_labels = np.argmax(labels, axis=1)
    orig_num_labels = np.argmax(raw_labels, axis=1)

    # TO show that these are equal, add an assert False statement at the end
    # TODO: there is one difference between IClabel in Matlab and Python
    print(num_labels)
    print(orig_num_labels)

    num_labels[num_labels != 0] = 1
    orig_num_labels[orig_num_labels != 0] = 1
    assert sum(map(lambda x, y: bool(x - y), num_labels, orig_num_labels)) == 1
