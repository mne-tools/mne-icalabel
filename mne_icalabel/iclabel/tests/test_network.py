try:
    from importlib.resources import files  # type: ignore
except ImportError:
    from importlib_resources import files  # type: ignore

from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import loadmat

from mne_icalabel.datasets import icalabel
from mne_icalabel.iclabel.network import ICLabelNet, _format_input, run_iclabel

dataset_path = Path(icalabel.data_path()) / "iclabel"


# Network weights
torch_iclabel_path = str(files("mne_icalabel.iclabel").joinpath("assets/iclabelNet.pt"))
matconvnet_iclabel_path = dataset_path / "network/netICL.mat"

# Network forward pass input/output
matconvnet_fw_input_path = dataset_path / "network/network_input.mat"
matconvnet_fw_output_path = dataset_path / "network/network_output.mat"

# Features (similar to network_input)
features_raw_path = dataset_path / "features/features-raw.mat"
features_epo_path = dataset_path / "features/features-epo.mat"

# Features formatted
features_formatted_raw_path = dataset_path / "features/features-formatted-raw.mat"
features_formatted_epo_path = dataset_path / "features/features-formatted-epo.mat"

# ICLabel output
iclabel_output_raw_path = dataset_path / "iclabel-output-raw.mat"
iclabel_output_epo_path = dataset_path / "iclabel-output-epo.mat"


def test_weights():
    """Compare the weights of pytorch model and matconvnet model."""
    network_python = torch.load(torch_iclabel_path)
    network_matlab = loadmat(matconvnet_iclabel_path)

    # load weights from matlab network
    weights_matlab = network_matlab["params"]["value"][0, :]
    # format weights from matlab network to torch convention
    for k, weight in enumerate(weights_matlab):
        if weight.ndim == 4:
            weights_matlab[k] = weight.transpose((3, 2, 0, 1))
        elif weight.ndim == 3:
            weights_matlab[k] = weight.transpose((2, 0, 1))

    network_python_layers = [layer for layer in network_python.keys() if "seq" not in layer]
    network_matlab_layers = [elt[0] for elt in network_matlab["params"]["name"][0, :]]

    # match layer names torch -> matconvnet
    keyword_mapping = {
        "img": "image",
        "psds": "psdmed",
        "autocorr": "autocorr",
        "weight": "kernel",
        "bias": "bias",
    }

    for python_layer in network_python_layers:
        split = python_layer.split(".")
        if len(split) == 2:
            _, param = split
            matlab_layer = f"discriminator_conv_{keyword_mapping[param]}"
        elif len(split) == 3:
            feature, idx, param = split
            feature = keyword_mapping[feature.split("_")[0]]
            idx = int(idx[-1])
            param = keyword_mapping[param]
            matlab_layer = f"discriminator_{feature}_layer{idx}_conv_{param}"
        else:
            raise ValueError("Unexpected layer name.")

        # find matlab_layer idx
        idx = network_matlab_layers.index(matlab_layer)
        # compare layers weights
        assert np.allclose(network_python[python_layer], weights_matlab[idx])


def test_network_outputs():
    """
    Compare that the ICLabel network in python and matlab outputs the same
    values for a common set of features (input to the forward pass).

    Notes
    -----
    The forward pass has been run in matconvnet with the same input features.
    The corresponding MATLAB code can be found in 'data/network_output.txt'.
    """
    # load features to use for the forward pass
    features = loadmat(matconvnet_fw_input_path)["input"][0, :]
    # features is a (6, ) array with:
    # - elements in position [0, 2, 4] -> 1-element array with the var name
    # - elements in position [1, 3, 5] -> data arrays
    assert "in_image" == features[0][0]
    images = features[1]
    assert "in_psdmed" == features[2][0]
    psd = features[3]
    assert "in_autocorr" == features[4][0]
    autocorr = features[5]

    # reshape the features to fit torch format
    images = np.transpose(images, (3, 2, 0, 1))
    psd = np.transpose(psd, (3, 2, 0, 1))
    autocorr = np.transpose(autocorr, (3, 2, 0, 1))

    # convert to tensors
    images = torch.from_numpy(images).float()
    psd = torch.from_numpy(psd).float()
    autocorr = torch.from_numpy(autocorr).float()

    # run the forward pass on pytorch
    iclabel_net = ICLabelNet()
    iclabel_net.load_state_dict(torch.load(torch_iclabel_path))
    torch_labels = iclabel_net(images, psd, autocorr)
    torch_labels = torch_labels.detach().numpy()  # (30, 7)

    # load the matconvnet output of the forward pass on those 3 feature arrays
    matlab_labels = loadmat(matconvnet_fw_output_path)["labels"]  # (30, 7)

    # Compare both outputs
    assert np.allclose(matlab_labels, torch_labels, atol=1e-7)


@pytest.mark.parametrize(
    "eeglab_feature_file, eeglab_feature_formatted_file",
    [
        (features_raw_path, features_formatted_raw_path),
        (features_epo_path, features_formatted_epo_path),
    ],
)
def test_format_input(eeglab_feature_file, eeglab_feature_formatted_file):
    """Test formatting of input feature before feeding them to the network."""
    features_eeglab = loadmat(eeglab_feature_file)["features"]
    topo, psd, autocorr = _format_input(
        features_eeglab[0, 0], features_eeglab[0, 1], features_eeglab[0, 2]
    )

    features_formatted_eeglab = loadmat(eeglab_feature_formatted_file)["features"]
    topo_eeglab = features_formatted_eeglab[0, 0]
    psd_eeglab = features_formatted_eeglab[0, 1]
    autocorr_eeglab = features_formatted_eeglab[0, 2]

    assert np.allclose(topo, topo_eeglab)
    assert np.allclose(psd, psd_eeglab)
    assert np.allclose(autocorr, autocorr_eeglab)


@pytest.mark.parametrize(
    "eeglab_feature_file, eeglab_output_file",
    [
        (features_raw_path, iclabel_output_raw_path),
        (features_epo_path, iclabel_output_epo_path),
    ],
)
def test_run_iclabel(eeglab_feature_file, eeglab_output_file):
    """Test that the network outputs the same values for the features in
    'features_raw_path' and 'features_epo_path' that contains the features
    extracted in EEGLAB. This set of feature is compared with the set of
    features retrieved in python in 'test_features.py:test_get_features'."""
    features_eeglab = loadmat(eeglab_feature_file)["features"]
    # run the forward pass on pytorch
    labels = run_iclabel(features_eeglab[0, 0], features_eeglab[0, 1], features_eeglab[0, 2])

    # load the labels from EEGLAB
    matlab_labels = loadmat(eeglab_output_file)["labels"]  # (30, 7)

    # Compare both outputs
    assert np.allclose(matlab_labels, labels, atol=1e-7)
