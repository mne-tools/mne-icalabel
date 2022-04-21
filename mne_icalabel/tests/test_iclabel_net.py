try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

from mne.io import read_raw
from mne.preprocessing import read_ica_eeglab
import numpy as np
from scipy.io import loadmat
import torch

from mne_icalabel.ica_label import ica_eeg_features
from mne_icalabel.ica_net import ICLabelNet, run_iclabel


# Network weights
torch_iclabel_path = str(files("mne_icalabel").joinpath("assets/iclabelNet.pt"))
matconvnet_iclabel_path = str(files("mne_icalabel.tests").joinpath("data/netICL.mat"))

# Network forward pass input/output
matconvnet_fw_input_path = str(files("mne_icalabel.tests").joinpath("data/network_input.mat"))
matconvnet_fw_output_path = str(files("mne_icalabel.tests").joinpath("data/network_output.mat"))

# Raw files with ICA decomposition
raw_eeglab_path = str(files("mne_icalabel.tests").joinpath("data/sample.set"))


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

    network_python_layers = [
        layer for layer in network_python.keys() if "seq" not in layer
    ]
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
    features = loadmat(matconvnet_fw_input_path)['input'][0, :]
    # features is a (6, ) array with:
    # - elements in position [0, 2, 4] -> 1-element array with the var name
    # - elements in position [1, 3, 5] -> data arrays
    assert 'in_image' == features[0][0]
    images = features[1]
    assert 'in_psdmed' == features[2][0]
    psd = features[3]
    assert 'in_autocorr' == features[4][0]
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
    assert np.allclose(matlab_labels, torch_labels, rtol=1e-5, atol=1e-5)


def test_labels():
    """Test that the ICLabel network in python and matlab outputs the same
    labels for a common ICA decomposition.

    Notes
    -----
    The raw and its ICA decomposition have been obtained in EEGLAB from one of
    its sample dataset.
    The corresponding MATLAB code can be found in 'data/sample.txt'.
    """
    raw = read_raw(raw_eeglab_path, preload=True)
    ica = read_ica_eeglab(raw_eeglab_path)

    features = ica_eeg_features(raw, ica)
    # TODO: Feature extraction is failing for now. To be completed when feature
    # extraction is fully tested.
    topo = features[0].astype(np.float32)
    psds = features[1].astype(np.float32)
    autocorr = features[2].astype(np.float32)
    labels = run_iclabel(topo, psds, autocorr)
