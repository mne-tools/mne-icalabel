try:
    from importlib.resources import files  # type: ignore
except ImportError:
    from importlib_resources import files  # type: ignore

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import ArrayLike


class _ICLabelNetImg(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=128,
            kernel_size=(4, 4),
            padding=1,
            stride=(2, 2),
        )
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=(4, 4),
            padding=1,
            stride=(2, 2),
        )
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.conv3 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=(4, 4),
            padding=1,
            stride=(2, 2),
        )
        self.relu3 = nn.LeakyReLU(negative_slope=0.2)
        self.sequential = nn.Sequential(
            self.conv1, self.relu1, self.conv2, self.relu2, self.conv3, self.relu3
        )

    def forward(self, x):  # noqa: D102
        return self.sequential(x)


class _ICLabelNetPSDS(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=128,
            kernel_size=(1, 3),
            padding=(0, 1),
            stride=(1, 1),
        )
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=(1, 3),
            padding=(0, 1),
            stride=(1, 1),
        )
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.conv3 = nn.Conv2d(
            in_channels=256,
            out_channels=1,
            kernel_size=(1, 3),
            padding=(0, 1),
            stride=(1, 1),
        )
        self.relu3 = nn.LeakyReLU(negative_slope=0.2)
        self.sequential = nn.Sequential(
            self.conv1, self.relu1, self.conv2, self.relu2, self.conv3, self.relu3
        )

    def forward(self, x):  # noqa: D102
        return self.sequential(x)


class _ICLabelNetAutocorr(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=128,
            kernel_size=(1, 3),
            padding=(0, 1),
            stride=(1, 1),
        )
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=(1, 3),
            padding=(0, 1),
            stride=(1, 1),
        )
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.conv3 = nn.Conv2d(
            in_channels=256,
            out_channels=1,
            kernel_size=(1, 3),
            padding=(0, 1),
            stride=(1, 1),
        )
        self.relu3 = nn.LeakyReLU(negative_slope=0.2)
        self.sequential = nn.Sequential(
            self.conv1, self.relu1, self.conv2, self.relu2, self.conv3, self.relu3
        )

    def forward(self, x):  # noqa: D102
        return self.sequential(x)


class ICLabelNet(nn.Module):
    """The ICLabel neural network."""

    def __init__(self):
        super().__init__()

        self.img_conv = _ICLabelNetImg()
        self.psds_conv = _ICLabelNetPSDS()
        self.autocorr_conv = _ICLabelNetAutocorr()

        self.conv = nn.Conv2d(
            in_channels=712,
            out_channels=7,
            kernel_size=(4, 4),
            padding=0,
            stride=(1, 1),
        )
        self.softmax = nn.Softmax(dim=1)

        self.seq = nn.Sequential(self.conv, self.softmax)

    @staticmethod
    def reshape_fortran(x: torch.Tensor, shape) -> torch.Tensor:  # noqa: D102
        x = x.permute(*reversed(range(len(x.shape))))
        return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))

    def reshape_concat(self, tensor: torch.Tensor) -> torch.Tensor:  # noqa: D102
        tensor = self.reshape_fortran(tensor, [-1, 1, 1, 100])
        tensor = torch.cat([tensor, tensor, tensor, tensor], 1)
        tensor = torch.cat([tensor, tensor, tensor, tensor], 2)
        tensor = tensor.permute((0, 3, 1, 2))
        return tensor

    def forward(
        self, images: torch.Tensor, psds: torch.Tensor, autocorr: torch.Tensor
    ) -> torch.Tensor:  # noqa: D102
        out_img = self.img_conv(images)
        out_psds = self.psds_conv(psds)
        out_autocorr = self.autocorr_conv(autocorr)

        # PSDS reshape, concat, permute
        psds_perm = self.reshape_concat(out_psds)

        # Autocorr reshape, concat, permute
        autocorr_perm = self.reshape_concat(out_autocorr)

        concat = torch.cat([out_img, psds_perm, autocorr_perm], 1)

        labels = self.seq(concat)

        labels = labels.squeeze()
        labels = self.reshape_fortran(labels.permute(1, 0), [-1, 4])
        labels = torch.mean(labels, 1)
        labels = self.reshape_fortran(labels, [7, -1])
        labels = labels.permute(1, 0)

        return labels


def _format_input(topo: ArrayLike, psd: ArrayLike, autocorr: ArrayLike):
    """Replicate the input formatting in EEGLAB -ICLabel.

    .. code-block:: matlab

       images = cat(4, images, -images, images(:, end:-1:1, :, :), ...
                    -images(:, end:-1:1, :, :));
       psds = repmat(psds, [1 1 1 4]);
       autocorrs = repmat(autocorrs, [1 1 1 4]);
    """
    formatted_topo = np.concatenate(
        (topo, -1 * topo, np.flip(topo, axis=1), np.flip(-1 * topo, axis=1)),
        axis=3,
    )
    formatted_psd = np.tile(psd, (1, 1, 1, 4))
    formatted_autocorr = np.tile(autocorr, (1, 1, 1, 4))
    return formatted_topo, formatted_psd, formatted_autocorr


def _format_input_for_torch(topo: ArrayLike, psd: ArrayLike, autocorr: ArrayLike):
    """Format the features to the correct shape and type for pytorch."""
    topo = np.transpose(topo, (3, 2, 0, 1))
    psd = np.transpose(psd, (3, 2, 0, 1))
    autocorr = np.transpose(autocorr, (3, 2, 0, 1))

    topo = torch.from_numpy(topo).float()
    psd = torch.from_numpy(psd).float()
    autocorr = torch.from_numpy(autocorr).float()

    return topo, psd, autocorr


def run_iclabel(images: ArrayLike, psds: ArrayLike, autocorr: ArrayLike):
    """Run the ICLabel network on the provided set of features.

    The features are un-formatted and are as-returned by
    `~mne_icalabel.iclabel.get_iclabel_features`.

    Parameters
    ----------
    images : array of shape (n_components, 1, 32, 32)
        The topoplot images.
    psds : array of shape (n_components, 1, 1, 100)
        The power spectral density features.
    autocorr : array of shape (n_components, 1, 1, 100)
        The autocorrelation features.

    Returns
    -------
    labels : array of shape (n_components, n_classes)
        The predicted numerical probability values for all labels in ICLabel output.
        Columns are ordered with 'Brain', 'Muscle', 'Eye', 'Heart',
        'Line Noise', 'Channel Noise', and 'Other'.
    """
    ica_network_file = files("mne_icalabel.iclabel").joinpath("assets/iclabelNet.pt")

    # Get network and load weights
    iclabel_net = ICLabelNet()
    iclabel_net.load_state_dict(torch.load(ica_network_file))

    # Format input and get labels
    labels = iclabel_net(*_format_input_for_torch(*_format_input(images, psds, autocorr)))
    labels = labels.detach().numpy()

    # outputs are ordered as in
    # https://github.com/sccn/ICLabel/blob/e8abc99e0c371ff49eff115cf7955fafc7f7969a/iclabel.m#L60-L62
    return labels
