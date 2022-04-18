try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

import torch.nn as nn
import torch
import numpy as np
from numpy.typing import ArrayLike


class ICLabelNetImg(nn.Module):
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

    def forward(self, x):
        return self.sequential(x)


class ICLabelNetPSDS(nn.Module):
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

    def forward(self, x):
        return self.sequential(x)


class ICLabelNetAutocorr(nn.Module):
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

    def forward(self, x):
        return self.sequential(x)


class ICLabelNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.img_conv = ICLabelNetImg()
        self.psds_conv = ICLabelNetPSDS()
        self.autocorr_conv = ICLabelNetAutocorr()

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
    def reshape_fortran(x: torch.Tensor, shape) -> torch.Tensor:
        if len(x.shape) > 0:
            x = x.permute(*reversed(range(len(x.shape))))
        return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))  # noqa

    def reshape_concat(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.reshape_fortran(tensor, [-1, 1, 1, 100])
        tensor = torch.cat([tensor, tensor, tensor, tensor], 1)
        tensor = torch.cat([tensor, tensor, tensor, tensor], 2)
        tensor = tensor.permute((0, 3, 1, 2))
        return tensor

    def forward(
        self, images: torch.Tensor, psds: torch.Tensor, autocorr: torch.Tensor
    ) -> torch.Tensor:
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


def format_input(images: ArrayLike, psd: ArrayLike, autocorr: ArrayLike):
    formatted_images = np.concatenate(
        (images, -1 * images, np.flip(images, axis=1), np.flip(-1 * images, axis=1)),
        axis=3,
    )
    formatted_psd = np.tile(psd, (1, 1, 1, 4))
    formatted_autocorr = np.tile(autocorr, (1, 1, 1, 4))

    formatted_images = torch.from_numpy(np.transpose(formatted_images, (3, 2, 0, 1)))
    formatted_psd = torch.from_numpy(np.transpose(formatted_psd, (3, 2, 0, 1)))
    formatted_autocorr = torch.from_numpy(
        np.transpose(formatted_autocorr, (3, 2, 0, 1))
    )

    return formatted_images, formatted_psd, formatted_autocorr


def run_iclabel(images: ArrayLike, psds: ArrayLike, autocorr: ArrayLike) -> ArrayLike:
    submodule = "mne_icalabel"
    filename = "assets/iclabelNet.pt"
    ica_network_file = files(submodule).joinpath(filename)

    # Get network and load weights
    iclabel_net = ICLabelNet()
    iclabel_net.load_state_dict(torch.load(ica_network_file))

    # Format input and get labels
    labels = iclabel_net(*format_input(images, psds, autocorr))
    return labels.detach().numpy()
