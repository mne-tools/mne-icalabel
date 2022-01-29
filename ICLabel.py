import torch.nn as nn
import torch
import numpy as np
import mne


class ICLabelNetImg(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=128,
                               kernel_size=(4, 4),
                               padding=1,
                               stride=(2, 2))
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.conv2 = nn.Conv2d(in_channels=128,
                               out_channels=256,
                               kernel_size=(4, 4),
                               padding=1,
                               stride=(2, 2))
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.conv3 = nn.Conv2d(in_channels=256,
                               out_channels=512,
                               kernel_size=(4, 4),
                               padding=1,
                               stride=(2, 2))
        self.relu3 = nn.LeakyReLU(negative_slope=0.2)
        self.sequential = nn.Sequential(self.conv1, self.relu1,
                                        self.conv2, self.relu2,
                                        self.conv3, self.relu3)

    def forward(self, x):
        return self.sequential(x)


class ICLabelNetPSDS(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=128,
                               kernel_size=(1, 3),
                               padding=(0, 1),
                               stride=(1, 1))
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.conv2 = nn.Conv2d(in_channels=128,
                               out_channels=256,
                               kernel_size=(1, 3),
                               padding=(0, 1),
                               stride=(1, 1))
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.conv3 = nn.Conv2d(in_channels=256,
                               out_channels=1,
                               kernel_size=(1, 3),
                               padding=(0, 1),
                               stride=(1, 1))
        self.relu3 = nn.LeakyReLU(negative_slope=0.2)
        self.sequential = nn.Sequential(self.conv1, self.relu1,
                                        self.conv2, self.relu2,
                                        self.conv3, self.relu3)

    def forward(self, x):
        return self.sequential(x)


class ICLabelNetAutocorr(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=128,
                               kernel_size=(1, 3),
                               padding=(0, 1),
                               stride=(1, 1))
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.conv2 = nn.Conv2d(in_channels=128,
                               out_channels=256,
                               kernel_size=(1, 3),
                               padding=(0, 1),
                               stride=(1, 1))
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.conv3 = nn.Conv2d(in_channels=256,
                               out_channels=1,
                               kernel_size=(1, 3),
                               padding=(0, 1),
                               stride=(1, 1))
        self.relu3 = nn.LeakyReLU(negative_slope=0.2)
        self.sequential = nn.Sequential(self.conv1, self.relu1,
                                        self.conv2, self.relu2,
                                        self.conv3, self.relu3)

    def forward(self, x):
        return self.sequential(x)


class ICLabelNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.img_conv = ICLabelNetImg()
        self.psds_conv = ICLabelNetPSDS()
        self.autocorr_conv = ICLabelNetAutocorr()

        self.conv = nn.Conv2d(in_channels=712,
                              out_channels=7,
                              kernel_size=(4, 4),
                              padding=0,
                              stride=(1, 1))
        self.softmax = nn.Softmax(dim=1)

        self.seq = nn.Sequential(self.conv, self.softmax)

    @staticmethod
    def reshape_fortran(x: torch.Tensor, shape) -> torch.Tensor:
        if len(x.shape) > 0:
            x = x.permute(*reversed(range(len(x.shape))))
        return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))

    def reshape_concat(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.reshape_fortran(tensor, [-1, 1, 1, 100])
        tensor = torch.concat([tensor, tensor, tensor, tensor], 1)
        tensor = torch.concat([tensor, tensor, tensor, tensor], 2)
        tensor = torch.permute(tensor, (0, 3, 1, 2))
        return tensor

    def forward(self, images: torch.Tensor, psds: torch.Tensor, autocorr: torch.Tensor) -> torch.Tensor:
        out_img = self.img_conv(images)
        out_psds = self.psds_conv(psds)
        out_autocorr = self.autocorr_conv(autocorr)

        # PSDS reshape, concat, permute
        psds_perm = self.reshape_concat(out_psds)

        # Autocorr reshape, concat, permute
        autocorr_perm = self.reshape_concat(out_autocorr)

        concat = torch.concat([out_img, psds_perm, autocorr_perm], 1)

        labels = self.seq(concat)

        labels = labels.squeeze()
        labels = self.reshape_fortran(labels.permute(1, 0), [-1, 4])
        labels = torch.mean(labels, 1)
        labels = self.reshape_fortran(labels, [7, -1])
        labels = labels.permute(1, 0)

        return labels


def format_input(images: np.ndarray, psd: np.ndarray, autocorr: np.ndarray):
    formatted_images = np.concatenate((images,
                                       -1 * images,
                                       np.flip(images, axis=1),
                                       np.flip(-1 * images, axis=1)),
                                      axis=3)
    formatted_psd = np.repeat(psd, 4, axis=3)
    formatted_autocorr = np.repeat(autocorr, 4, axis=3)

    formatted_images = torch.from_numpy(
        np.transpose(formatted_images, (3, 2, 0, 1)))
    formatted_psd = torch.from_numpy(np.transpose(formatted_psd, (3, 2, 0, 1)))
    formatted_autocorr = torch.from_numpy(
        np.transpose(formatted_autocorr, (3, 2, 0, 1)))

    return formatted_images, formatted_psd, formatted_autocorr


def run_iclabel(images: np.ndarray, psds: np.ndarray, autocorr: np.ndarray) -> np.ndarray:
    # Get network and load weights
    iclabel_net = ICLabelNet()
    iclabel_net.load_state_dict(torch.load('iclabelNet.pt'))

    # Format input and get labels
    labels = iclabel_net(*format_input(images, psds, autocorr))
    return labels.detach().numpy()


def mne_to_eeglab_locs(raw: mne.io.BaseRaw):
    """Obtain EEGLab-like spherical coordinate from EEG channel positions.

    TODO: @JACOB:
    - Where is (0,0,0) defined in MNE vs EEGLab?
    - some text description of how the sphere coordinates differ between MNE
    and EEGLab.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Instance of raw object with a `mne.montage.DigMontage` set with
        ``n_channels`` channel positions.

    Returns
    -------
    Rd : np.array of shape (1, n_channels)
        Angle in spherical coordinates of each EEG channel.
    Th : np.array of shape (1, n_channels)
        Degree in spherical coordinates of each EEG channel.
    """
    def _sph2topo(_theta, _phi):
        """
        Convert spherical coordinates to topo.
        """
        az = _phi
        horiz = _theta
        angle = -1 * horiz
        radius = (np.pi / 2 - az) / np.pi
        return angle, radius

    def _cart2sph(_x, _y, _z):
        """
        Convert cartesian coordinates to spherical.
        """
        azimuth = np.arctan2(_y, _x)
        elevation = np.arctan2(_z, np.sqrt(_x ** 2 + _y ** 2))
        r = np.sqrt(_x ** 2 + _y ** 2 + _z ** 2)
        # theta,phi,r
        return azimuth, elevation, r

    # get the channel position dictionary
    montage = raw.get_montage()
    positions = montage.get_positions()
    ch_pos = positions['ch_pos']

    # get locations as a 2D array
    locs = np.vstack(list(ch_pos.values()))

    # Obtain carthesian coordinates
    x = locs[:, 1]

    # be mindful of the nose orientation in eeglab and mne
    # TODO: @Jacob, please expand on this.
    y = -1 * locs[:, 0]
    # see https://github.com/mne-tools/mne-python/blob/24377ad3200b6099ed47576e9cf8b27578d571ef/mne/io/eeglab/eeglab.py#L105
    z = locs[:, 2]

    # Obtain Spherical Coordinates
    sph = np.array([_cart2sph(x[i], y[i], z[i]) for i in range(len(x))])
    theta = sph[:, 0]
    phi = sph[:, 1]

    # Obtain Polar coordinates (as in eeglab)
    topo = np.array([_sph2topo(theta[i], phi[i]) for i in range(len(theta))])
    rd = topo[:, 1]
    th = topo[:, 0]

    return rd.reshape([1, -1]), np.degrees(th).reshape([1, -1])


def mne_iclabel(epochs):
    from mne.preprocessing import ICA
    from eeg_features import eeg_features

    ica = ICA(n_components=None, max_iter='auto',
              random_state=97, method='infomax',
              fit_params=dict(extended=True))
    ica.fit(epochs)

    icaact = ica.get_sources(epochs).get_data()
    print(icaact.shape)
    icaact = np.transpose(icaact, [1, 2, 0])

    # weights (unmixing matrix)
    icaweights = ica.unmixing_matrix_

    icawinv = np.linalg.pinv(ica.unmixing_matrix_ @ ica.pca_components_.T)

    srate = 128
    pnts = 384
    trials = 80

    rd, th = mne_to_eeglab_locs(epochs)
    
    features = eeg_features(icaact=icaact,
                            trials=trials,
                            srate=srate,
                            pnts=pnts,
                            subset=None,
                            icaweights=icaweights,
                            icawinv=icawinv,
                            th=th,
                            rd=rd)
    
    topo = features[0].astype(np.float32)
    psds = features[1].astype(np.float32)
    autocorr = features[2].astype(np.float32)

    labels = run_iclabel(topo, psds, autocorr)

    return ica, labels


def main():
    import mne
    import os
    # import scipy.io as sio
    #
    # features = sio.loadmat('features.mat')['features']
    #
    # images = features[0, 0]
    # psds = features[0, 1]
    # autocorrs = features[0, 2]
    #
    # print(images.shape, psds.shape, autocorrs.shape)
    # print(images.dtype, psds.dtype, autocorrs.dtype)
    #
    # labels = run_iclabel(images, psds, autocorrs)
    #
    # Print out
    np.set_printoptions(precision=4)
    np.set_printoptions(suppress=True)
    #
    # labels_mat = sio.loadmat('labels.mat')['labels']
    #
    # # Print labels side by side
    # print('PyTorch                                            MATLAB')
    # for pytorch_out, matlab_out in zip(labels, labels_mat):
    #     print(pytorch_out, matlab_out)

    eeglab_file = os.path.join(
        'eeglab2021.1', 'sample_data', 'eeglab_data_epochs_ica.set')

    epochs = mne.io.read_epochs_eeglab(eeglab_file)

    ica, labels = mne_iclabel(epochs)

    print(labels)


if __name__ == "__main__":
    main()
