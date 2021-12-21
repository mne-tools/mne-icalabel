import pytest
import os
import importlib.resources

import numpy as np
import scipy.io as sio
from numpy.testing import assert_array_almost_equal, assert_array_equal
import mne
from mne.preprocessing import ICA

from mne_icalabel.ica_features import autocorr_fftw, rpsd, topoplot
from eeg_features import eeg_autocorr_fftw, eeg_rpsd, eeg_topoplot


# load in test data from original Matlab ICLabel
corr_data_file_path = str(importlib.resources.files(
    'mne_icalabel.tests').joinpath('data/autocorr_data.mat'))
rpsd_data_file_path = str(importlib.resources.files(
    'mne_icalabel.tests').joinpath('data/rpsd_data.mat'))
topoplot_data_file_path = str(importlib.resources.files(
    'mne_icalabel.tests').joinpath('data/topoplot_data.mat'))

corr_data = sio.loadmat(corr_data_file_path)
rpsd_data = sio.loadmat(rpsd_data_file_path)
topoplot_data = sio.loadmat(topoplot_data_file_path)
# full_data = sio.loadmat('./data/full_data.mat')


def _create_test_ica_component():
    sample_data_folder = mne.datasets.sample.data_path()
    sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                        'sample_audvis_filt-0-40_raw.fif')
    raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False)
    # Here we'll crop to 60 seconds and drop gradiometer channels for speed
    raw.crop(tmax=60.).pick_types(meg='mag', eeg=True, stim=True, eog=True)
    raw.drop_channels(raw.info['bads'])
    raw.drop_channels(raw.ch_names[:-20])
    raw.load_data()

    filt_raw = raw.copy().filter(l_freq=1., h_freq=None, verbose=False)
    # fit ICA
    ica = ICA(n_components=len(raw.ch_names) - 1,
              max_iter='auto', random_state=97, verbose=False)
    ica.fit(filt_raw, verbose=False)
    return ica, filt_raw


def test_eeg_autocorr_fftw():
    """Test autocorrelation feature function is correct."""
    ica, raw = _create_test_ica_component()

    # sfreq can be float
    sfreq = raw.info['sfreq']

    # get the ICA waveforms
    ica_raw = ica.get_sources(raw)
    ica_act = np.atleast_3d(ica_raw.get_data())

    autocorr_feature = autocorr_fftw(ica_act, sfreq)
    test_autocorr_feature = eeg_autocorr_fftw(
        ica_act, 0, srate=int(sfreq), pnts=len(raw.times))
    assert_array_equal(autocorr_feature, test_autocorr_feature)

    # test against EEGlab's matlab test data
    icaact = corr_data['icaact']
    srate = corr_data['srate'][0, 0]
    resamp = autocorr_fftw(icaact, srate)
    matlab_resamp = sio.loadmat('test_data/autocorr_data.mat')['resamp']
    assert_array_almost_equal(resamp, matlab_resamp)


@pytest.mark.skip(reason='Doesntwork')
def test_rpsd_feature():
    """Test PSD feature generation.

    TODO: doesn't work.
    """
    ica, raw = _create_test_ica_component()

    # sfreq can be float
    sfreq = raw.info['sfreq']

    # get the ICA waveforms
    ica_weights = ica.mixing_matrix_
    ica_raw = ica.get_sources(raw)
    ica_act = np.atleast_3d(ica_raw.get_data())

    test_rpsd_feature = eeg_rpsd(
        ica_act, icaweights=ica_weights, trials=1, srate=int(sfreq), pnts=len(raw.times))
    rpsd_feature = rpsd(ica_act, sfreq)
    assert_array_equal(rpsd_feature, test_rpsd_feature)


@pytest.mark.skip(reason='Doesntwork')
def test_topoplot_feature():
    """Test topoplot feature generation."""
    ica, raw = _create_test_ica_component()

    # get the ICA waveforms
    ica_weights = ica.mixing_matrix_
    ica_sphere = ica.pca_components_

    # compute the inputs for topoplot
    ica_winv = np.linalg.pinv(ica_weights.dot(ica_sphere))
    theta = np.arange(ica.n_components_)
    rho = np.arange(ica.n_components_)

    # topoplot_feature = topoplot(ica_winv, theta, rho)
    # test_topoplot_feature = eeg_topoplot(ica_winv, theta, rho)
    # assert_array_equal(topoplot_feature, test_topoplot_feature)

    # Test against Matlab's EEGLab test data
    icawinv = topoplot_data['icawinv']
    Rd = topoplot_data['Rd']
    Th = topoplot_data['Th']
    plotchans = topoplot_data['plotchans']
    expected_topo = topoplot_data['temp_topo']

    print(expected_topo.shape)
    print(Rd.shape, Th.shape)
    print(plotchans)
    print(icawinv.shape)
    # Python output
    i = 10
    Zi = eeg_topoplot(icawinv=icawinv[:, i:i + 1], Rd=Rd, Th=Th,
                      plotchans=plotchans)
    assert_array_equal(Zi, expected_topo)
