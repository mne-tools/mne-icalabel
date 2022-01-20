import numpy as np
import scipy.io as sio
from eeg_features import eeg_autocorr_fftw, eeg_features, eeg_rpsd, eeg_topoplot
import matplotlib.pyplot as plt


def test_autocorr():
    """
    Tests the autocorrelation feature port.
    """
    corr_data = sio.loadmat('test_data/autocorr_data.mat')

    icaact = corr_data['icaact']
    srate = corr_data['srate'][0, 0]
    trials = corr_data['trials'][0, 0]
    pnts = corr_data['pnts']

    resamp = eeg_autocorr_fftw(icaact, trials, srate, pnts=pnts)

    matlab_resamp = sio.loadmat('test_data/autocorr_data.mat')['resamp']

    print('AutoCorr:', np.allclose(resamp, matlab_resamp, rtol=1e-7, atol=1e-7))


def test_rpsd():
    """
    Tests the RPSD feature port.
    """
    rpsdData = sio.loadmat('test_data/rpsd_data.mat')

    # Inputs
    icaact = rpsdData['icaact']
    icaweights = rpsdData['icaweights']
    srate = rpsdData['srate'][0, 0]
    trials = rpsdData['trials'][0, 0]
    pnts = rpsdData['pnts'][0, 0]

    # This is the correct MATLAB output to test against
    outMat = rpsdData['psd']

    # Subset is supposed to be random
    # We include it to remove having to replicate randomness
    subset = rpsdData['subset'] - 1

    psdmed = eeg_rpsd(icaact, icaweights, pnts, srate, trials, pct_data=100, subset=subset)
    print('PSD:', np.allclose(psdmed, outMat))


def test_topoplot(plot=False):
    """
    Tests the topoplot feature.

    Args:
        plot (bool, optional): Heatmap plot flag. Defaults to False.
    """
    topoplot_data = sio.loadmat('test_data/topoplot_data.mat')

    # Inputs
    icawinv = topoplot_data['icawinv']
    rd = topoplot_data['Rd']
    th = topoplot_data['Th']
    plotchans = np.squeeze(topoplot_data['plotchans']) - 1

    print(rd)
    print(th)
    # Python output
    i = 10
    zi = eeg_topoplot(icawinv=icawinv[:, i:i + 1], Rd=rd, Th=th, plotchans=plotchans)

    # Actual output
    temp_topo = topoplot_data['temp_topo']

    print('Topomap:', np.allclose(zi, temp_topo, equal_nan=True))  # need equal_nan because there are nan values in both

    if plot:
        _, axes = plt.subplots(1, 2)
        axes[0].imshow(zi)
        axes[0].set_title('Python')
        axes[1].imshow(temp_topo)
        axes[1].set_title('Matlab')
        plt.show()


def main():
    test_autocorr()
    test_rpsd()
    test_topoplot()


if __name__ == "__main__":
    main()
