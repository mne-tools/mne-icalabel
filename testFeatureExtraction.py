import numpy as np
import scipy.io as sio
from eeg_autocorr_fftw import eeg_autocorr_fftw
from eeg_rpsd import eeg_rpsd
from eeg_topoplot import eeg_topoplot, gdatav4
import warnings


def testAutoCorr():
    """
    Tests the autocorrelation feature port.
    """
    corrData = sio.loadmat('autoCorrData.mat')

    icaact = corrData['icaact']
    srate = corrData['srate'][0,0]
    trials = corrData['trials'][0,0]
    pnts = 384

    resamp = eeg_autocorr_fftw(icaact, trials, srate, pnts=pnts)

    matlab_resamp = sio.loadmat('resamp.mat')['resamp']
    max_feat = np.max(matlab_resamp)
    relative_tol = float(input())

    print('AutoCorr:', np.allclose(resamp, matlab_resamp, rtol=0, atol=max_feat * relative_tol))


def testRPSD():
    """
    Tests the RPSD feature port.
    """
    rpsdData = sio.loadmat('rpsdData.mat')

    # Inputs
    icaact = rpsdData['icaact']
    icaweights = rpsdData['icaweights']
    srate = rpsdData['srate'][0,0]
    trials = rpsdData['trials'][0,0]
    pnts = rpsdData['pnts'][0,0]
    nfreqs = rpsdData['nfreqs'][0,0]

    # This is the correct MATLAB output to test against
    outMat = rpsdData['psdmed']

    # Subset is supposed to be random
    # We include it to remove having to replicate randomness
    subset = rpsdData['subset']

    psdmed = eeg_rpsd(icaact, icaweights, pnts, srate, nfreqs, trials, pct_data=100, subset=subset)
    print('PSD:', np.allclose(psdmed, outMat))


def testTopoplot():
    """
    Tests the Topoplot feature port.
    """
    topoplot_data = sio.loadmat('topoplot_data.mat')
    
    # Inputs
    icawinv = topoplot_data['icawinv']
    Rd = topoplot_data['Rd']
    Th = topoplot_data['Th']
    plotchans = topoplot_data['plotchans']
    
    Zi = eeg_topoplot(icawinv=icawinv[:,:1], Rd=Rd, Th=Th, plotchans=plotchans)
    
    temp_topo = topoplot_data['temp_topo']
    
    # print('Topomap:', np.allclose(Zi, temp_topo, rtol=1e-1, atol=1e-1))
    print(Zi)
    print()
    print(temp_topo)


def main():
    # testAutoCorr()
    # testRPSD()
    np.set_printoptions(edgeitems=30, linewidth=100000, precision=1)
    testTopoplot()

if __name__ == "__main__":
    main()

    

