import numpy as np
import scipy.io as sio
from eeg_autocorr_fftw import eeg_autocorr_fftw
from eeg_rpsd import eeg_rpsd

def testAutoCorr():
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

def testDipoleFit():
    pass

def main():
    # testAutoCorr()
    testRPSD()

if __name__ == "__main__":
    main()

    

