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
    rpsdData = sio.load('autoCorrData.mat')

    icaact = rpsdData['icaact']
    srate = rpsdData['srate'][0,0]
    trials = rpsdData['trials'][0,0]
    pnts = 384

    eeg_rpsd(icaweights, pnts, srate, nfreqs)

def main():
    # testAutoCorr()
    testRPSD()

if __name__ == "__main__":
    main()

    

