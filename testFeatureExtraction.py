import numpy as np
import scipy.io as sio
from eeg_autocorr_fftw import eeg_autocorr_fftw

corrData = sio.loadmat('autoCorrData.mat')

icaact = corrData['icaact']
srate = corrData['srate'][0,0]
trials = corrData['trials'][0,0]
pnts = 384

resamp = eeg_autocorr_fftw(icaact, trials, srate, pnts=pnts)

matlab_resamp = sio.loadmat('resamp.mat')['resamp']

print(np.allclose(resamp, matlab_resamp, rtol=0, atol=0.8))