import numpy as np
import scipy.io as sio
from eeg_autocorr_fftw import eeg_autocorr_fftw

corrData = sio.loadmat('autoCorrData.mat')

icaact = corrData['icaact']
srate = corrData['srate'][0,0]
trials = corrData['trials'][0,0]
pnts = 384

# X = np.array([1,2,3,4,5])
# Y = np.fft.fft(X)

# print(np.abs(np.fft.ifft(Y)))

eeg_autocorr_fftw(icaact, trials, srate, pnts=384)