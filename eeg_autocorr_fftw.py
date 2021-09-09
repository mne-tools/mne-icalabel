import math
import numpy as np
import scipy.signal as ss
from scipy.fft import fft, ifft

def eeg_autocorr_fftw(icaact, trials, srate, pnts, pct_data = 100):
    nfft = 2**(math.ceil(math.log2(abs(2*pnts - 1))))
    ac = np.zeros((len(icaact), nfft, trials), dtype=np.float64)
    
    for it in range(len(icaact)):
        X = fft(icaact[it:it+1,:,:], n = nfft, axis = 1)
        ac[it:it+1,:,:] = np.power(np.abs(X),2)
    
    ac = np.abs(ifft(np.mean(ac, axis=2), n=None, axis=1)) # ifft
    
    if pnts < srate:
        ac = np.hstack((ac[:,0:pnts], np.zeros((len(ac), srate - pnts + 1))))
    else:
        ac = ac[:,0:srate+1]

    ac = ac[:,0:srate+1] / ac[:,1][:,None]

    resamp = ss.resample_poly(ac.T, 100, srate).T

    return resamp[:,1:]
    