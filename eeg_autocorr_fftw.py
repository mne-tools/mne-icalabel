import math
import numpy as np

def eeg_autocorr_fftw(icaact, trials, srate, pnts, pct_data = 100):
    nfft = 2^math.ceil(math.log2(abs(2*pnts - 1)))
    ac = np.zeros((len(icaact), nfft, trials))
    
    for it in range(len(icaact)):
        X = np.fft.fft(icaact[it,:,:], n = nfft, axis = 0)
        ac[it,:,:] = np.power(np.abs(X),2)
    
    ac = np.fft.ifft(np.mean()) # ifft

    print(ac.shape)


