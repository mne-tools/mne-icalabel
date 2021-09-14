import numpy as np
import math
from scipy.fft import fft

def eeg_rpsd(icaact, icaweights, pnts, srate, nfreqs, trials, pct_data=100, subset = None):
    # Clean input cutoff freq
    nyquist = math.floor(srate / 2)
    if nfreqs > nyquist:
        nfreqs = nyquist
    
    ncomp = len(icaweights)
    n_points = min(pnts, srate)
    window = np.hamming(n_points).reshape(1,-1)[:,:,np.newaxis]

    cutoff = math.floor(pnts / n_points) * n_points
    index = np.ceil(np.arange(0,cutoff - n_points+1,n_points / 2)).astype(np.int64).reshape(1,-1) + np.arange(0,n_points).reshape(-1,1)
    
    n_seg = index.shape[1] * trials
    if subset is None:
        subset = np.random.permutation(n_seg)[:math.ceil(n_seg * pct_data / 100)]
    subset -= 1 # because matlab uses indices starting at 1
    subset = np.squeeze(subset)

    psdmed = np.zeros((ncomp, nfreqs))
    denom = srate * np.sum(np.power(window,2))
    for it in range(ncomp):
        temp = icaact[it, index, :].reshape(1,index.shape[0], n_seg, order='F')
        temp = temp[:,:,subset] * window
        temp = fft(temp, n_points, 1)
        temp = temp * np.conjugate(temp)
        temp = temp[:, 1:nfreqs + 1, :] * 2 / denom
        if nfreqs == nyquist:
            temp[:,-1,:] /= 2
        psdmed[it, :] = 20 * np.real(np.log10(np.median(temp, axis=2)))

    return psdmed
