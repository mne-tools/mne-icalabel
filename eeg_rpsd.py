import numpy as np
import math

def eeg_rpsd(pnts, srate, nfreqs, pct_data=100):
    # Clean input cutoff freq
    nyquist = math.floor(srate / 2)
    if nfreqs > nyquist:
        nfreqs = nyquist
    
    # ncomp = len(icaweights)
    n_points = min(pnts, srate)
    window = np.hamming(n_points)

    print(window.shape)
    

