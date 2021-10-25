import numpy as np
import scipy.io as sio
from eeg_features import eeg_autocorr_fftw, eeg_features, eeg_rpsd, eeg_topoplot
import matplotlib.pyplot as plt


def testAutoCorr():
    """
    Tests the autocorrelation feature port.
    """
    corr_data = sio.loadmat('test_data/autocorr_data.mat')

    icaact = corr_data['icaact']
    srate = corr_data['srate'][0,0]
    trials = corr_data['trials'][0,0]
    pnts = corr_data['pnts']

    resamp = eeg_autocorr_fftw(icaact, trials, srate, pnts=pnts)

    matlab_resamp = sio.loadmat('test_data/autocorr_data.mat')['resamp']
    
    print('AutoCorr:', np.allclose(resamp, matlab_resamp, rtol=1e-7, atol=1e-7))


def testRPSD():
    """
    Tests the RPSD feature port.
    """
    rpsdData = sio.loadmat('test_data/rpsd_data.mat')

    # Inputs
    icaact = rpsdData['icaact']
    icaweights = rpsdData['icaweights']
    srate = rpsdData['srate'][0,0]
    trials = rpsdData['trials'][0,0]
    pnts = rpsdData['pnts'][0,0]

    # This is the correct MATLAB output to test against
    outMat = rpsdData['psdmed']

    # Subset is supposed to be random
    # We include it to remove having to replicate randomness
    subset = rpsdData['subset']

    psdmed = eeg_rpsd(icaact, icaweights, pnts, srate, trials, pct_data=100, subset=subset)
    print('PSD:', np.allclose(psdmed, outMat))


def testTopoplot(plot = False):
    """
    Tests the topoplot feature.

    Args:
        plot (bool, optional): Heatmap plot flag. Defaults to False.
    """
    topoplot_data = sio.loadmat('test_data/topoplot_data.mat')
    
    # Inputs
    icawinv = topoplot_data['icawinv']
    Rd = topoplot_data['Rd']
    Th = topoplot_data['Th']
    plotchans = topoplot_data['plotchans']
    
    # Python output
    i = 10
    Zi = eeg_topoplot(icawinv=icawinv[:,i:i+1], Rd=Rd, Th=Th, plotchans=plotchans)
    
    # Actual output
    temp_topo = topoplot_data['temp_topo']
    
    print('Topomap:', np.allclose(Zi, temp_topo, equal_nan=True)) # need equal_nan because there are nan values in both
    
    if plot:
        _,axes = plt.subplots(1,2)
        axes[0].imshow(Zi)
        axes[0].set_title('Python')
        axes[1].imshow(temp_topo)
        axes[1].set_title('Matlab')
        plt.show()
        
def testFull():
    full_data = sio.loadmat('test_data/full_data.mat')
    
    # Inputs
    icawinv = full_data['icawinv']
    Rd = full_data['Rd']
    Th = full_data['Th']
    plotchans = full_data['plotchans']
    topo = full_data['topo']
    
    icaact = full_data['icaact']
    icaweights = full_data['icaweights']
    srate = full_data['srate'][0,0]
    trials = full_data['trials'][0,0]
    pnts = full_data['pnts'][0,0]
    psd = full_data['psd']
    subset = full_data['subset']
    
    features = eeg_features(icaact=icaact, 
                            icaweights=icaweights, 
                            srate=srate, 
                            trials=trials, 
                            pnts=pnts, 
                            icawinv=icawinv, 
                            Th=Th, Rd=Rd, 
                            plotchans=plotchans,
                            subset=subset)
    matlab_features = full_data['features']
    
    # Test Topo
    print('Topo:', np.allclose(matlab_features[0,0], features[0]))
    
    # Test PSD
    print('PSD:', np.allclose(matlab_features[0,1], features[1]))
    
    # Test Autocorr
    print('Autocorr:', np.allclose(matlab_features[0,2], features[2]))
    


def main():
    # testAutoCorr()
    # testRPSD()
    # testTopoplot()
    testFull()


if __name__ == "__main__":
    main()

    

