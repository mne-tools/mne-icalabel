This is a reference implementation of the Artifact Subspace Reconstruction (ASR) method in MATLAB. 

For convenience a few wrapper functions that call the ASR core functions are included in 
the sub-directory called "extras". These are not part of the ASR method but are useful to get started or 
can serve as code samples. See also readme-extras.txt for basic usage examples of those functions.


Using the core implementation directly
======================================

The reference implementation may require the Signal Processing toolbox or alternatively a pre-computed 
IIR filter kernel if spectrally weighted statistics should be used (this is an optional feature; the filter 
kernel can be set to A=1 and B=1 to disable spectral weighting).

In all cases the signal that is passed to the method should be zero-mean, i.e., first high-pass filtered if necessary. 
In the following code samples the data is randomly generated with zero mean.

The general calling convention of the method is:

1)  Calibrate the parameters using the asr_calibrate function and some reasonably clean 
    calibration data, e.g., resting-state EEG, as in the following code snippet. 
    The recommended data length is ca. 1 minute or longer, the absolute minimum would be ca. 15 seconds. 
    There are optional parameters as documented in the function, in particular a tunable threshold parameter 
    governing the aggressiveness of the cleaning (although the defaults are sensible for testing purposes).

    calibdata = randn(20,10000);          % simulating 20-channel, 100-second random data at 100 Hz
    state = asr_calibrate(calibdata,100)  % calibrate the parameters of the method (sampling rate 100Hz)


2a) Apply the processing function to data that shall be cleaned, 
    either in one large block (entire recording) as follows...

    rawdata = randn(20,1000000);                  % simulating random data to process
    cleandata = asr_process(rawdata,100,state);   % apply the processing to the data (sampling rate 100Hz); see documentation of the function for optional parameters.

    Also note that for offline processing the method may be calibrated on the same data that should also be 
    cleaned, as long as the fraction of contaminated data is lower than a theoretical upper bound of 
    50% (a conservative empirical estimate would be approx. 30%). For extremely contaminated data one may 
    extract one or more clean segments from the data and calibrate on those.


2b) ... or alternatively apply the processing function to data in an online/incremental (chunk-by-chunk) 
    fashion, as follows:

    while 1
        newchunk = randn(20,50);                               % here using 0.5-second chunks; the chunk length is near-arbitrary and may vary, but transitions will be sharper for very short chunks.
        [cleanchunk,state] = asr_process(newchunk,100,state);  % apply the processing to the data and update the filter state (as in MATLAB's filter())
    end

	
