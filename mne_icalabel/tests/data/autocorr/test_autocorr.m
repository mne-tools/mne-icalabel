%% Test 'eeg_autocorr_welch' for raw instances

% The file 'autocorr-raw.mat' was obtained by using the EEGLAB sample
% dataset 'sample-raw.set'.

% ----------------------------------------------
% sha1: 1367b5203dd1f97f2878ce26d0f55c5453074663
% ----------------------------------------------

% Load
EEG = pop_loadset('sample-raw.set');
EEG = eeg_checkset(EEG);

% Calculate ICA activations
EEG.icaact = eeg_getica(EEG);
EEG.icaact = double(EEG.icaact);

% Retrieve autocorr
autocorr = eeg_autocorr_welch(EEG);

% Reshape and cast
autocorr = single(permute(autocorr, [3 2 4 1]));

% Save
save('autocorr-raw', 'autocorr');


%% Test 'eeg_autocorr' for short raw instances

% The file 'autocorr-short-raw.mat' was obtained by using the EEGLAB sample
% dataset 'sample-short-raw.set'.

% ----------------------------------------------
% sha1: 344e069e252d1189f600c14c52ace16ba6ba7d37
% ----------------------------------------------

% Load
EEG = pop_loadset('sample-short-raw.set');
EEG = eeg_checkset(EEG);

% Calculate ICA activations
EEG.icaact = eeg_getica(EEG);
EEG.icaact = double(EEG.icaact);

% Retrieve autocorr
autocorr = eeg_autocorr(EEG);

% Reshape and cast
autocorr = single(permute(autocorr, [3 2 4 1]));

% Save
save('autocorr-short-raw', 'autocorr');


%% Test 'eeg_autocorr' for very short raw instances

% The file 'autocorr-very-short-raw.mat' was obtained by using the EEGLAB
% sample dataset 'sample-very-short-raw.set'.

% ----------------------------------------------
% sha1: 9f080284e6e6e17e58dbf0812f8afac8e96bde15
% ----------------------------------------------

% Load
EEG = pop_loadset('sample-very-short-raw.set');
EEG = eeg_checkset(EEG);

% Calculate ICA activations
EEG.icaact = eeg_getica(EEG);
EEG.icaact = double(EEG.icaact);

% Retrieve autocorr
autocorr = eeg_autocorr(EEG);

% Reshape and cast
autocorr = single(permute(autocorr, [3 2 4 1]));

% Save
save('autocorr-very-short-raw', 'autocorr');


%% Test 'eeg_autocorr_fftw' for epoch instances

% The file 'autocorr-very-short-raw.mat' was obtained by using the EEGLAB
% sample dataset 'sample-epo.set'.

% ----------------------------------------------
% sha1: a658b34a7dec10d4cce7cdf715511c5da179d280
% ----------------------------------------------

% Load
EEG = pop_loadset('sample-epo.set');
EEG = eeg_checkset(EEG);

% Calculate ICA activations
EEG.icaact = eeg_getica(EEG);
EEG.icaact = double(EEG.icaact);

% Retrieve autocorr
autocorr = eeg_autocorr_fftw(EEG);

% Reshape and cast
autocorr = single(permute(autocorr, [3 2 4 1]));

% Save
save('autocorr-epo', 'autocorr');
