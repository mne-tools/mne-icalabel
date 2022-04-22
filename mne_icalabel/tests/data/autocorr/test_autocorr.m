%% Test 'eeg_autocorr' for short raw instances

% The file 'autocorr-short-raw.mat' was obtained by using the EEGLAB sample
% dataset.

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
% sample dataset.

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
