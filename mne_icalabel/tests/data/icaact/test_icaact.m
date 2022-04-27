%% RAW - 'icaact-raw.mat'

% The file 'icaact-raw.mat' was obtained from the ICA decomposition in the
% 'sample-raw.set' dataset.

% ----------------------------------------------
% sha1: 43161dc1bd347a1bdee88e877a325af24d46f948
% ----------------------------------------------

% Load
EEG = pop_loadset('sample-raw.set');
EEG = eeg_checkset(EEG);

% Calculate ICA activations
EEG.icaact = eeg_getica(EEG);
EEG.icaact = double(EEG.icaact);

% Save
icaact = EEG.icaact;
save('icaact-raw', 'icaact');


%% EPOCHS - 'icaact-epo.mat'

% The file 'icaact-epo.mat' was obtained from the ICA decomposition in the
% 'sample-epo.set' dataset.

% ----------------------------------------------
% sha1: 8eaffe74ffdc48df2e4e42c9c430da5f83a350bb
% ----------------------------------------------

% Load
EEG = pop_loadset('sample-epo.set');
EEG = eeg_checkset(EEG);

% Calculate ICA activations
EEG.icaact = eeg_getica(EEG);
EEG.icaact = double(EEG.icaact);

% Save
icaact = EEG.icaact;
save('icaact-epo', 'icaact');
