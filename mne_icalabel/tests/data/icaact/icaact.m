%% RAW - 'icaact-raw.mat'

% The file 'icaact-raw.mat' was obtained from the ICA decomposition in the
% 'sample-raw.set' dataset.

% ------------------------------------------------------------------------
% sha1:
% ------------------------------------------------------------------------

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

% ------------------------------------------------------------------------
% sha1:
% ------------------------------------------------------------------------

% Load
EEG = pop_loadset('sample-epo.set');
EEG = eeg_checkset(EEG);

% Calculate ICA activations
EEG.icaact = eeg_getica(EEG);
EEG.icaact = double(EEG.icaact);

% Save
icaact = EEG.icaact;
save('icaact-epo', 'icaact');
