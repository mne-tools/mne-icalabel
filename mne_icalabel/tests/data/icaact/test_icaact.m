%% RAW - 'icaact-raw.mat'

% The file 'icaact-raw.mat' was obtained from the ICA decomposition in the
% 'sample-raw.set' dataset.

% ----------------------------------------------
% sha1: afbc5a9d6210b15dbc4fd766cdac8518165bbbdf
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
% sha1: 2893be99b7f9fb991f8e800960c07040a402cc1d
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
