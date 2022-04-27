%% RAW - 'features-raw.mat'

% The file 'features-raw.mat' was obtained from the ICA decomposition in
% the 'sample-raw.set' dataset.

% ----------------------------------------------
% sha1:
% ----------------------------------------------

EEG = pop_loadset('sample-raw.set');
EEG = eeg_checkset(EEG);

% Calculate ICA activations
EEG.icaact = eeg_getica(EEG);
EEG.icaact = double(EEG.icaact);

% Extract features
features = ICL_feature_extractor(EEG, true);

% Export
save('features-raw', 'features')


%% EPOCHS - 'features-epo.mat'

% The file 'features-epo.mat' was obtained from the ICA decomposition in
% the 'sample-epo.set' dataset.

% ----------------------------------------------
% sha1:
% ----------------------------------------------

EEG = pop_loadset('sample-epo.set');
EEG = eeg_checkset(EEG);

% Calculate ICA activations
EEG.icaact = eeg_getica(EEG);
EEG.icaact = double(EEG.icaact);

% Extract features
features = ICL_feature_extractor(EEG, true);

% Export
save('features-epo', 'features')
