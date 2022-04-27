%% RAW

% The files 'features-raw.mat' and 'feature-formatted-raw' were obtained 
% from the ICA decomposition in the 'sample-raw.set' dataset.

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
save('features-raw', 'features')

% Format
features{1} = cat(4, features{1}, -features{1}, features{1}(:, end:-1:1, :, :), -features{1}(:, end:-1:1, :, :));
features{2} = repmat(features{2}, [1 1 1 4]);
features{3} = repmat(features{3}, [1 1 1 4]);
save('features-formatted-raw', 'features')


%% EPOCHS

% The files 'features-epo.mat' and 'feature-formatted-epo' were obtained 
% from the ICA decomposition in the 'sample-epo.set' dataset.

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

% Format
features{1} = cat(4, features{1}, -features{1}, features{1}(:, end:-1:1, :, :), -features{1}(:, end:-1:1, :, :));
features{2} = repmat(features{2}, [1 1 1 4]);
features{3} = repmat(features{3}, [1 1 1 4]);
save('features-formatted-epo', 'features')
