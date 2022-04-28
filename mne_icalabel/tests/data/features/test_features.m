%% RAW

% The files 'features-raw.mat' and 'feature-formatted-raw' were obtained
% from the ICA decomposition in the 'sample-raw.set' dataset.

% 'features-raw' -------------------------------
% sha1: cb5ce64a66b13bc8a8535b5108a0a11b74403e36
% ----------------------------------------------

% 'features-formatted-raw' ---------------------
% sha1: 721412c67aa98d8b8505edd4a48d7fda80edf8a5
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

% 'features-epo' -------------------------------
% sha1: 487e3f0c24f67fb59554c452a5a809128d3d23bc
% ----------------------------------------------

% 'features-formatted-epo' ---------------------
% sha1: 2d5f240819007df1d7d228f0065d69e6b3739a68
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
