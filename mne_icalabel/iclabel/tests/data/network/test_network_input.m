% The file 'network_input.mat' contains the input provided to the forward
% pass of the matconvnet version of the ICLabel neural network. The
% features were extracted from the 'sample-raw.set' dataset.

% ----------------------------------------------
% sha1: f157823dc75cf656c89dd082a9efc83c270bcee7
% ----------------------------------------------

% Load
EEG = pop_loadset('sample-raw.set');
EEG = eeg_checkset(EEG);

% Calculate ICA activations
EEG.icaact = eeg_getica(EEG);
EEG.icaact = double(EEG.icaact);

% Extract features
features = ICL_feature_extractor(EEG, true);
images = features{1};
psds = features{2};
autocorrs = features{3};

% Format features
images = cat(4, images, -images, images(:, end:-1:1, :, :), -images(:, end:-1:1, :, :));
psds = repmat(psds, [1 1 1 4]);
autocorrs = repmat(autocorrs, [1 1 1 4]);
input = {
    'in_image', single(images), ...
    'in_psdmed', single(psds), ...
    'in_autocorr', single(autocorrs)
};

% Save
save('network_input', 'input');
