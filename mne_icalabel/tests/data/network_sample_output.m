clc
% Importing the sample fif file and executing ICA decomposition on the
% input
[ALLEEG , ~, ~, ALLCOM] = eeglab;
EEG = pop_fileio('../tests/data/ica-test-raw.fif', 'dataformat','auto');
[ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, 0,'setname','ica_set_test','gui','off'); 
EEG = eeg_checkset( EEG );
pop_eegplot( EEG, 1, 1, 1);
EEG = eeg_checkset( EEG );
EEG = pop_runica(EEG, 'icatype', 'runica', 'extended',1,'interrupt','on');
[ALLEEG, EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);


% Extracting the features post ICA decomposition
path
which ICL_feature_extractor.m ;
features = ICL_feature_extractor(EEG, true);
images = features{1};
psds = features{2};
autocorrs = features{3};

% Loading Network
netStruct = load('netICL.mat')
net = dagnn.DagNN.loadobj(netStruct)

%% format network inputs
images = cat(4, images, -images, images(:, end:-1:1, :, :), -images(:, end:-1:1, :, :));
psds = repmat(psds, [1 1 1 4]);
autocorrs = repmat(autocorrs, [1 1 1 4]);
input = {
    'in_image', single(images), ...
    'in_psdmed', single(psds), ...
    'in_autocorr', single(autocorrs)
};

% check path (sometimes mex file not first which create a problem)
path2vl_nnconv = which('-all', 'vl_nnconv');
if isempty(findstr('mex', path2vl_nnconv{1})) && length(path2vl_nnconv) > 1
    addpath(fileparts(path2vl_nnconv{2}));
end

% Saving the features to compare with torch features
save('../tests/data/matlab_images', 'images');
save('../tests/data/matlab_psds', 'psds');
save('../tests/data/matlab_autocorrs', 'autocorrs');

%Applying forward pass
net.eval(input);
out = net.getVar(net.getOutputs()).value;

%% extract result
labels = squeeze(net.getVar(net.getOutputs()).value)';

labels = reshape(mean(reshape(labels', [], 4), 2), 7, [])';



save('../tests/data/matlab_labels', 'labels');

eeglab redraw;
