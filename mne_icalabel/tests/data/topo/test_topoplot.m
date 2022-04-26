%% RAW - Single component

% The file 'topo1-raw' contains the output of 'topoplotFast.m' for the 
% first component in the 'sample-raw.set' dataset.

% ----------------------------------------------
% sha1:
% ----------------------------------------------

% Load
EEG = pop_loadset('sample-raw.set');
EEG = eeg_checkset(EEG);

% Calculate ICA activations
EEG.icaact = eeg_getica(EEG);
EEG.icaact = double(EEG.icaact);

it = 1;
Values = EEG.icawinv(:, it);
loc_file = EEG.chanlocs(EEG.icachansind);
[~, topo1, ~] = topoplotFast(Values, loc_file, 'noplot', 'on');

save('topo1-raw', 'topo1')


%% RAW - Topographic feature

% The file 'topo-feature-raw' contains the topographic feature retrieved
% from the 'sample-raw.set' dataset.

% ----------------------------------------------
% sha1:
% ----------------------------------------------

% Load
EEG = pop_loadset('sample-raw.set');
EEG = eeg_checkset(EEG);

% Calculate ICA activations
EEG.icaact = eeg_getica(EEG);
EEG.icaact = double(EEG.icaact);

% Compute topographic feature
ncomp = size(EEG.icawinv, 2);
topo = zeros(32, 32, 1, ncomp);
for it = 1:ncomp
    [~, temp_topo, plotrad] = ...
        topoplotFast(EEG.icawinv(:, it), EEG.chanlocs(EEG.icachansind), ...
        'noplot', 'on');
    temp_topo(isnan(temp_topo)) = 0;
    topo(:, :, 1, it) = temp_topo / max(abs(temp_topo(:)));
end

% cast
topo = single(topo);

% Export
save('topo-feature-raw', 'topo')