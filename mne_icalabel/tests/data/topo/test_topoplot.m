%% RAW

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