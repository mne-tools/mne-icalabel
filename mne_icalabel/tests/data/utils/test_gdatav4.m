% The file 'gdatav4-raw.mat' and 'gdatav4-epo.mat' were obtained by adding
% a breakpoint to line 960 in 'topoplotFast.m', running line 960 and
% saving:
%
%     gdatav4 = struct(...
%         'inty', double(inty), ...
%         'intx', double(intx), ...
%         'intValues', double(intValues), ...
%         'yi', double(yi), ...
%         'xi', double(xi), ...
%         'Xi', Xi, ...
%         'Yi', Yi, ...
%         'Zi', Zi);
%     save('gdatav4-***', 'gdatav4');


%% RAW

% ----------------------------------------------
% sha1: e2ccab702a0a752cd6e6dc69ff3af50dc499daf8
% ----------------------------------------------

EEG = pop_loadset('sample-raw.set');
EEG = eeg_checkset(EEG);

it = 1;
Values = EEG.icawinv(:, it);
loc_file = EEG.chanlocs(EEG.icachansind);
[~, temp_topo, ~] = topoplotFast(Values, loc_file, 'noplot', 'on');


%% EPOCHS

% ----------------------------------------------
% sha1: 8f7a2327e81d182d2aff92ccce36cb790fa1bf1e
% ----------------------------------------------

EEG = pop_loadset('sample-epo.set');
EEG = eeg_checkset(EEG);

it = 1;
Values = EEG.icawinv(:, it);
loc_file = EEG.chanlocs(EEG.icachansind);
[~, temp_topo, ~] = topoplotFast(Values, loc_file, 'noplot', 'on');
