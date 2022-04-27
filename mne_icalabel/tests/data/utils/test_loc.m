%% RAW - 'loc-raw.mat'

% The file 'loc-raw.mat' was obtained from the 'sample-raw.set' dataset and
% contains the 'th' and 'rd' electrode coordinates.

% ----------------------------------------------
% sha1:
% ----------------------------------------------

EEG = pop_loadset('sample-raw.set');
EEG = eeg_checkset(EEG);

loc_file = EEG.chanlocs(EEG.icachansind);
[~, ~, th, rd, ~] = readlocs( loc_file );

loc = struct(...
    'th', th, ...
    'rd', rd);
save('loc-raw', 'loc');


%% EPOCHS - 'loc-epo.mat'

% The file 'loc-epo.mat' was obtained from the 'sample-epo.set' dataset and
% contains the 'th' and 'rd' electrode coordinates.

% ----------------------------------------------
% sha1:
% ----------------------------------------------

EEG = pop_loadset('sample-epo.set');
EEG = eeg_checkset(EEG);

loc_file = EEG.chanlocs(EEG.icachansind);
[~, ~, th, rd, ~] = readlocs( loc_file );

loc = struct(...
    'th', th, ...
    'rd', rd);
save('loc-epo', 'loc');
