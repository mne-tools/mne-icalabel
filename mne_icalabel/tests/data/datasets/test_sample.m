%% RAW - 'sample-raw.set'

% The file 'sample-raw.set' was obtained by using the sample EEGLAB
% dataset. The EOG channels have been dropped and the dataset has been
% cropped between 0 and 10 seconds. Before computing ICA, the data has
% been re-referenced to a common average.

% ----------------------------------------------
% sha1: 1137c8156c92427e4eac3420561f4226c7d1a9ab
% ----------------------------------------------

% Load
file_dataset = 'eeglab2022.0/sample_data/eeglab_data.set';
EEG = pop_loadset(file_dataset);
EEG = eeg_checkset(EEG);

% Drop non-EEG channel and crop dataset
idx = eeg_chaninds(EEG, {'EOG1', 'EOG2'});
EEG = pop_select(EEG, 'nochannel', idx, 'time', [0, 10]);

% Change .ref field from 'common' to 'averef' to skip buggy 'pop_reref.m'
EEG.ref = 'averef';

% Run ICA
EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1, 'interrupt', 'off');

% Save
pop_saveset(EEG, 'filename', 'sample-raw.set', 'savemode', 'onefile');

% ------------------------------------------------------------------------
% Load in Python
%{
from mne.io import read_raw
from mne.preprocessing import read_ica_eeglab

fname = 'sample-raw.set'
raw = mne.io.read_raw(fname, preload=True)
ica = read_ica_eeglab(fname)
%}


%% RAW - 'sample-short-raw.set'

% The file 'sample-raw.set' was obtained by using the sample EEGLAB
% dataset. The EOG channels have been dropped and the dataset has been
% cropped between 0 and 10 seconds. Before computing ICA, the data has
% been re-referenced to a common average.

% ----------------------------------------------
% sha1: 71c05706ac5531ae19fbc0235ddd71e089d8e82b
% ----------------------------------------------

% Load
file_dataset = 'eeglab2022.0/sample_data/eeglab_data.set';
EEG = pop_loadset(file_dataset);
EEG = eeg_checkset(EEG);

% Drop non-EEG channel and crop dataset
idx = eeg_chaninds(EEG, {'EOG1', 'EOG2'});
EEG = pop_select(EEG, 'nochannel', idx, 'time', [0, 3]);

% Change .ref field from 'common' to 'averef' to skip buggy 'pop_reref.m'
EEG.ref = 'averef';

% Run ICA
EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1, 'interrupt', 'off');

% Save
pop_saveset(EEG, 'filename', 'sample-short-raw.set', 'savemode', 'onefile');

% ------------------------------------------------------------------------
% Load in Python
%{
from mne.io import read_raw
from mne.preprocessing import read_ica_eeglab

fname = 'sample-short-raw.set'
raw = mne.io.read_raw(fname, preload=True)
ica = read_ica_eeglab(fname)
%}


%% RAW - 'sample-very-short-raw.set'

% The file 'sample-raw.set' was obtained by using the sample EEGLAB
% dataset. The EOG channels have been dropped and the dataset has been
% cropped between 0 and 10 seconds. Before computing ICA, the data has
% been re-referenced to a common average.

% ----------------------------------------------
% sha1: 5a65b589530af23c6db598627e552567440bdb2c
% ----------------------------------------------

% Load
file_dataset = 'eeglab2022.0/sample_data/eeglab_data.set';
EEG = pop_loadset(file_dataset);
EEG = eeg_checkset(EEG);

% Drop non-EEG channel and crop dataset
idx = eeg_chaninds(EEG, {'EOG1', 'EOG2'});
EEG = pop_select(EEG, 'nochannel', idx, 'time', [0, 0.5]);

% Change .ref field from 'common' to 'averef' to skip buggy 'pop_reref.m'
EEG.ref = 'averef';

% Run ICA
EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1, 'interrupt', 'off');

% Save
pop_saveset(EEG, 'filename', 'sample-very-short-raw.set', 'savemode', 'onefile');

% ------------------------------------------------------------------------
% Load in Python
%{
from mne.io import read_raw
from mne.preprocessing import read_ica_eeglab

fname = 'sample-very-short-raw.set'
raw = mne.io.read_raw(fname, preload=True)
ica = read_ica_eeglab(fname)
%}


%% EPOCHS - 'sample-epo.set'

% The file 'sample-epo.set' was obtained by using the sample EEGLAB
% dataset. The EOG channels have been dropped and the dataset has been
% cropped by selecting the 3 first 'rt' epochs with [tmin=0, tmax=1] (s).
% Before computing ICA, the data has been re-referenced to a common
% average.

% ----------------------------------------------
% sha1: 92aa789f6012b3b58ba7dd5be3a99d0a99e9c6e5
% ----------------------------------------------

% Load
file_dataset = 'eeglab2022.0/sample_data/eeglab_data.set';
EEG = pop_loadset(file_dataset);
EEG = eeg_checkset(EEG);

% Create epochs (trials)
[events, number] = eeg_eventtypes(EEG);
EEG = pop_epoch(EEG, {'rt'}, [0, 1]);

% Drop non-EEG channel and crop dataset
idx = eeg_chaninds(EEG, {'EOG1', 'EOG2'});
EEG = pop_select(EEG, 'nochannel', idx, 'trial', [1, 2, 3]);

% Change .ref field from 'common' to 'averef' to skip buggy 'pop_reref.m'
EEG.ref = 'averef';

% Run ICA
EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1, 'interrupt', 'off');

% Save
pop_saveset(EEG, 'filename', 'sample-epo.set', 'savemode', 'onefile');

% ------------------------------------------------------------------------
% Load in Python
%{
from mne import read_epochs_eeglab
from mne.preprocessing import read_ica_eeglab

fname = 'sample-epo.set'
epochs= mne.read_epochs_eeglab(fname)
ica = read_ica_eeglab(fname)
%}
