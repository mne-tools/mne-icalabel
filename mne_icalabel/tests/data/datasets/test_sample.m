%% RAW - 'sample-raw.set'

% The file 'sample-raw.set' was obtained by using the sample EEGLAB 
% dataset. The EOG channels have been dropped and the dataset has been 
% cropped between 0 and 10 seconds.

% ----------------------------------------------
% sha1: 055fea07348f1f379fe7fb62bc1379f0b948f43a
% ----------------------------------------------

% Load
file_dataset = 'eeglab2022.0/sample_data/eeglab_data.set';
EEG = pop_loadset(file_dataset);
EEG = eeg_checkset(EEG);

% Drop non-EEG channel and crop dataset
idx = eeg_chaninds(EEG, {'EOG1', 'EOG2'});
EEG = pop_select(EEG, 'nochannel', idx, 'time', [0, 10]);

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

%% EPOCHS - 'sample-epo.set'

% The file 'sample-epo.set' was obtained by using the sample EEGLAB 
% dataset. The EOG channels have been dropped and the dataset has been 
% cropped by selecting the 3 first 'rt' epochs with [tmin=0, tmax=1] (s).

% ----------------------------------------------
% sha1: e9c7968c8a151f758a8dc7461976f17cba10c4f7
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
