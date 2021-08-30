%% Getting started with EEGLAB history
% The line below was added by us to locate data files
eeglab_path = fileparts(which('eeglab.m'));

% Start eeglab
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;

% Change option to process multiple datasets
pop_editoptions( 'option_storedisk', 0);

% Load the dataset (We modified the path manually here)
EEG = pop_loadset( 'eeglab_data.set', fullfile(eeglab_path, 'sample_data'));

% Load the channel location file, enabling automatic detection of channel file format'; We modified the path manually here
EEG.chanlocs=pop_chanedit(EEG.chanlocs, 'load',{ fullfile(eeglab_path, 'sample_data', 'eeglab_chan32.locs'), 'filetype', 'autodetect'});

% Store the dataset into EEGLAB
[ALLEEG EEG CURRENTSET ] = eeg_store(ALLEEG, EEG);

% High pass filter the data with cutoff frequency of 1 Hz.
EEG = pop_eegfilt( EEG, 1, 0, [], [0]); 

% Below, create a new dataset with the name filtered Continuous EEG Data
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, 'setname', 'filtered Continuous EEG Data');% Now CURRENTSET= 2
EEG = pop_reref( EEG, [], 'refstate',0); % Re-refrence the new dataset

% This might be a good time to add a comment to the dataset.
EEG.comments = pop_comments(EEG.comments,'','Dataset was highpass filtered at 1 Hz and rereferenced.',1);

% You can see the comments stored with the dataset either by typing >> EEG.comments or selecting the menu option Edit->About this dataset.
EEG = pop_epoch( EEG, { 'square' }, [-1 2], 'newname', 'Continuous EEG Data epochs', 'epochinfo', 'yes');

% Extract epochs time locked to the event - 'square', from 1 second before to 2 seconds after those time-locking events.
% Now, either overwrite the parent dataset, if you don't need the continuous version any longer, or create a new dataset
%(by removing the 'overwrite', 'on' option in the function call below).
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, 'setname', 'Continuous EEG Data epochs', 'overwrite', 'on');
EEG = pop_rmbase( EEG, [-1000 0]); % Remove baseline

% Add a description of the epoch extraction to EEG.comments.
EEG.comments = pop_comments(EEG.comments,'','Extracted ''square'' epochs [-1 2] sec, and removed baseline.',1);
[ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);  %Modify the dataset in the EEGLAB main window
eeglab redraw % Update the EEGLAB window to view changes

[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
EEG = pop_loadset( 'eeglab_data.set', fullfile(eeglab_path, 'sample_data')); 
EEG.chanlocs=pop_chanedit(EEG.chanlocs, 'load',{ fullfile(eeglab_path, 'sample_data', 'eeglab_chan32.locs'), 'filetype', 'autodetect'});
EEG = pop_eegfilt( EEG, 1, 0, [], [0]); 
EEG = pop_reref( EEG, [], 'refstate',0);
EEG.comments = pop_comments(EEG.comments,'','Dataset was highpass filtered at 1 Hz and rereferenced.',1);
EEG = pop_epoch( EEG, { 'square' }, [-1 2], 'newname', 'Continuous EEG Data epochs', 'epochinfo', 'yes');
EEG = pop_rmbase( EEG, [-1000 0]);
EEG.comments = pop_comments(EEG.comments,'','Extracted ''square'' epochs [-1 2] sec, and removed baseline.',1);
[ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG, 1);
eeglab redraw 

%% Reduce sampling rate
% Reduce the sampling rate to 128 Hz (the above example was already sampled at 128 Hz'')
EEG = pop_resample( EEG, 128);

% Save it as a new dataset with the name Continuous EEG Data resampled
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, 'setname', 'Continuous EEG Data resampled');

% Now on the GUI we returned to the previous dataset (before downsampling)
EEG = eeg_retrieve(ALLEEG, 1); CURRENTSET = 1;

%% Plot ERP maps 
% Every 100 ms from 0 ms to 500 ms [0:100:500]
% with the plot title - 'ERP image', in 2 rows and 3 columns. Below, the 0 means do not plot dipoles.
% Plot marks showing the locations of the electrodes on the scalp maps.
pop_topoplot(EEG,1, [0:100:500] , 'Topographic plot', [2:3] ,0, 'electrodes', 'on');

%% Topographic plot
% Define variables:
times = [0:100:500];
pos = round(eeg_lat2point(times/1000, 1, EEG.srate, [EEG.xmin EEG.xmax]));

% Convert times to points (or >pos = round( (times/1000-EEG.xmin)/(EEG.xmax-EEG.xmin) * (EEG.pnts-1))+1;)
% See the event tutorial for more information on processing latencies
mean_data = mean(EEG.data(:,pos,:),3);

% Average over all trials in the desired time window (the third dimension of 
% EEG.data allows to access different data trials). See tutorial about data structures
maxlim = max(mean_data(:));
minlim = min(mean_data(:));
maplimits = [ -max(maxlim, -minlim) max(maxlim, -minlim)]; % Get the data range for scaling the map colors

% Plot the scalp map series
figure
for k = 1:6
    sbplot(2,3,k);
    % A more flexible version of subplot
    topoplot( mean_data(:,k), EEG.chanlocs, 'maplimits', maplimits, 'electrodes', 'on', 'style', 'both');
    title([ num2str(times(k)) ' ms']);
end
cbar; % A more flexible version of Matlab colorbar
