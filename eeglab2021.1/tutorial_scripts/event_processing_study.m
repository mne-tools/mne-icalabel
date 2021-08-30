%% Modify the events of datasets for creating STUDY designs

% load the epoched tutorial dataset
eeglab_path = fileparts(which('eeglab.m')); % get EEGLAB path
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab; % start EEGLAB
pop_editoptions( 'option_storedisk', 0); % Change option to process multiple datasets
EEG = pop_loadset( 'eeglab_data_epochs_ica.set', fullfile(eeglab_path, 'sample_data')); % load data
[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG);

% scan all datasets and modify events
% there is only one here so it is for illustration purpose
commands = {}; % for building the STUDY
for iDat = 1:length(ALLEEG)
     for iEvent = 1:length(ALLEEG(iDat).event)-1
           curEvent  = ALLEEG(iDat).event(iEvent); % current event
           nextEvent = ALLEEG(iDat).event(iEvent+1); % next event

           % only find reaction time event following time-locking events (TLE) within the same epoch
           if strcmpi( curEvent.type, 'square') && strcmpi( nextEvent.type, 'rt') && nextEvent.epoch == curEvent.epoch
                ALLEEG(iDat).event(iEvent).rt = (nextEvent.latency - curEvent.latency)/ALLEEG(iDat).srate * 1000; % latency of reaction time in ms
           end

     end
     % save dataset
     fileName = fullfile( ALLEEG(iDat).filepath, [ ALLEEG(iDat).setname(1:end-4) '_rtevents.set' ]);
     ALLEEG(iDat).saved = 'no';
     ALLEEG(iDat) = pop_saveset(ALLEEG, fileName);
     
     % add to list of dataset to build STUDY
     if isempty( ALLEEG(iDat).subject),  ALLEEG(iDat).subject = sprintf('S%2.2d', iDat); end % create subject name
     commands = { commands{:} 'index' iDat 'load' fileName 'subject' ALLEEG(iDat).subject }; 
end

% create study
[STUDY, ALLEEG] = std_editset( [], [], 'commands', commands,'updatedat','off' );
CURRENTSTUDY = true;
eeglab redraw

% Use menu item STUDY -> Select/Edit STUDY design then 
% under "Edit the independent variables for this design", press "New"
% You should be able to select event field 'rt' for creating designs
