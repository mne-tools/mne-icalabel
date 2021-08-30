%% Working with events
eeglab_path = fileparts(which('eeglab.m')); % get EEGLAB path
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab; % start EEGLAB
pop_editoptions( 'option_storedisk', 0); % Change option to process multiple datasets
EEG = pop_loadset( 'eeglab_data.set', fullfile(eeglab_path, 'sample_data')); % load data

%% Adjust event latency
for iEvent=1:length(EEG.event)
     EEG.event(iEvent).latency = EEG.event(iEvent).latency + 10;
end
[ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG, CURRENTSET); % Store dataset

%% Add new cue events to a loaded dataset 0.1 second before time-locking event
nevents = length(EEG.event);
for index = 1 : nevents
    if ischar(EEG.event(index).type) && strcmpi(EEG.event(index).type, 'square')
    % Add events relative to existing events
        EEG.event(end+1) = EEG.event(index); % Add event to end of event list
        % Specifying the event latency to be 0.1 sec before the referent event (in real data points)
        EEG.event(end).latency = EEG.event(index).latency - 0.1*EEG.srate;
        EEG.event(end).type = 'cue'; % Make the type of the new event cue
    end
end

EEG = eeg_checkset(EEG, 'eventconsistency'); % Check all events for consistency
[ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG, CURRENTSET); % Store dataset
eeglab redraw % Redraw the main EEGLAB window
