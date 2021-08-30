% eeg_icalabelstat - show some stats about IC label
%
% Input:
%  EEG       - EEGLAB dataset
%  threshold - threshold for belonging to a category (default 0.9 
%              corresponding to 90%
%
% Author: A. Delorme

function eeg_icalabelstat(EEG, threshold)

if nargin < 1
    help eeg_icalabelstat;
    return;
end
if nargin < 2
    threshold = 0.9;
end

ics = EEG.etc.ic_classification.ICLabel;
if length(threshold) == 1, threshold(1:length(ics.classes)) = threshold; end
for iIC = 1:length(ics.classes)
    ICfound = find(ics.classifications(:,iIC) > threshold(iIC));
    fprintf('%30s: %d/%d components at %d%% threshold\n', sprintf('IClabel class "%s"',ics.classes{iIC}), length(ICfound), size(ics.classifications,1), round(threshold(iIC)*100));
end