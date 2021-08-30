% pop_icflag - Flag components as atifacts
%
% Usage:
%   EEG = pop_icflag(EEG, thresh)
%
% Input
%   EEG    - EEGLAB dataset
%   thresh - [7x2 float] array with threshold values with limits to include
%            for selection as artifacts. The 6 categories are (in order) Brain, Muscle,
%            Eye, Heart, Line Noise, Channel Noise, Other.
%
% Example: 
% % the threshold below will only select component if they are in the 
% % eye category with at least 90% confidence
% threshold = [0 0;0 0; 0.9 1; 0 0; 0 0; 0 0; 0 0];
% EEG = pop_icflag(EEG, threshold)
%
% % the threshold below will select component if they are in the brain
% % category with less than 20% confidence
% threshold = [0 0.2;0 0; 0 0; 0 0; 0 0; 0 0; 0 0];
% EEG = pop_icflag(EEG, threshold)
%
% Author: Arnaud Delorme

function [EEG,com] = pop_icflag(EEG, thresh)

if nargin < 1
    help pop_icflag
    return
end

% check IC label has been done
try 
    classification = EEG(1).etc.ic_classification.ICLabel.classifications;
    if isempty(classification), err; end
catch
    disp('No labeling found, use Tools > Classify components using ICLabel > Label components first')
    return;
end

if nargin < 2
    if length(EEG) == 1
        eeg_icalabelstat(EEG);
    end
    
    cat     = { 'Brain'  'Muscle'  'Eye'  'Heart'  'Line Noise'  'Channel Noise'  'Other' };
    defaultMin = { ''       '0.9'     '0.9'  ''       ''            ''               '' };
    defaultMax = { ''       '1'       '1'    ''       ''            ''               '' };
    
    geom = [2 0.4 0.4];
    row  = { { 'style' 'text' 'string' 'Probability range for ' } { 'style' 'edit' 'string' '' } { 'style' 'edit' 'string' '' } };
    allRows = { { 'style' 'text' 'string' 'Select range for flagging component for rejection' 'fontweight' 'bold' } {} {  'style' 'text' 'string' 'Min' } {  'style' 'text' 'string' 'Max' } };
    allGeom = { 1 geom };
    for iCat = 1:length(cat)
        tmpRow = row;
        tmpRow{1}{end} = [ tmpRow{1}{end} '"' cat{iCat} '"' ];
        tmpRow{2}{end} = defaultMin{iCat};
        tmpRow{3}{end} = defaultMax{iCat};
        allRows = { allRows{:} tmpRow{:} };
        allGeom{end+1} = geom;
    end
    
    res = inputgui(allGeom, allRows);
    if isempty(res)
        com = '';
        return
    end
    
    thresh = cellfun(@str2double, res, 'uniformoutput', false);
    thresh(cellfun(@isempty, thresh)) = { NaN };
    thresh = [ thresh{:} ];
    thresh = reshape(thresh, 2, 7)';
end

if length(EEG) > 1
    [ EEG, com ] = eeg_eval( 'pop_icflag', EEG, 'params', { thresh } );
else
    % perform rejection
    flagReject = zeros(1,size(EEG.icaweights,1))';
    for iCat = 1:7
        tmpReject  = EEG.etc.ic_classification.ICLabel.classifications(:,iCat) > thresh(iCat,1) & EEG.etc.ic_classification.ICLabel.classifications(:,iCat) < thresh(iCat,2);
        flagReject = flagReject | tmpReject;
    end
    EEG.reject.gcompreject = flagReject;
    fprintf('%d components flagged for rejection, to reject them use Tools > Remove components from data\n', sum(flagReject));
    com = sprintf('EEG = pop_icflag(EEG, %s);',vararg2str(thresh));
end

