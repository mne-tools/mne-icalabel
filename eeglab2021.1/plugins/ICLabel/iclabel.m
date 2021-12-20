function EEG = iclabel(EEG, version)
%ICLABEL Function for EEG IC labeling
%   Label independent components using ICLabel.  Go to 
%   https://sccn.ucsd.edu/wiki/ICLabel for a tutorial on this plug-in. Go 
%   to labeling.ucsd.edu/tutorial/about for more information. To report a
%   bug or issue, please create an "Issue" post on the GitHub page at 
%   https://github.com/sccn/ICLabel/issues or send an email to 
%   eeglab@sccn.ucsd.edu.
% 
%   Inputs:
%       EEG: EEGLAB EEG structure. Must have an attached ICA decomposition.
%       version (optional): Version of ICLabel to use. Default
%       (recommended) version is used if passed 'default', '', or left
%       empty. Pass 'lite' to use ICLabelLite or 'beta' to use the original
%       beta version of ICLabel (only recommended for replicating old
%       results).
%
%   Results are stored in EEG.etc.ic_classifications.ICLabel. The matrix of
%   label vectors is stored under "classifications" and the cell array of
%   class names are stored under "classes". The version if ICLabel used is
%   stored under The matrix stored under "version". "classifications" is
%   organized with each column matching to the equivalent element in
%   "classes" and each row matching to the equivalent IC. For example, if
%   you want to see what percent ICLabel attributes IC 7 to the class
%   "eye", you would look at:
%       EEG.etc.ic_classifications.ICLabel.classifications(7, 3)
%   since EEG.etc.ic_classifications.ICLabel.classes{3} is "eye".


% check inputs
if ~exist('version', 'var') || isempty(version)
    version = 'default';
else
    version = lower(version);
end
assert(any(strcmp(version, {'default', 'lite', 'beta'})), ...
    ['Invalid network version choice. ' ...
     'Version must be one of the following: ' ...
     '''default'', ''lite'', or ''beta''.'])
if any(strcmpi(version, {'', 'default'}))
    flag_autocorr = true;
else
    flag_autocorr = false;
end
 
% check for ica
assert(isfield(EEG, 'icawinv') && ~isempty(EEG.icawinv), ...
    'You must have an ICA decomposition to use ICLabel')

% extract features
disp 'ICLabel: extracting features...'
features = ICL_feature_extractor(EEG, flag_autocorr);

% run ICL
disp 'ICLabel: calculating labels...'
labels = run_ICL(version, features{:});

% save into EEG
disp 'ICLabel: saving results...'
EEG.etc.ic_classification.ICLabel.classes = ...
    {'Brain', 'Muscle', 'Eye', 'Heart', ...
     'Line Noise', 'Channel Noise', 'Other'};
EEG.etc.ic_classification.ICLabel.classifications = labels;
EEG.etc.ic_classification.ICLabel.version = version;
    
