function [featuresmat, feature_label] = ICL_feature_extration_full(EEG, study_ID, dataset_ID)

if ~exist('study_ID', 'var') || isempty(study_ID)
    study_ID = nan;
end
if ~exist('dataset_ID', 'var') || isempty(dataset_ID)
    dataset_ID = nan;
end

% =========================================================================
% EEG stuff 
% =========================================================================

% % for testing only
% lower2 = 2;
% EEG = pop_subcomp(EEG, setdiff(1:size(EEG.icaweights,1), ...
%     randperm(size(EEG.icaweights,1), min(lower2,size(EEG.icaweights,1)))),0);

assert(~isempty(EEG.icawinv), ...
    'No ica weights in the current EEG dataset! Compute ICA on your data first.')
n_ic= size(EEG.icawinv,2); % ncps is number of components

% =========================================================================
% Convert to average reference
% =========================================================================

pop_reref(EEG,[],'exclude',setdiff(1:EEG.nbchan,EEG.icachansind));

% calculate ica activations if missing
if isempty(EEG.icaact)
    EEG.icaact = eeg_getdatact(EEG,'component',1:n_ic);
end

% =========================================================================
% Computing dipole
% =========================================================================

% 1 dipole
if isfield(EEG, 'dipfit') && ~isempty(EEG.dipfit) ...
        && isfield(EEG.dipfit, 'model') && ~isempty(EEG.dipfit.model) ...
        && length(EEG.dipfit.model) == n_ic
    % reuse existing dipoles
    ind2dp = cellfun(@(c) size(c, 1), {EEG.dipfit.model.posxyz}) == 2;
    
    % 2 dipole
    EEG.dipfit2 = getfield(pop_multifit( ...
        EEG, find(~ind2dp), 'threshold', 100, 'dipoles', 2), 'dipfit');
    
    % 1 dipole
    for it = find(ind2dp)
        EEG.dipfit.model(it).posxyz(2, :) = [];
        EEG.dipfit.model(it).momxyz(2, :) = [];
    end
    EEG = pop_multifit(EEG, find(ind2dp), 'threshold', 100, 'dipoles', 1);
    
else
    % recompute all
    try
        EEG = pop_multifit(EEG,[],'threshold',100);
    catch
        EEG = pop_dipfit_settings(EEG);
        EEG = pop_multifit(EEG,[],'threshold',100);
    end
    
    % 2 dipole
    EEG.dipfit2 = getfield(pop_multifit( ...
        EEG,[],'threshold',100,'dipoles',2),'dipfit');
end


% =========================================================================
% Creating structure for sasica 
% =========================================================================

optvals = struct('FASTER_blinkchans',     '',...
                'chancorr_channames',     '',...
                'chancorr_corthresh',     0.6,...
                'EOGcorr_Heogchannames',  [] ,...
                'EOGcorr_corthreshH',     'auto',  ...
                'EOGcorr_Veogchannames',  '',...
                'EOGcorr_corthreshV',     'auto',...
                'resvar_thresh',          15,...
                'SNR_snrcut',             1,...
                'SNR_snrBL',              [-Inf,0],...
                'SNR_snrPOI',             [0  Inf],...
                'trialfoc_focaltrialout', 'auto',...
                'focalcomp_focalICAout',  'auto',...
                'autocorr_autocorrint',    20,...
                'autocorr_dropautocorr',  'auto');            
            

% Enable/disable fields
fenable = struct('MARA_enable',      0,...
                 'FASTER_enable',    1,...
                 'ADJUST_enable',    1,...
                 'chancorr_enable',  1,...
                 'EOGcorr_enable',   0,...
                 'resvar_enable',    0,... % !!! changed!
                 'SNR_enable',       1,...
                 'trialfoc_enable',  0,...
                 'focalcomp_enable', 1,...
                 'autocorr_enable',  1);
        
fields = regexp(fieldnames(fenable),'[^_]*','match')';

for i_f = 1:length(fields)
    instr.(fields{i_f}{1}).(fields{i_f}{2}) = ...
        fenable.([fields{i_f}{1} '_' fields{i_f}{2}]);
end

fields2 = regexp(fieldnames(optvals),'[^_]*','match')';

for i_f = 1:numel(fields2)
    if ~isempty(strfind(fields2{i_f}{2},'channame'))
        if ischar(optvals.([fields2{i_f}{1} '_' fields2{i_f}{2}]))
            instr.(fields2{i_f}{1}).(fields2{i_f}{2}) = ...
                eval(['[chnb(''' optvals.([fields2{i_f}{1} '_' fields2{i_f}{2}]) ''')]']);
        else
            instr.(fields2{i_f}{1}).(fields2{i_f}{2}) = ...
                eval(['[chnb([' optvals.([fields2{i_f}{1} '_' fields2{i_f}{2}])  '])]']);
        end
    else
        try
            instr.(fields2{i_f}{1}).(fields2{i_f}{2}) = ...
                eval(['[' optvals.([fields2{i_f}{1} '_' fields2{i_f}{2}]) ']']);
        catch
            instr.(fields2{i_f}{1}).(fields2{i_f}{2}) = ...
                optvals.([fields2{i_f}{1} '_' fields2{i_f}{2}]) ;
        end

    end
end
instr.opts.noplot = 1;


% =========================================================================
% Running sasica
% =========================================================================

[EEG] = myeeg_SASICA(EEG,instr);
disp 'leaving sasica'


% =========================================================================
% Generating topos
% =========================================================================

topo = zeros(n_ic,740);
plotrad = zeros(n_ic,1);

disp 'saving topomap'
for i = 1:n_ic
    scalpmap_norm = EEG.icawinv(:,i)/sqrt(sum(EEG.icawinv(:,i).^2));
    [~,Zi,plotrad(i)] = topoplotFast( scalpmap_norm, EEG.chanlocs, 'chaninfo', EEG.chaninfo, ...
        'shading', 'interp', 'numcontour', 3,'electrodes','on','noplot','on');
    topo(i,:) = Zi(~isnan(Zi));
end
disp 'finished saving topomap'
EEG.reject.SASICA.topo = topo;
EEG.reject.SASICA.plotrad = plotrad;


% =========================================================================
% Generating psds
% =========================================================================

[EEG.reject.SASICA.spec, EEG.reject.SASICA.specvar, ...
    EEG.reject.SASICA.speckrt] = eeg_rpsd(EEG, 100);


% =========================================================================
% Generating autocorrs
% =========================================================================

if EEG.trials == 1
    if EEG.pnts / EEG.srate > 5
        EEG.reject.SASICA.autocorr = eeg_autocorr_welch(EEG);
    else
        EEG.reject.SASICA.autocorr = eeg_autocorr(EEG);
    end
else
    EEG.reject.SASICA.autocorr = eeg_autocorr_fftw(EEG);
end


% =========================================================================
% Generate and save feature vectors
% =========================================================================

[~, feature_labels] = genfvect;
featuresmat = nan(n_ic,length(feature_labels));

disp 'saving features'
for i = 1:n_ic
    [featuresmat(i,:), feature_label] = genfvect(EEG,study_ID,dataset_ID,i);
end

           
% =========================================================================
% =========================================================================
% =========================================================================
function [nb,channame,strnames] = chnb(channame, varargin)

% chnb() - return channel number corresponding to channel names in an EEG
%           structure
%
% Usage:
%   >> [nb]                 = chnb(channameornb);
%   >> [nb,names]           = chnb(channameornb,...);
%   >> [nb,names,strnames]  = chnb(channameornb,...);
%   >> [nb]                 = chnb(channameornb, labels);
%
% Input:
%   channameornb  - If a string or cell array of strings, it is assumed to
%                   be (part of) the name of channels to search. Either a
%                   string with space separated channel names, or a cell
%                   array of strings.
%                   Note that regular expressions can be used to match
%                   several channels. See regexp.
%                   If only one channame pattern is given and the string
%                   'inv' is attached to it, the channels NOT matching the
%                   pattern are returned.
%   labels        - Channel names as found in {EEG.chanlocs.labels}.
%
% Output:
%   nb            - Channel numbers in labels, or in the EEG structure
%                   found in the caller workspace (i.e. where the function
%                   is called from) or in the base workspace, if no EEG
%                   structure exists in the caller workspace.
%   names         - Channel names, cell array of strings.
%   strnames      - Channel names, one line character array.
narginchk(1, 2);
if nargin == 2
    labels = varargin{1};
else

    try
        EEG = evalin('caller','EEG');
    catch
        try
            EEG = evalin('base','EEG');
        catch
            error('Could not find EEG structure');
        end
    end
    if not(isfield(EEG,'chanlocs'))
        error('No channel list found');
    end
    EEG = EEG(1);
    labels = {EEG.chanlocs.labels};
end
if iscell(channame) || ischar(channame)

    if ischar(channame) || iscellstr(channame)
        if iscellstr(channame) && numel(channame) == 1 && isempty(channame{1})
            channame = '';
        end
        tmp = regexp(channame,'(\S*) ?','tokens');
        channame = {};
        for i = 1:numel(tmp)
            if iscellstr(tmp{i}{1})
                channame{i} = tmp{i}{1}{1};
            else
                channame{i} = tmp{i}{1};
            end
        end
        if isempty(channame)
            nb = [];
            return
        end
    end
    if numel(channame) == 1 && not(isempty(strmatch('inv',channame{1})))
        cmd = 'exactinv';
        channame{1} = strrep(channame{1},'inv','');
    else
        channame{1} = channame{1};
        cmd = 'exact';
    end
    nb = regexpcell(labels,channame,[cmd 'ignorecase']);

elseif isnumeric(channame)
    nb = channame;
    if nb > numel(labels)
        nb = [];
    end
end
channame = labels(nb);
strnames = sprintf('%s ',channame{:});
if not(isempty(strnames))
    strnames(end) = [];
end
