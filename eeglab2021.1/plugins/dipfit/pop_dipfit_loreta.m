% pop_dipfit_loreta() - localize ICA components using eLoreta
%
% Usage:
%  >> EEGOUT = pop_dipfit_gridsearch( EEGIN ); % pop up interactive window
%  >> EEGOUT = pop_dipfit_gridsearch( EEGIN, comps );
%
% Inputs:
%   EEGIN     - input dataset
%   comps     - [integer array] component indices
%
% Outputs:
%   EEGOUT      output dataset
%
% Authors: Arnaud Delorme, SCCN, La Jolla 2018
%
% More help: type help ft_sourceanalysis and help ft_sourceplot for 
%            parameters to use these functions.

% Copyright (C) 2018 Arnaud Delorme
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 2 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

function com = pop_dipfit_loreta(EEG, select, varargin)

if nargin < 1
    help pop_dipfit_loreta;
    return;
end

if ~plugin_askinstall('Fieldtrip-lite', 'ft_dipolefitting'), return; end;

EEGOUT = EEG;
com = '';

if ~isfield(EEG, 'chanlocs')
    error('No electrodes present');
end

if ~isfield(EEG, 'icawinv')
    error('No ICA components to fit');
end
        
if ~isfield(EEG, 'dipfit')
    error('General dipole fit settings not specified');
end

if ~isfield(EEG.dipfit, 'sourcemodel') || isempty(EEG.dipfit.sourcemodel)
    error('You need to compute a Liedfield matrix first');
end
if ~isfield(EEG.dipfit, 'coordformat') || ~strcmpi(EEG.dipfit.coordformat, 'MNI')
    error('For this function, you must use the template BEM model MNI in dipole fit settings');
end

dipfitdefs;
if nargin < 2
     uilist = { { 'style' 'text'        'string'  [ 'Enter indices of components ' 10 '(one figure generated per component)'] } ...
                { 'style' 'edit'        'string'  '1' } ...
                { 'style' 'text'        'string'  'ft_sourceanalysis parameters' } ...
                { 'style' 'edit'        'string'  '''method'', ''eloreta''' } ...
                { 'style' 'text'        'string'  'ft_sourceplot parameters' } ...
                { 'style' 'edit'        'string'  '''method'', ''slice''' } };
     optiongui = { 'geometry', { 1 1 1 1 1 1 }, 'geomvert', [2 1 1 1 1 1], 'uilist', uilist, 'helpcom', 'pophelp(''pop_dipfit_loreta'')', ...
                  'title', 'Localization of ICA components using eLoreta -- pop_dipfit_loreta()' };
	[result] = inputgui( optiongui{:});
    
    if isempty(result)
        % user pressed cancel
        return
    end
    
    % decode parameters
    select = eval( [ '[' result{1} ']' ]);
    try, params1 = eval( [ '{' result{2} '}' ]); catch, error('ft_sourceanalysis parameters badly formated'); end
    try, params2 = eval( [ '{' result{3} '}' ]); catch, error('ft_sourceplot parameters badly formated'); end
    options = { 'ft_sourceanalysis_params' params1 'ft_sourceplot_params' params2 };
else
    options = varargin;
end

if ~isempty(setdiff(select, [1:size(EEG.icaweights,1)]))
    error('Some component indices out of range');
end

g = finputcheck(options, { 'ft_sourceanalysis_params'  'cell'    {}         { 'method' 'eloreta' };
                           'ft_sourceplot_params'      'cell'    []         { 'method' 'slice' } }, 'pop_dipfit_loreta');
if isstr(g), error(g); end;

%% compute spectral params (only need to be done once to get the right structures)
dataPre = eeglab2fieldtrip(EEG, 'preprocessing', 'dipfit');
cfg = [];
cfg.method    = 'mtmfft';
cfg.output    = 'powandcsd';
cfg.tapsmofrq = 2;
cfg.foilim    = [10 10];
freqPre = ft_freqanalysis(cfg, dataPre);
freqPre = rmfield(freqPre,'labelcmb');

%% read headmodel
p = fileparts(which('eeglab'));
headmodel = load('-mat', EEG.dipfit.hdmfile);
headmodel = headmodel.vol;

%% prepare leadfield matrix
cfg                 = [];
cfg.elec            = freqPre.elec;
cfg.headmodel       = headmodel;
cfg.reducerank      = 2;
cfg.sourcemodel.unit       = 'mm';
cfg.resolution = 5;
cfg.channel         = { 'all' };
fprintf('\nGrid creation below is only to assess inside/outside brain voxels, use DIPFIT to create Leadfield matrix\n');
grid = ft_prepare_sourcemodel(cfg);

% source localization
cfg              = struct(g.ft_sourceanalysis_params{:}); 
cfg.frequency    = 18;  

sourcemodeltmp = EEG.dipfit.sourcemodel;
if isfield(sourcemodeltmp, 'tri')
    fprintf(2, '\nYou are using a surface source model. Plotting interpolated 3-D volume is not recommended\n\n');
end

% find position futher than 5 mm from source model
% same as above but specific to the current model
% also will not interpolate white matter if no voxel as present there
% grid.inside = grid.inside;
% grid.inside(:) = true;
% for iPos = 1:size(grid.pos,1)
%     if all(sqrt(sum(bsxfun(@minus, tmp.pos, grid.pos(iPos,:)).^2,2)) > 10)
%         grid.inside(iPos) = false;
%     end
% end

sourcemodeltmp.inside    = [ sourcemodeltmp.inside;grid.inside(~grid.inside)];
sourcemodeltmp.pos(end+1:end+sum(~grid.inside),:) = grid.pos(~grid.inside,:);
sourcemodeltmp.leadfield = [ sourcemodeltmp.leadfield cell(1, sum(~grid.inside)) ];

cfg.sourcemodel = sourcemodeltmp;
cfg.headmodel    = headmodel;
cfg.dics.projectnoise = 'yes';
cfg.dics.lambda       = 0;

for iSelect = select(:)'
    freqPre.powspctrm = EEG.icawinv(:,iSelect).*EEG.icawinv(:,iSelect);
    freqPre.crsspctrm = EEG.icawinv(:,iSelect)*EEG.icawinv(:,iSelect)';
    
    sourcePost_nocon = ft_sourceanalysis(cfg, freqPre);

    %% load MRI and plot
    mri = load('-mat', EEG.dipfit.mrifile);
    mri = ft_volumereslice([], mri.mri);
    
    cfg2            = [];
    cfg2.downsample = 2;
    cfg2.parameter = 'avg.pow';
    sourcePost_nocon.oridimord = 'pos';
    sourcePost_nocon.momdimord = 'pos';
    sourcePostInt_nocon  = ft_sourceinterpolate(cfg2, sourcePost_nocon , mri);
    
    cfg2              = struct(g.ft_sourceplot_params{:});
    cfg2.funparameter = 'avg.pow';
    ft_sourceplot(cfg2,sourcePostInt_nocon);
    textsc(sprintf('eLoreta source localization of component %d power', iSelect), 'title');
end

%% history
disp('Done');
com = sprintf('pop_dipfit_loreta(EEG, %s);', vararg2str( { select }));
