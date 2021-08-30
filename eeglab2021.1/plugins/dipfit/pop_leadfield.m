% pop_leadfield - compute Leadfield Matrix
%
% Usage:
%  EEG = pop_leadfield(EEG, 'key', 'val', ...);
%
% Inputs:
%  EEG - EEGLAB dataset
%
% Required inputs:
%  'sourcemodel' - [string] source model file
%
% Optional inputs:
%  'sourcemodel2mni' - [9x float] homogeneous transformation matrix to convert
%                      sourcemodel to MNI space.
%  'downsample'      - 1 downsampling of the source model. Valid only for
%                      volumetric models.
%
% Output:
%  EEG - EEGLAB dataset with field 'dipfit.leadfield' containing the Leadfield matrix.
%
% Author: Arnaud Delorme, UCSD, 2021
%
% Example
%   p = fileparts(which('eeglab')); % path
%   EEG = pop_leadfield(EEG,  'sourcemodel', fullfile(p, 'functions', 'supportfiles', ...
%   'head_modelColin27_5003_Standard-10-5-Cap339.mat'), 'sourcemodel2mni', ...
%   [0 -26.6046230000 -46 0.1234625600 0 -1.5707963000 1000 1000 1000]);
%
% Use pop_roi_act(EEG) to compute activity

% Copyright (C) Arnaud Delorme, arnodelorme@gmail.com
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
%
% 1. Redistributions of source code must retain the above copyright notice,
% this list of conditions and the following disclaimer.
%
% 2. Redistributions in binary form must reproduce the above copyright notice,
% this list of conditions and the following disclaimer in the documentation
% and/or other materials provided with the distribution.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
% THE POSSIBILITY OF SUCH DAMAGE.

% TO DO - Arno
% - Centralize reading head mesh and Atlas (there might be a function in
% Fieldtrip to do that) ft_read_volume ft_read_mesh
% - Make compatible with all Fieldtrip and FSL Atlases
% - Downsampling of Atlas - check bug submitted to Fieldtrip
% - Plot inside(blue) vs outside(red) voxels for source volume

function [EEG,com] = pop_leadfield(EEG, varargin)

% define source models
p  = fileparts(which('eeglab.m'));
roi(1).label = 'Surface source model: Colin27 (with Desikan-Kilianny atlas)';
roi(1).file  = fullfile( p, 'functions', 'supportfiles', 'head_modelColin27_5003_Standard-10-5-Cap339.mat');
roi(1).align = [0 -24 -45 0 0 -1.5707963 1000 1000 1000];
roi(1).enable = 'off';
roi(1).scale  = NaN;
roi(1).atlasliststr = { 'Desikan-Kiliany (68 ROIs)' };
roi(1).atlaslist    = { 'Desikan-Kiliany' };
roi(1).atlasind  = 1;

p  = fileparts(which('pop_roi_activity.m'));
roi(2).label = 'Surface source model: Use Brainstorm ICBM152 (with Desikan-Kilianny atlas)';
roi(2).file  = fullfile(p, 'tess_cortex_mid_low_2000V.mat');
roi(2).align = [0 -24 -45 0 0 -1.5707963000 1000 1000 1000];
roi(2).enable = 'off';
roi(2).scale  = NaN;
[ roi(2).atlasliststr, roi(2).atlaslist] = getatlaslist(roi(2).file);
roi(2).atlasind  = 2;

roi(3).label = 'Volumetric source model: LORETA-KEY';
roi(3).file  = fullfile(p, 'LORETA-Talairach-BAs.mat');
roi(3).align = [];
roi(3).enable = 'off';
roi(3).scale  = NaN;
roi(3).atlasliststr = { 'LORETA-Talairach-BAs (44 x 2 ROIs)' };
roi(3).atlaslist    = { 'LORETA-Talairach-BAs' };
roi(3).atlasind  = 1;

p  = fileparts(which('ft_defaults.m'));
roi(4).label = 'Volumetric source model: AFNI with TTatlas+tlrc atlas (Fieldtrip)';
roi(4).file  = fullfile(p, 'template','atlas','afni','TTatlas+tlrc.HEAD');
roi(4).align = [ ];
roi(4).enable = 'off';
roi(4).scale  = 4;
roi(4).atlasliststr = { '' };
roi(4).atlaslist    = { '' };
roi(4).atlasind  = 1;

roi(5).label = 'Custom source model';
roi(5).file  = '';
roi(5).align = [];
roi(5).enable = 'on';
roi(5).scale  = 1;
roi(5).atlasliststr = { '' };
roi(5).atlaslist    = { '' };
roi(5).atlasind     = 1;

com = '';
if nargin < 1
    if nargout > 0
        EEG = roi;
    else
        help pop_roi_activity;
    end
    return
end

% special callback for custom source models
if ~isstruct(EEG)
    fig = EEG;
    userdat = get(fig, 'userdata');
    EEG = userdat{5};
    
    if strcmpi(varargin{1}, 'select') % atlas
        usrdat = userdat{3}(get(findobj(gcf, 'tag', 'selection3'), 'value'));
        strAlign = num2str(usrdat.align);
        strAlign = regexprep(strAlign, ' +', '  ');
        set(findobj(gcf, 'tag', 'push3')     , 'enable', usrdat.enable);
        set(findobj(gcf, 'tag', 'strfile3')  , 'string', usrdat.file, 'enable', usrdat.enable);
        set(findobj(gcf, 'tag', 'transform3'), 'string', strAlign, 'enable', 'on'); % usrdat.enable );
        set(findobj(gcf, 'tag', 'atlas')     , 'string', usrdat.atlasliststr, 'value', usrdat.atlasind, 'enable', 'on' );
        if ~isnan(usrdat.scale)
            set(findobj(gcf, 'tag', 'scale')     , 'string',int2str(usrdat.scale), 'enable', 'on' );
        else
            set(findobj(gcf, 'tag', 'scale')     , 'string', '1', 'enable', 'off' );
        end
        userdat{4} = usrdat.scale;
        set(gcf, 'userdata', userdat);
        
    elseif strcmpi(varargin{1}, 'load') % atlas
        [tmpfilename, tmpfilepath] = uigetfile('*', 'Select a text file');
        if tmpfilename(1) ~=0, set(findobj('parent', gcbf, 'tag', 'strfile3'), 'string', fullfile(tmpfilepath,tmpfilename)); end
        
    elseif strcmpi(varargin{1}, 'selectcoreg')
        plot3dmeshalign(EEG.dipfit.hdmfile, get( findobj(fig, 'tag', 'strfile3'), 'string'), str2num(get( findobj(fig, 'tag', 'transform3'), 'string')));
    end
    return
    
end

% use DIPFIT settings?
dipfitOK = false;
if all(cellfun(@(x)isfield(x, 'coordformat'), { EEG.dipfit }))
    dipfitOK = strcmpi(EEG(1).dipfit.coordformat, 'MNI');
end
if dipfitOK
    for iEEG = 2:length(EEG)
        if ~isequal(EEG(iEEG).dipfit.hdmfile, EEG(1).dipfit.hdmfile) || ...
                ~isequal(EEG(iEEG).dipfit.mrifile, EEG(1).dipfit.mrifile) || ...
                ~isequal(EEG(iEEG).dipfit.chanfile, EEG(1).dipfit.chanfile) || ...
                ~isequal(EEG(iEEG).dipfit.coordformat, EEG(1).dipfit.coordformat) || ...
                ~isequal(EEG(iEEG).dipfit.coord_transform, EEG(1).dipfit.coord_transform)
            dipfitOK = false;
        end
    end
end

if nargin < 2
    if ~dipfitOK
        warndlg2( strvcat( ...
            'You need to set DIPFIT to use the MNI head model not set in one or more datasets.', ...
            ' You will not be able to compute the Leadfield matrix unless you correct this.', ...
            '(use menu item Tools > Locate Dipoles with DIPFIT > Head model and settings).'), 'Use DIPFIT Leadfield matrix');
        return
    end

    cb_select = 'pop_leadfield(gcf, ''select'');';
    cb_load   = 'pop_leadfield(gcf, ''load'');';
    cb_selectcoreg = 'pop_leadfield(gcf, ''selectcoreg'');';
    uiToolTip =  ['Volumetric atlases often have million of voxels and ' 10 ...
            'require to be downsampled to a few thousand voxels' 10 ...
            'to be used as source models. This message box does not' 10 ...
            'apply to surface atlases.' ];
    
    rowg = [0.1 0.8 0.8 0.2];
    % uigeom = { 1 1 rowg rowg 1 rowg rowg [0.1 0.6 0.9 0.3] 1 rowg 1 [0.5 1 0.35 0.5] [0.5 1 0.35 0.5] [0.5 1 0.35 0.5] [1] [0.9 1.2 1] };
    uigeom = { 1 1 rowg rowg [0.1 0.8 0.2 0.8] };
    uilist = { { 'style' 'text' 'string' 'Choose source model for Leadfield matrix' 'fontweight' 'bold'} ...
        { 'style' 'popupmenu' 'string' { roi.label }  'tag' 'selection3' 'value' 3 'callback' cb_select } ...
        {} { 'style' 'text' 'string' 'File name'}                       { 'style' 'edit' 'string' 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx' 'tag' 'strfile3'   'userdata', 'sourcemodel' } { 'style' 'pushbutton' 'string' '...'  'userdata', 'sourcemodel' 'tag' 'push3' 'callback' cb_load }  ...
        {} { 'style' 'text' 'string' 'Transformation to MNI (if any)' } { 'style' 'edit' 'string' 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx' 'tag' 'transform3' 'userdata', 'sourcemodel' } { 'style' 'pushbutton' 'string' '...'  'userdata', 'sourcemodel' 'callback' cb_selectcoreg }  ...
        {} { 'style' 'text' 'string' 'Downsampling (AFNI only)' 'Tooltip' uiToolTip 'userdata' 'scale'}  { 'style' 'edit' 'string' '' 'tag' 'scale' 'enable' 'off' 'userdata' 'scale'} { }  ...
        };
%         {} ...
%         { 'style' 'checkbox' 'string' 'Export ROI to EEGLAB (will replace ICA components)' 'tag' 'export' 'value' 0 'enable' 'off'} ...
%         {} { 'style' 'text' 'string' 'Autoregressive model order' } { 'style' 'edit' 'string' '20' 'tag' 'morder' } { } ...
%         {} { 'style' 'text' 'string' 'Bootstrap if any (n)' }       { 'style' 'edit' 'string' '' 'tag' 'naccu' }    { } ...
%         { 'style' 'checkbox' 'string' 'Compute TRGC' 'tag' 'trgc' 'value' 1 } ...
%         { 'style' 'checkbox' 'string' 'Compute cross-spectrum' 'tag' 'crossspec' 'value' 1 } ...
    [result,usrdat,~,out] = inputgui('geometry', uigeom, 'uilist', uilist, 'helpcom', 'pophelp(''pop_roi_activity'')', ...
        'title', 'Compute ROI activity', 'userdata', {[] [] roi [] EEG}, 'eval', [cb_select 'set(findobj(gcf, ''tag'', ''down''), ''string'', '''');' ]);
    if isempty(result), return, end
    if isempty(usrdat{3}), usrdat{3} = 1; end
                     
    options = {
        'sourcemodel' out.strfile3 ...
        'sourcemodel2mni' str2num(out.transform3) ...
        'downsample' str2num(out.scale) ...
        };
else
    options = varargin;
end

% process multiple datasets
% -------------------------
if length(EEG) > 1
    % check that the dipfit settings are the same
    if nargin < 2
        [ EEG, com ] = eeg_eval( 'pop_roi_activity', EEG, 'warning', 'on', 'params', options );
    else
        [ EEG, com ] = eeg_eval( 'pop_roi_activity', EEG, 'params', options );
    end
    return;
end

%    'export2icamatrix' 'string' {'on', 'off'}   'off';
g = finputcheck(options, { ...
    'downsample'      'integer'  { }             4; % volume only
    'sourcemodel'     'string'  { }             '';
    'sourcemodel2mni' 'real'    { }             [] }, 'pop_roi_activity');
if ischar(g), error(g); end

% Source model
headmodel = load('-mat', EEG.dipfit.hdmfile);
EEG.dipfit.coord_transform = EEG.dipfit.coord_transform;
dataPre = eeglab2fieldtrip(EEG, 'preprocessing', 'dipfit'); % does the transformation
ftPath = fileparts(which('ft_defaults'));

% Prepare the liedfield matrix
[~,~,ext] = fileparts(g.sourcemodel);
if strcmpi(ext, '.nii')
    atlas = ft_read_atlas(g.sourcemodel);
    mri = sum(atlas.tissue(:,:,:,:),4) > 0;
    [r,c,v] = ind2sub(size(mri),find(mri));
    xyz = [r c v ones(length(r),1)];
    xyz = atlas.transform*xyz';
    if nargin > 1 && ~isempty(transform)
        xyz = traditionaldipfit(transform)*xyz;
    end
    disp('DOWNSAMPLING NOT IMPLEMENTED FOR THIS TYPE OF ATLAS');
elseif strcmpi(ext, '.head')
    [~, sourcemodelOri.pos, ~ ] = load_afni_atlas(g.sourcemodel, EEG.dipfit.hdmfile, g.sourcemodel2mni, g.downsample);
elseif strcmpi(ext, '.mat') % && isfield(g.sourcemodel, 'tri')
    sourcemodelOri = transform_move_inward(g.sourcemodel, EEG.dipfit.hdmfile, g.sourcemodel2mni);
end

cfg      = [];
cfg.elec = dataPre.elec;
%     cfg.grid    = sourcemodelOri;   % source points
if isfield(headmodel, 'vol')
    cfg.headmodel = headmodel.vol;   % volume conduction model
else
    cfg.headmodel = headmodel;   % volume conduction model
end
cfg.sourcemodel.inside = ones(size(sourcemodelOri.pos,1),1) > 0;
cfg.sourcemodel.pos    = sourcemodelOri.pos;
if isfield(sourcemodelOri, 'tri')
    cfg.sourcemodel.tri    = sourcemodelOri.tri;
end
cfg.singleshell.batchsize = 5000; % speeds up the computation
EEG.dipfit.sourcemodel = ft_prepare_leadfield(cfg);
EEG.dipfit.sourcemodel.file = g.sourcemodel;
EEG.dipfit.sourcemodel.coordtransform = g.sourcemodel2mni;

% remove vertices not modeled (no longer necessary - makes holes in model)
%     indRm = find(sourcemodel.inside == 0);
%     rowRm = [];
%     for ind = 1:length(indRm)
%         sourcemodel.tri(sourcemodel.tri(:,1) == indRm(ind),:) = [];
%         sourcemodel.tri(sourcemodel.tri(:,2) == indRm(ind),:) = [];
%         sourcemodel.tri(sourcemodel.tri(:,3) == indRm(ind),:) = [];
%         sourcemodel.tri(sourcemodel.tri(:) > indRm(ind)) = sourcemodel.tri(sourcemodel.tri(:) > indRm(ind)) - 1;
%     end
%     sourcemodel.pos(indRm,:) = [];
%     sourcemodel.leadfield(indRm) = [];

if nargout > 1
    com = sprintf( 'EEG = pop_leadfield(EEG, %s);', vararg2str( options ));
end

% -----------------------------------
% surface only - move vertices inward
% -----------------------------------
function [sourcemodelout, transform] = transform_move_inward(sourcemodel, headmodel, transform)

if ischar(headmodel)
    headmodel = load('-mat', headmodel);
    if isfield(headmodel, 'vol')
        headmodel = headmodel.vol;
        headmodel.unit = 'mm';
    end
end
if ischar(sourcemodel)
    try
        sourcemodel = load('-mat', sourcemodel);
    catch
        error('WARNING: could not open source model file')
    end
    if isfield(sourcemodel, 'cortex')
        sourcemodel = sourcemodel.cortex;
    end
else
    % Likely a volume atlas
    sourcemodelout = sourcemodel;
    return
end
if isfield(sourcemodel, 'inside')
    pos = sourcemodel.transform * [sourcemodel.pos(logical(sourcemodel.inside),:) ones(sum(sourcemodel.inside),1) ]';
    sourcemodel = [];
    sourcemodel.pos = pos(1:3,:)';
end
    
newsourcemodel = [];
if isfield(sourcemodel, 'Vertices') && isfield(sourcemodel, 'Faces')
    newsourcemodel.pos = sourcemodel.Vertices;
    newsourcemodel.tri = sourcemodel.Faces;
elseif isfield(sourcemodel, 'Vertices')
    newsourcemodel.pos = sourcemodel.Vertices;
    newsourcemodel.tri = [];
elseif isfield(sourcemodel, 'vertices')
    newsourcemodel.pos = sourcemodel.vertices;
    newsourcemodel.tri = sourcemodel.faces;
else
    newsourcemodel.pos = sourcemodel.pos;
    if isfield(newsourcemodel, 'tri')
        newsourcemodel.tri = sourcemodel.tri;
    else
        newsourcemodel.tri = [];
    end
end

cfg = [];
pos = [newsourcemodel.pos ones(size(newsourcemodel.pos,1),1) ];
if ~isempty(transform)
    pos = traditionaldipfit(transform)*pos';
else
    pos = pos';
end
pos(4,:) = [];
cfg.sourcemodel.pos = pos';
cfg.sourcemodel.tri = newsourcemodel.tri;
cfg.sourcemodel.unit = headmodel.unit;
cfg.moveinward = 1;
cfg.headmodel = headmodel;
disp('moving source model inward if necessary');
sourcemodelout = ft_prepare_sourcemodel(cfg);
transform = [];

% plot3dmeshalign(headmodel);
% 
% hold on;
% plot3dmeshalign(tmp2, [], [1 0 0])

% -------------------
% get list of atlases
% -------------------
function [atlasstr, atlas] = getatlaslist(fileName)

data = load('-mat', fileName);

atlasstr = {};
atlas    = {};
for iAtlas = 1:length(data.Atlas)
    if ~isempty(data.Atlas(iAtlas).Scouts) && ~strcmpi(data.Atlas(iAtlas).Name, 'Structures')
        atlasstr{end+1} = [data.Atlas(iAtlas).Name ' (' int2str(length(data.Atlas(iAtlas).Scouts)) ' ROIs)' ];
        atlas{   end+1} = data.Atlas(iAtlas).Name;
    end
end

