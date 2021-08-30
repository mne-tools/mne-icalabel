function [EEG, varargout] = pop_iclabel(EEG, icversion)
%POP_ICLABEL Function for EEG IC labeling
%   Label independent components using ICLabel. Go to
%   https://sccn.ucsd.edu/wiki/ICLabel for a tutorial on this plug-in. Go
%   to labeling.ucsd.edu/tutorial/about for more information. This is a
%   beta version and results may change in the near future. For direct
%   usage of ICLabel in scripts and functions, the function "iclabel" is
%   suggested. To report a bug or issue, please create an "Issue" post on 
%   the GitHub page at https://github.com/sccn/ICLabel/issues or send an 
%   email to eeglab@sccn.ucsd.edu.
%
%   Inputs
%       EEG: EEGLAB EEG structure. Must have an attached ICA decomposition.
%       version: Which version of the ICLabel classifier to use. If not 
%           provided, 'Default' is used. The three  ` choices are:
%           'Default' - the full classifier validated in the accompanying
%               publications
%           'Lite' - the lite version of the classifier which excludes 
%               autocorrelation as a feature for performance reasons.
%           'Beta' - included only to maintain the repoducibility of any 
%               studies which may previously have used it.
%
%   Results are stored in EEG.etc.ic_classifications.ICLabel. The matrix of
%   label vectors is stored under "classifications" and the cell array of
%   class names are stored under "classes". The matrix stored under
%   "classifications" is organized with each column matching to the
%   equivalent element in "classes" and each row matching to the equivalent
%   IC. For example, if you want to see what percent ICLabel attributes IC
%   7 to the class "eye", you would look at:
%       EEG.etc.ic_classifications.ICLabel.classifications(7, 3)
%   since EEG.etc.ic_classifications.ICLabel.classes{3} is "eye"

if ~exist('icversion', 'var')
    try
        [~, icversion] = evalc(['inputdlg3(' ...
            '''prompt'', {''Select which icversion of ICLabel to use:'', ''Default (recommended)|Lite|Beta''},' ... 
            '''style'', {''text'', ''popupmenu''},' ...
            '''default'', {[], 1},' ...
            '''tag'', {'''', [''"Default" and "Lite" are validated in the ''' ...
                         '''ICLabel publication. Beta is included ''' ...
                         '''only to maintain the repoducibility of any studies ''' ...
                         '''which may have used it'']},' ...
            '''title'', ''ICLabel'');']);
% The above code is equivalent to the block below except it successfully
% supresses an uneccessary warning from supergui.
%         icversion = inputdlg3( ...
%             'prompt', {'Select which icversion of ICLabel to use:', 'Default (recommended)|Lite|Beta'}, ... 
%             'style', {'text', 'popupmenu'}, ...
%             'default', {[], 1}, ...
%             'tag', {'', ['"Default" and "Lite" are validated in the ' ...
%                          'ICLabel publication. Beta is included ' ...
%                          'only to maintain the repoducibility of any studies ' ...
%                          'which may have used it']}, ...
%             'title', 'ICLabel');
        icversion = icversion{2};
    catch
        icversion = 0;
    end
    
    switch icversion
        case 0
            varargout = { [] };
            return
        case 1
            icversion = 'default';
        case 2
            icversion = 'lite';
        case 3
            icversion = 'beta';
    end

end

if length(EEG) > 1
    [EEG,com] = eeg_eval( 'iclabel', EEG, 'params', { icversion } );   

    disp('*************************************************');
    disp('Scan datasets which have common ICA decomposition');
    disp('and average component classification probabilities');
    disp('*************************************************');
    if ~isempty(com)
        sameICA = std_findsameica(EEG);
        if any(cellfun(@length, sameICA) > 1)
            for iSame = 1:length(sameICA)
                if ~isempty(sameICA{iSame})
                    % average matrix
                    icMatrix = zeros(size(EEG(sameICA{iSame}(1)).etc.ic_classification.ICLabel.classifications), 'single');
                    for iDat = 1:length(sameICA{iSame})
                        icMatrix = icMatrix + EEG(sameICA{iSame}(iDat)).etc.ic_classification.ICLabel.classifications/length(sameICA{iSame});
                    end
                    % copy average matrix to datasets
                    for iDat = 1:length(sameICA{iSame})
                        EEG(sameICA{iSame}(iDat)).etc.ic_classification.ICLabel.classifications = icMatrix;
                    end
                end
            end
            
            % resave datasets if needed
            for iDat = 1:length(EEG)
                EEG(iDat).saved = 'no';
            end
            eeglab_options;
            if option_storedisk
                EEG = pop_saveset(EEG, 'savemode', 'resave');
            end
        end
    end
else
    EEG = iclabel(EEG, icversion);
end
varargout = {['EEG = pop_iclabel(EEG, ''' icversion ''');']};

% % visualize with viewprops
% try
%     pop_viewprops(EEG, 0);
% catch
%     try
%         addpath(fullfile(fileparts(which('iclabel')), 'viewprops'))
%         vp_com = pop_viewprops(EEG, 0);
%     catch
%         disp('ICLabel: Install the viewprops eeglab plugin to see IC label visualizations.')
%     end
% end
%     

% inputdlg3() - A comprehensive gui automatic builder. This function takes
%               text, type of GUI and default value and builds
%               automatically a simple graphic interface.
%
% Usage:
%   >> [outparam outstruct] = inputdlg3( 'key1', 'val1', 'key2', 'val2', ... );
% 
% Inputs:
%   'prompt'     - cell array of text
%   'style'      - cell array of style for each GUI. Default is edit.
%   'default'    - cell array of default values. Default is empty.
%   'tags'       - cell array of tag text. Default is no tags.
%   'tooltip'    - cell array of tooltip texts. Default is no tooltip.
%
% Output:
%   outparam   - list of outputs. The function scans all lines and
%                add up an output for each interactive uicontrol, i.e
%                edit box, radio button, checkbox and listbox.
%   userdat    - 'userdata' value of the figure.
%   strhalt    - the function returns when the 'userdata' field of the
%                button with the tag 'ok' is modified. This returns the
%                new value of this field.
%   outstruct  - returns outputs as a structure (only tagged ui controls
%                are considered). The field name of the structure is
%                the tag of the ui and contain the ui value or string.
%
% Note: the function also adds three buttons at the bottom of each 
%       interactive windows: 'CANCEL', 'HELP' (if callback command
%       is provided) and 'OK'.
%
% Example:
%   res = inputdlg3('prompt', { 'What is your name' 'What is your age' } );
%   res = inputdlg3('prompt', { 'Chose a value below' 'Value1|value2|value3' ...
%                   'uncheck the box' }, ...
%                   'style',  { 'text' 'popupmenu' 'checkbox' }, ...
%                   'default',{ 0 2 1 });
%
% Author: Arnaud Delorme, Tim Mullen, Christian Kothe, SCCN, INC, UCSD
%
% See also: supergui(), eeglab()

% Copyright (C) Arnaud Delorme, SCCN, INC, UCSD, 2010, arno@ucsd.edu
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either icversion 2 of the License, or
% (at your option) any later icversion.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

function [result, userdat, strhalt, resstruct] = inputdlg3( varargin)

if nargin < 2
   help inputdlg3;
   return;
end;	

% check input values
% ------------------
[opt addopts] = finputcheck(varargin, { 'prompt'  'cell'  []   {};
                                        'style'   'cell'  []   {};
                                        'default' 'cell'  []   {};
                                        'tag'     'cell'  []   {};
                                        'tooltip','cell'  []   {}}, 'inputdlg3', 'ignore');
if isempty(opt.prompt),  error('The ''prompt'' parameter must be non empty'); end;
if isempty(opt.style),   opt.style = cell(1,length(opt.prompt)); opt.style(:) = {'edit'}; end;
if isempty(opt.default), opt.default = cell(1,length(opt.prompt)); opt.default(:) = {0}; end;
if isempty(opt.tag),     opt.tag = cell(1,length(opt.prompt)); opt.tag(:) = {''}; end;

% creating GUI list input
% -----------------------
uilist = {};
uigeometry = {};
outputind  = ones(1,length(opt.prompt));
for index = 1:length(opt.prompt)
    if strcmpi(opt.style{index}, 'edit')
        uilist{end+1} = { 'style' 'text' 'string' opt.prompt{index} };
        uilist{end+1} = { 'style' 'edit' 'string' opt.default{index} 'tag' opt.tag{index} 'tooltip' opt.tag{index}};
        uigeometry{index} = [2 1];
    else
        uilist{end+1} = { 'style' opt.style{index} 'string' opt.prompt{index} 'value' opt.default{index} 'tag' opt.tag{index} 'tooltip' opt.tag{index}};
        uigeometry{index} = [1];
    end;
    if strcmpi(opt.style{index}, 'text')
        outputind(index) = 0;
    end;
end;

w = warning('off', 'MATLAB:namelengthmaxexceeded');
[tmpresult, userdat, strhalt, resstruct] = inputgui('uilist', uilist,'geometry', uigeometry, addopts{:});
warning(w.state, 'MATLAB:namelengthmaxexceeded') %  warning suppression added by luca
result = cell(1,length(opt.prompt));
result(find(outputind)) = tmpresult;
