% POP_VIEWPROPS See  common properties of many EEG channel or component
%   Creates a figure containing a scalp topography or channel location for
%   each selected component or channel. Pressing the button above the scalp
%   topopgraphies will open pop_prop_extended for that component or
%   channel. If pop_viewprops is called with only the first two arguments, 
%   a GUI opens to select the rest. If only one argument is given, typecomp
%   will be set to channels (1) and the GUI will open.
%
%   Inputs
%       EEG: EEGLAB EEG structure
%       typecomp: 0 for component, 1 for channel
%       chanorcomp:  channel or component index to plot
%       spec_opt:  cell array of options which are passed to spectopo()
%       erp_opt:  cell array of options which are passed to erpimage()
%       scroll_event:  0 to hide events in scroll plot, 1 to show them
%       classifier_name:  string indicating which component classifier to
%           use (must match a field name in EEG.etc.ic_classification)
%       fig: figure handle for the figure to use.
%
%   See also: pop_prop_extended()
%
%   Adapted from pop_selectcomps Luca Pion-Tonachini (2017)

% Copyright (C) 2001 Arnaud Delorme, Salk Institute, arno@salk.edu
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

% 01-25-02 reformated help & license -ad 

function [com] = pop_viewprops( EEG, typecomp, chanorcomp, spec_opt, erp_opt, scroll_event, classifier_name, fig)

COLACC = [0.75 1 0.75];
PLOTPERFIG = 35;
com = '';

if nargin < 1
	help pop_viewprops;
	return;
end;

if nargin < 2
	typecomp = 1; % default
end;	

if nargin < 3
	promptstr    = { fastif(typecomp,'Channel indices to plot:','Component indices to plot:') ...
                     'Spectral options (see spectopo() help):','Erpimage options (see erpimage() help):' ...
                     [' Draw events over scrolling ' fastif(typecomp,'channel','component') ' activity']};
    if typecomp
        inistr       = { ['1:' int2str(length(EEG.chanlocs))] ['''freqrange'', [2 ' num2str(min(80, EEG.srate/2)) ']'] '' 1};
    else
        inistr       = { ['1:' int2str(size(EEG.icawinv, 2))] ['''freqrange'', [2 ' num2str(min(80, EEG.srate/2)) ']'] '' 1};
    end
    stylestr     = {'edit', 'edit', 'edit', 'checkbox'};
    
    % labels when available
    if ~typecomp && isfield(EEG.etc, 'ic_classification')
        classifiers = fieldnames(EEG.etc.ic_classification);
        if ~isempty(classifiers)
            iclabel_ind = find(strcmpi(classifiers, 'ICLabel'));
            promptstr = [promptstr {classifiers}];
            inistr = [inistr {fastif(isempty(iclabel_ind), 1, iclabel_ind)}];
            stylestr = [stylestr {'popupmenu'}];
        end
    end
    
    try
        result       = inputdlg3( 'prompt', promptstr,'style', stylestr, ...
            'default',  inistr, 'title', 'View many chan or comp. properties -- pop_viewprops');
    catch
        result = [];
    end
	if size( result, 1 ) == 0
        return; end
   
	chanorcomp   = eval( [ '[' result{1} ']' ] );
    spec_opt     = eval( [ '{' result{2} '}' ] );
    erp_opt     = eval( [ '{' result{3} '}' ] );
    scroll_event     = result{4};
    if ~typecomp && isfield(EEG.etc, 'ic_classification') && ~isempty(classifiers)
        classifiers = fieldnames(EEG.etc.ic_classification);
        classifier_name = classifiers{result{5}};
    end

    if length(chanorcomp) > PLOTPERFIG
        ButtonName=questdlg2(strvcat(['More than ' int2str(PLOTPERFIG) fastif(typecomp,' channels',' components') ' so'],...
            'this function will pop-up several windows'), 'Confirmation', 'Cancel', 'OK','OK');
        if  ~isempty( strmatch(lower(ButtonName), 'cancel')), return; end;
    end;

end;
if ~exist('spec_opt', 'var') || ~iscell(spec_opt)
    spec_opt = {}; end
if ~exist('erp_opt', 'var') || ~iscell(erp_opt)
    erp_opt = {}; end
if ~exist('scroll_event', 'var')
    scroll_event = 1; end
if ~exist('classifier_name', 'var')
            classifier_name = ''; end
fprintf('Drawing figure...\n');
currentfigtag = ['selcomp' num2str(rand)]; % generate a random figure tag

if length(chanorcomp) > PLOTPERFIG
    for index = 1:PLOTPERFIG:length(chanorcomp)
        pop_viewprops(EEG, typecomp, chanorcomp(index:min(length(chanorcomp),index+PLOTPERFIG-1)), ...
            spec_opt, erp_opt, scroll_event, classifier_name);
    end;
    com = sprintf('pop_viewprops( %s, %d, %s, %s, %s, %d, ''%s'' )', ...
        inputname(1), typecomp, hlp_tostring(chanorcomp), hlp_tostring(spec_opt), ...
        hlp_tostring(erp_opt), scroll_event, classifier_name);
    return;
end;

try
    icadefs; 
catch
	BACKCOLOR = [0.8 0.8 0.8];
end;

% set up the figure
% -----------------
column =ceil(sqrt( length(chanorcomp) ))+1;
rows = ceil(length(chanorcomp)/column);
if ~exist('fig','var')
	figure('name', [ 'View ' fastif(typecomp,'channels','components') ' properties - pop_viewprops() (dataset: ' EEG.setname ')'], 'tag', currentfigtag, ...
		   'numbertitle', 'off', 'color', BACKCOLOR);
	set(gcf,'MenuBar', 'none');
	pos = get(gcf,'Position');
    if ~typecomp && isfield(EEG.etc, 'ic_classification')
    	set(gcf,'Position', [pos(1) 20 800/7*column 600/5*rows*1.2]);
    else
    	set(gcf,'Position', [pos(1) 20 800/7*column 600/5*rows]);
    end
    incx = 120;
    incy = 110;
    sizewx = 100/column;
    if rows > 2
        sizewy = 90/rows;
	else 
        sizewy = 80/rows;
    end;
    pos = get(gca,'position'); % plot relative to current axes
	q = [pos(1) pos(2) 0 0];
	s = [pos(3) pos(4) pos(3) pos(4)]./100;
	axis off;
end;

% figure rows and columns
% -----------------------  
if ~typecomp && EEG.nbchan > 64
    disp('More than 64 electrodes: electrode locations not shown');
    plotelec = 0;
else
    plotelec = 1;
end;
count = 1;
for ri = chanorcomp
	if exist('fig','var')
		button = findobj('parent', fig, 'tag', ['comp' num2str(ri)]);
		if isempty(button) 
			error( 'pop_viewprops(): figure does not contain the component button');
		end;	
	else
		button = [];
	end;		
		 
	if isempty( button )
		% compute coordinates
		% -------------------
		X = mod(count-1, column)/column * incx-10;  
        Y = (rows-floor((count-1)/column))/rows * incy - sizewy*1.3;  

		% plot the head
		% -------------
        if ~strcmp(get(gcf, 'tag'), currentfigtag);
            figure(findobj('tag', currentfigtag));
        end;
		ha = axes('Units','Normalized', 'Position',[X Y sizewx sizewy].*s+q);
        if typecomp
            topoplot( ri, EEG.chanlocs, 'chaninfo', EEG.chaninfo, ...
                     'electrodes','off', 'style', 'blank', 'emarkersize1chan', 12);
        else
            if plotelec
                topoplot( EEG.icawinv(:,ri), EEG.chanlocs, 'verbose', ...
                          'off', 'style' , 'fill', 'chaninfo', EEG.chaninfo, 'numcontour', 8);
            else
                topoplot( EEG.icawinv(:,ri), EEG.chanlocs, 'verbose', ...
                          'off', 'style' , 'fill','electrodes','off', 'chaninfo', EEG.chaninfo, 'numcontour', 8);
            end;
            % labels
            if ~typecomp && isfield(EEG.etc, 'ic_classification')
                classifiers = fieldnames(EEG.etc.ic_classification);
                if ~isempty(classifiers)
                    if ~exist('classifier_name', 'var') || isempty(classifier_name)
                        if any(strcmpi(classifiers, 'ICLabel'));
                            classifier_name = 'ICLabel';
                        else
                            classifier_name = classifiers{1};
                        end
                    else
                        classifier_name = classifiers{strcmpi(classifiers, classifier_name)};
                    end
                    if ri == chanorcomp(1) && size(EEG.icawinv, 2) ...
                            ~= size(EEG.etc.ic_classification.(classifier_name).classifications, 1)
                        warning(['The number of ICs do not match the number of IC classifications. This will result in incorrectly plotted labels. Please rerun ' classifier_name])
                    end
                    [prob, classind] = max(EEG.etc.ic_classification.(classifier_name).classifications(ri, :));
                    t = title(sprintf('%s : %.1f%%', ...
                        EEG.etc.ic_classification.(classifier_name).classes{classind}, ...
                        prob*100));
                    set(t, 'Position', get(t, 'Position') .* [1 -1.2 1])
                end
            end
        end
		axis square;

		% plot the button
		% ---------------
         if ~strcmp(get(gcf, 'tag'), currentfigtag);
             figure(findobj('tag', currentfigtag));
         end
		button = uicontrol(gcf, 'Style', 'pushbutton', 'Units','Normalized', 'Position',...
                           [X Y+sizewy sizewx sizewy*0.18].*s+q, 'tag', ['comp' num2str(ri)]);
		set( button, 'callback', {@pop_prop_extended, EEG, typecomp, ri, NaN, spec_opt, erp_opt, scroll_event, classifier_name} );
	end;
    if typecomp
        set( button, 'backgroundcolor', COLACC, 'string', EEG.chanlocs(ri).labels); 	
    else
        set( button, 'backgroundcolor', COLACC, 'string', int2str(ri)); 	
    end
	drawnow;
	count = count +1;
end;

com = sprintf('pop_viewprops( %s, %d, %s, %s, %s, %d, ''%s'' )', ...
    inputname(1), typecomp, hlp_tostring(chanorcomp), hlp_tostring(spec_opt), ...
    hlp_tostring(erp_opt), scroll_event, classifier_name);
end


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

function [result, userdat, strhalt, resstruct] = inputdlg3( varargin);

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

[tmpresult, userdat, strhalt, resstruct] = inputgui('uilist', uilist,'geometry', uigeometry, addopts{:});
result = cell(1,length(opt.prompt));
result(find(outputind)) = tmpresult;
end


% the following functions are from BCILAB
function str = hlp_tostring(v,stringcutoff,prec)
% Get an human-readable string representation of a data structure.
% String = hlp_tostring(Data,StringCutoff)
%
% The resulting string representations are usually executable, but there are corner cases (e.g.,
% certain anonymous function handles and large data sets), which are not supported. For
% general-purpose serialization, see hlp_serialize/hlp_deserialize.
%
% In:
%   Data : a data structure
%
%   StringCutoff : optional maximum string length for atomic fields (default: 0=off)
%
%   Precision : maximum significant digits (default: 15)
%
% Out:
%   String : string form of the data structure
%
% Notes:
%   hlp_tostring has builtin support for displaying expression data structures.
%
% Examples:
%   % get a string representation of a data structure
%   hlp_tostring({'test',[1 2 3], struct('field','value')})
%
% See also:
%   hlp_serialize
%
%                                Christian Kothe, Swartz Center for Computational Neuroscience, UCSD
%                                2010-04-15
%
%                                adapted from serialize.m
%                                (C) 2006 Joger Hansegord (jogerh@ifi.uio.no)

% Copyright (C) Christian Kothe, SCCN, 2010, christian@sccn.ucsd.edu
%
% This program is free software; you can redistribute it and/or modify it under the terms of the GNU
% General Public License as published by the Free Software Foundation; either version 2 of the
% License, or (at your option) any later version.
%
% This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
% even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
% General Public License for more details.
%
% You should have received a copy of the GNU General Public License along with this program; if not,
% write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307
% USA

if nargin < 2
    stringcutoff = 0; end
if nargin < 3
    prec = 15; end

str = serializevalue(v);


    function val = serializevalue(v)
        % Main hub for serializing values
        if isnumeric(v) || islogical(v)
            val = serializematrix(v);
        elseif ischar(v)
            val = serializestring(v);
        elseif isa(v,'function_handle')
            val = serializefunction(v);
        elseif is_impure_expression(v)
            val = serializevalue(v.tracking.expression);
        elseif has_canonical_representation(v)
            val = serializeexpression(v);
        elseif is_dataset(v)
            val = serializedataset(v);
        elseif isstruct(v)
            val = serializestruct(v);
        elseif iscell(v)
            val = serializecell(v);
        elseif isobject(v)
            val = serializeobject(v);
        else
            try
                val = serializeobject(v);
            catch
                error('Unhandled type %s', class(v));
            end
        end
    end

    function val = serializestring(v)
        % Serialize a string
        if any(v == '''')
            val = ['''' strrep(v,'''','''''') ''''];
            try
                if ~isequal(eval(val),v)
                    val = ['char(' serializevalue(uint8(v)) ')']; end
            catch
                val = ['char(' serializevalue(uint8(v)) ')'];
            end
        else
            val = ['''' v ''''];
        end
        val = trim_value(val,'''');
    end

    function val = serializematrix(v)
        % Serialize a matrix and apply correct class and reshape if required
        if ndims(v) < 3 %#ok<ISMAT>
            if isa(v, 'double')
                if size(v,1) == 1 && length(v) > 3 && isequal(v,v(1):v(2)-v(1):v(end))
                    % special case: colon sequence
                    if v(2)-v(1) == 1
                        val = ['[' num2str(v(1)) ':' num2str(v(end)) ']'];
                    else
                        val = ['[' num2str(v(1)) ':' num2str(v(2)-v(1)) ':' num2str(v(end)) ']'];
                    end
                elseif size(v,2) == 1 && length(v) > 3 && isequal(v',v(1):v(2)-v(1):v(end))
                    % special case: colon sequence
                    if v(2)-v(1) == 1
                        val = ['[' num2str(v(1)) ':' num2str(v(end)) ']'''];
                    else
                        val = ['[' num2str(v(1)) ':' num2str(v(2)-v(1)) ':' num2str(v(end)) ']'''];
                    end
                else
                    val = mat2str(v,prec);
                end
            else
                val = mat2str(v,prec,'class');
            end
            val = trim_value(val,']');
        else
            if isa(v, 'double')
                val = mat2str(v(:),prec);
            else
                val = mat2str(v(:),prec,'class');
            end
            val = trim_value(val,']');
            val = sprintf('reshape(%s, %s)', val, mat2str(size(v)));
        end
    end

    function val = serializecell(v)
        % Serialize a cell
        if isempty(v)
            val = '{}';
            return
        end
        cellSep = ', ';
        if isvector(v) && size(v,1) > 1
            cellSep = '; ';
        end
        
        % Serialize each value in the cell array, and pad the string with a cell
        % separator.
        vstr = cellfun(@(val) [serializevalue(val) cellSep], v, 'UniformOutput', false);
        vstr{end} = vstr{end}(1:end-2);
        
        % Concatenate the elements and add a reshape if requied
        val = [ '{' vstr{:} '}'];
        if ~isvector(v)
            val = ['reshape('  val sprintf(', %s)', mat2str(size(v)))];
        end
    end

    function val = serializeexpression(v)
        % Serialize an expression
        if numel(v) > 1
            val = ['['];
            for k = 1:numel(v)
                val = [val serializevalue(v(k)), ', ']; end
            val = [val(1:end-2) ']'];
        else
            if numel(v.parts) > 0
                val = [char(v.head) '('];
                for fieldNo = 1:numel(v.parts)
                    val = [val serializevalue(v.parts{fieldNo}), ', ']; end
                val = [val(1:end-2) ')'];
            else
                val = char(v.head);
            end
        end
    end

    function val = serializedataset(v) %#ok<INUSD>
        % Serialize a data set
        val = '<EEGLAB data set>';
    end

    function val = serializestruct(v)
        % Serialize a struct by converting the field values using struct2cell
        fieldNames   = fieldnames(v);
        fieldValues  = struct2cell(v);
        if ndims(fieldValues) > 6
            error('Structures with more than six dimensions are not supported');
        end
        val = 'struct(';
        for fieldNo = 1:numel(fieldNames)
            val = [val serializevalue( fieldNames{fieldNo}) ', '];
            val = [val serializevalue( permute(fieldValues(fieldNo, :,:,:,:,:,:), [2:ndims(fieldValues) 1]) ) ];
            val = [val ', '];
        end
        if numel(fieldNames)==0
            val = [val ')'];
        else
            val = [val(1:end-2) ')'];
        end
        if ~isvector(v)
            val = sprintf('reshape(%s, %s)', val, mat2str(size(v)));
        end
    end

    function val = serializeobject(v)
        % Serialize an object by converting to struct and add a call to the copy constructor
        val = sprintf('%s(%s)', class(v), serializevalue(struct(v)));
    end


    function val = serializefunction(v)
        % Serialize a function handle
        try
            val = ['@' char(get_function_symbol(v))];
        catch
            val = char(v);
        end
    end

    function v = trim_value(v,appendix)
        if nargin < 2
            appendix = ''; end
        % Trim a serialized value to a certain length
        if stringcutoff && length(v) > stringcutoff
            v = [v(1:stringcutoff) '...' appendix]; end
    end
end

function result___ = get_function_symbol(expression___)
% internal: some function_handle expressions have a function symbol (an @name expression), and this function obtains it
% note: we are using funny names here to bypass potential name conflicts within the eval() clause further below
if ~isa(expression___,'function_handle')
    error('the expression has no associated function symbol.'); end

string___ = char(expression___);
if string___(1) == '@'
    % we are dealing with a lambda function
    if is_symbolic_lambda(expression___)
        result___ = eval(string___(27:end-21));
    else
        error('cannot derive a function symbol from a non-symbolic lambda function.');
    end
else
    % we are dealing with a regular function handle
    result___ = expression___;
end
end

function res = is_symbolic_lambda(x)
% internal: a symbolic lambda function is one which generates expressions when invoked with arguments (this is what exp_symbol generates)
res = isa(x,'function_handle') && ~isempty(regexp(char(x),'@\(varargin\)struct\(''head'',\{.*\},''parts'',\{varargin\}\)','once'));
end

function res = is_impure_expression(x)
% an impure expression is a MATLAB structure with a .tracking.expression field
res = isfield(x,'tracking') && isfield(x.tracking,'expression');
end

function res = is_dataset(x)
% Determine whether some object is a data set.
res = all(isfield(x,{'data','srate'}));
end

function res = has_canonical_representation(x)
% determine whether an expression is represented as a struct with the fields 'head' and 'parts'.
res = all(isfield(x,{'head','parts'}));
end