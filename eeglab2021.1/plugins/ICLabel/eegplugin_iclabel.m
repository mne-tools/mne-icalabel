function vers = eegplugin_iclabel( fig, try_strings, catch_strings )
%EEGLABPLUGIN_ICLABEL EEGLAB plugin for EEG IC labeling
%   Label independent components using ICLabel. Go to
%   https://sccn.ucsd.edu/wiki/ICLabel for a tutorial on this plug-in. Go
%   to labeling.ucsd.edu/tutorial/about for more information. To report a
%   bug or issue, please create an "Issue" post on the GitHub page at 
%   https://github.com/sccn/ICLabel/issues or send an email to 
%   eeglab@sccn.ucsd.edu.
%
%   Results are stored in EEG.etc.ic_classifications.ICLabel. The matrix of
%   label vectors is stored under "classifications" and the cell array of
%   class names are stored under "classes". The matrix stored under
%   "classifications" is organized with each column matching to the
%   equivalent element in "classes" and each row matching to the equivalent
%   IC. For example, if you want to see what percent ICLabel attributes IC
%   7 to the class "eye", you would look at:
%       EEG.etc.ic_classifications.ICLabel.classifications(7, 3)
%   since EEG.etc.ic_classifications.ICLabel.classes{3} is "eye".

% version
vers = 'ICLabel1.3';

% input check
if nargin < 3
    error('eegplugin_iclabel requires 3 arguments');
end

% add items to EEGLAB tools menu
plotmenu = findobj(fig, 'tag', 'tools');

iclabelmenu = uimenu( plotmenu, 'label', 'Classify components using ICLabel','userdata', 'startup:off;study:on;roi:off');
lightMenuFlag = isempty(findobj(fig, 'Label', 'Reject data epochs'));
if lightMenuFlag, set(iclabelmenu, 'position', 9); else set(iclabelmenu, 'position', 12); end 

viewpropcom = [        'try, pop_viewprops(EEG, 0); catch,' ...
        'try, vp_path = fullfile(fileparts(which(''iclabel'')), ''viewprops'');' ...
        'addpath(vp_path); if length(EEG) == 1, LASTCOM = pop_viewprops(EEG, 0); end;' ...
        'disp(''ICLabel: Install the viewprops eeglab plugin to create IC visualizations like these elsewhere.''),' ...
        'catch, end; end;'];

if ~isfield(try_strings, 'check_ica_chanlocs')
    try_strings.check_ica_chanlocs = try_strings.check_ica;
end
uimenu( iclabelmenu, 'label', 'Label components', ...
    'callback', [try_strings.check_ica_chanlocs ...
        '[EEG, LASTCOM] = pop_iclabel(EEG);' ...
        catch_strings.store_and_hist ...
        'if ~isempty(LASTCOM),' ...
        viewpropcom ' end' ], 'userdata', 'startup:off;study:on');

uimenu( iclabelmenu, 'label', 'Flag components as artifacts', ...
    'callback', [try_strings.check_ica_chanlocs '[EEG, LASTCOM] = pop_icflag(EEG);' catch_strings.store_and_hist ], 'userdata', 'startup:off;study:on');

uimenu( iclabelmenu, 'label', 'View extended component properties', ...
    'callback', viewpropcom );

% activate matconvnet
activate_matconvnet();
end

