% eegplugin_dipfit() - DIPFIT is the dipole fitting Matlab Toolbox of 
%                      Robert Oostenveld and Arnaud Delorme
%
% Usage:
%   >> eegplugin_dipfit(fig, trystrs, catchstrs);
%
% Inputs:
%   fig        - [integer] eeglab figure.
%   trystrs    - [struct] "try" strings for menu callbacks.
%   catchstrs  - [struct] "catch" strings for menu callbacks.
%
% Notes:
%   To create a new plugin, simply create a file beginning with "eegplugin_"
%   and place it in your eeglab folder. It will then be automatically 
%   detected by eeglab. See also this source code internal comments.
%   For eeglab to return errors and add the function's results to 
%   the eeglab history, menu callback must be nested into "try" and 
%   a "catch" strings. For more information on how to create eeglab 
%   plugins, see http://www.sccn.ucsd.edu/eeglab/contrib.html
%
% Author: Arnaud Delorme, CNL / Salk Institute, 22 February 2003
%
% See also: eeglab()

% Copyright (C) 2003 Arnaud Delorme, Salk Institute, arno@salk.edu
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
% Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1.07  USA

function vers = eegplugin_dipfit(fig, trystrs, catchstrs)
    
    vers = 'dipfit4.2';
    if nargin < 3
        error('eegplugin_dipfit requires 3 arguments');
    end
    
    % find tools menu
    % ---------------
    menu = findobj(fig, 'tag', 'tools'); 
    % tag can be 
    % 'import data'  -> File > import data menu
    % 'import epoch' -> File > import epoch menu
    % 'import event' -> File > import event menu
    % 'export'       -> File > export
    % 'tools'        -> tools menu
    % 'plot'         -> plot menu

    % command to check that the '.source' is present in the EEG structure 
    % -------------------------------------------------------------------
    check_dipfit = [trystrs.no_check 'if ~isfield(EEG(1), ''dipfit''), error(''Run the dipole setting first''); end;'  ...
                    'if isempty(EEG(1).dipfit), error(''Run the dipole setting first''); end;'  ];
    check_dipfitnocheck = [ trystrs.no_check 'if ~isfield(EEG, ''dipfit''), error(''Run the dipole setting first''); end; ' ];
    check_chans = [ '[EEG,tmpres] = eeg_checkset(EEG, ''chanlocs_homogeneous'');' ...
                       'if ~isempty(tmpres), eegh(tmpres), end; clear tmpres;' ];
    
    % menu callback commands
    % ----------------------
    comsetting = [ trystrs.check_data check_chans '[EEG LASTCOM]=pop_dipfit_settings(EEG);'    catchstrs.store_and_hist ]; 
    combatch   = [ check_dipfit check_chans  '[EEG LASTCOM] = pop_dipfit_gridsearch(EEG);'    catchstrs.store_and_hist ];
    comfit     = [ check_dipfitnocheck check_chans [ 'EEG = pop_dipfit_nonlinear(EEG); ' ...
                        'LASTCOM = ''% === History not supported for manual dipole fitting ==='';' ]  catchstrs.store_and_hist ];
    comauto    = [ check_dipfit check_chans  '[EEG LASTCOM] = pop_multifit(EEG);'        catchstrs.store_and_hist ];
    % preserve the '=" sign in the comment above: it is used by EEGLAB to detect appropriate LASTCOM
    complot    = [ check_dipfit check_chans 'LASTCOM = pop_dipplot(EEG);'               catchstrs.add_to_hist ];
    comleadfield  = [ check_dipfit check_chans '[EEG, LASTCOM] = pop_leadfield(EEG);'   catchstrs.store_and_hist ];
    comloreta  = [ check_dipfit check_chans 'LASTCOM = pop_dipfit_loreta(EEG);'         catchstrs.add_to_hist ];
    
    % create menus
    % ------------
    submenu = uimenu( menu, 'Label', 'Source localization using DIPFIT', 'separator', 'on', 'tag', 'dipfit', 'userdata', 'startup:off;study:on');
    lightMenuFlag = isempty(findobj(fig, 'Label', 'Reject data epochs'));
    if ~isdeployed && lightMenuFlag, try set(submenu, 'position', 14); catch, end; end
    uimenu( submenu, 'Label', 'Head model and settings'  , 'CallBack', comsetting, 'userdata', 'startup:off;study:on');
    uimenu( submenu, 'Label', 'Component dipole coarse fit', 'CallBack', combatch, 'userdata', 'startup:off', 'separator', 'on');
    uimenu( submenu, 'Label', 'Component dipole fine fit'  , 'CallBack', comfit, 'userdata', 'startup:off');
    uimenu( submenu, 'Label', 'Component dipole plot '     , 'CallBack', complot, 'userdata', 'startup:off');
    uimenu( submenu, 'Label', 'Component dipole autofit'   , 'CallBack', comauto, 'userdata', 'startup:off;study:on');
    uimenu( submenu, 'Label', 'Distributed source Leadfield matrix', 'CallBack', comleadfield, 'userdata', 'startup:off', 'separator', 'on');
    uimenu( submenu, 'Label', 'Distributed source component modelling', 'CallBack', comloreta, 'userdata', 'startup:off');
