% eegplugin_clean_rawdata() - a wrapper to plug-in Christian's clean_artifact() into EEGLAB. .
% 
% Usage:
%   >> eegplugin_firstPassOutlierTrimmer(fig,try_strings,catch_strings);
%
%   see also: clean_artifacts

% Author: Makoto Miyakoshi and Christian Kothe, SCCN,INC,UCSD 2013
% History:
% 05/13/2014 ver 0.30 by Christian. Added better channel removal function that uses channel locations if present.
% 11/20/2013 ver 0.20 by Christian. Updated signal processing routines to current versions.
% 11/15/2013 ver 0.12 by Christian. Fixed a rare bug in asr_process.
% 07/16/2013 ver 0.11 by Makoto. Minor change on the name on GUI menu.
% 06/26/2013 ver 0.10 by Makoto. Created.

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

function vers = eegplugin_clean_rawdata(fig,try_strings,catch_strings)

vers = '2.4';
cmd = [ try_strings.check_data ...
        '[EEG,LASTCOM] = pop_clean_rawdata(EEG);' ...
        catch_strings.new_and_hist ];

p = fileparts(which('eegplugin_clean_rawdata'));
addpath(fullfile(p, 'manopt'));
addpath(fullfile(p, 'manopt', 'manopt','manifolds', 'grassmann'));
addpath(fullfile(p, 'manopt', 'manopt','tools'));
addpath(fullfile(p, 'manopt', 'manopt','core'));
addpath(fullfile(p, 'manopt', 'manopt','solvers', 'trustregions'));

% create menu
toolsmenu = findobj(fig, 'tag', 'tools');
uimenu( toolsmenu, 'label', 'Reject data using Clean Rawdata and ASR', 'userdata', 'startup:off;epoch:off;study:on', ...
    'callback', cmd, 'position', 7);
