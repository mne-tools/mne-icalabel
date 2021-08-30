% clean_rawdata(): a wrapper for EEGLAB to call Christian's clean_artifacts.
%
% Usage:
%   >>  EEG = clean_rawdata(EEG, arg_flatline, arg_highpass, arg_channel, arg_noisy, arg_burst, arg_window)
%
% ------------------ below is from clean_artifacts -----------------------
%
% This function removes flatline channels, low-frequency drifts, noisy channels, short-time bursts
% and incompletely repaird segments from the data. Tip: Any of the core parameters can also be
% passed in as [] to use the respective default of the underlying functions, or as 'off' to disable
% it entirely.
%
% Hopefully parameter tuning should be the exception when using this function -- however, there are
% 3 parameters governing how aggressively bad channels, bursts, and irrecoverable time windows are
% being removed, plus several detail parameters that only need tuning under special circumstances.
%
%   FlatlineCriterion: Maximum tolerated flatline duration. In seconds. If a channel has a longer
%                      flatline than this, it will be considered abnormal. Default: 5
%
%   Highpass :         Transition band for the initial high-pass filter in Hz. This is formatted as
%                      [transition-start, transition-end]. Default: [0.25 0.75].
%
%   ChannelCriterion : Minimum channel correlation. If a channel is correlated at less than this
%                      value to a reconstruction of it based on other channels, it is considered
%                      abnormal in the given time window. This method requires that channel
%                      locations are available and roughly correct; otherwise a fallback criterion
%                      will be used. (default: 0.85)
%
%   LineNoiseCriterion : If a channel has more line noise relative to its signal than this value, in
%                        standard deviations based on the total channel population, it is considered
%                        abnormal. (default: 4)
%
%   BurstCriterion : Standard deviation cutoff for removal of bursts (via ASR). Data portions whose
%                    variance is larger than this threshold relative to the calibration data are
%                    considered missing data and will be removed. The most aggressive value that can
%                    be used without losing much EEG is 3. For new users it is recommended to at
%                    first visually inspect the difference between the original and cleaned data to
%                    get a sense of the removed content at various levels. A quite conservative
%                    value is 5. Default: 5.
%
%
%   WindowCriterion :  Criterion for removing time windows that were not repaired completely. This may
%                      happen if the artifact in a window was composed of too many simultaneous
%                      uncorrelated sources (for example, extreme movements such as jumps). This is
%                      the maximum fraction of contaminated channels that are tolerated in the final
%                      output data for each considered window. Generally a lower value makes the
%                      criterion more aggressive. Default: 0.25. Reasonable range: 0.05 (very
%                      aggressive) to 0.3 (very lax).
%
% see also: clean_artifacts

% Author: Makoto Miyakoshi and Christian Kothe, SCCN,INC,UCSD
% History:
% 05/13/2014 ver 1.2 by Christian. Added better channel removal function (uses locations if available).
% 07/16/2013 ver 1.1 by Makoto and Christian. Minor update for help and default values.
% 06/26/2013 ver 1.0 by Makoto. Created.

% Copyright (C) 2013, Makoto Miyakoshi and Christian Kothe, SCCN,INC,UCSD
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

function cleanEEG = clean_rawdata(EEG, arg_flatline, arg_highpass, arg_channel, arg_noisy, arg_burst, arg_window)

disp('The function clean_rawdata has been deprecated and is only kept for backward');
disp('compatibility. Use the clean_artifacts function instead.');

if arg_flatline == -1; arg_flatline = 'off'; disp('flatchan rej disabled.'  ); end
if arg_highpass == -1; arg_highpass = 'off'; disp('highpass disabled.'      ); end
if arg_channel  == -1; arg_channel  = 'off'; disp('badchan rej disabled.'   ); end
if arg_noisy    == -1; arg_noisy    = 'off'; disp('noise-based rej disabled.'); end
if arg_burst    == -1; arg_burst    = 'off'; disp('burst clean disabled.'   ); end
if arg_window   == -1; arg_window   = 'off'; disp('bad window rej disabled.'); end

cleanEEG = clean_artifacts(EEG, 'FlatlineCriterion', arg_flatline,...
                                'Highpass',          arg_highpass,...
                                'ChannelCriterion',  arg_channel,...
                                'LineNoiseCriterion',  arg_noisy,...
                                'BurstCriterion',    arg_burst,...
                                'WindowCriterion',   arg_window);
