function [EEG,HP,BUR,removed_channels] = clean_artifacts(EEG,varargin)
% All-in-one function for artifact removal, including ASR.
% [EEG,HP,BUR] = clean_artifacts(EEG, Options...)
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
% Notes: 
%  * This function uses the Signal Processing toolbox for pre- and post-processing of the data
%    (removing drifts, channels and time windows); the core ASR method (clean_asr) does not require 
%    this toolbox but you will need high-pass filtered data if you use it directly.
%  * By default this function will identify subsets of clean data from the given recording to
%    enhance the robustness of the ASR calibration phase to strongly contaminated data; this uses
%    the Statistics toolbox, but can be skipped/bypassed if needed (see documentation).
%
% In:
%   EEG : Raw continuous EEG recording to clean up (as EEGLAB dataset structure).
%
%
%   NOTE: The following parameters are the core parameters of the cleaning procedure; they should be
%   passed in as Name-Value Pairs. If the method removes too many (or too few) channels, time
%   windows, or general high-amplitude ("burst") artifacts, you will want to tune these values.
%   Hopefully you only need to do this in rare cases.
%
%   ChannelCriterion : Minimum channel correlation. If a channel is correlated at less than this
%                      value to an estimate based on other channels, it is considered abnormal in
%                      the given time window. This method requires that channel locations are
%                      available and roughly correct; otherwise a fallback criterion will be used.
%                      (default: 0.85)
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
%   WindowCriterion : Criterion for removing time windows that were not repaired completely. This may
%                     happen if the artifact in a window was composed of too many simultaneous
%                     uncorrelated sources (for example, extreme movements such as jumps). This is
%                     the maximum fraction of contaminated channels that are tolerated in the final
%                     output data for each considered window. Generally a lower value makes the
%                     criterion more aggressive. Default: 0.25. Reasonable range: 0.05 (very
%                     aggressive) to 0.3 (very lax).
%
%   Highpass : Transition band for the initial high-pass filter in Hz. This is formatted as
%              [transition-start, transition-end]. Default: [0.25 0.75].
%
%   NOTE: The following are detail parameters that may be tuned if one of the criteria does
%   not seem to be doing the right thing. These basically amount to side assumptions about the
%   data that usually do not change much across recordings, but sometimes do.
%
%   ChannelCriterionMaxBadTime : This is the maximum tolerated fraction of the recording duration 
%                                during which a channel may be flagged as "bad" without being
%                                removed altogether. Generally a lower (shorter) value makes the
%                                criterion more aggresive. Reasonable range: 0.15 (very aggressive)
%                                to 0.6 (very lax). Default: 0.5.
%
%   BurstCriterionRefMaxBadChns: If a number is passed in here, the ASR method will be calibrated based 
%                                on sufficiently clean data that is extracted first from the
%                                recording that is then processed with ASR. This number is the
%                                maximum tolerated fraction of "bad" channels within a given time
%                                window of the recording that is considered acceptable for use as
%                                calibration data. Any data windows within the tolerance range are
%                                then used for calibrating the threshold statistics. Instead of a
%                                number one may also directly pass in a data set that contains
%                                calibration data (for example a minute of resting EEG).
%
%                                If this is set to 'off', all data is used for calibration. This will 
%                                work as long as the fraction of contaminated data is lower than the
%                                the breakdown point of the robust statistics in the ASR
%                                calibration (50%, where 30% of clearly recognizable artifacts is a
%                                better estimate of the practical breakdown point).
%
%                                A lower value makes this criterion more aggressive. Reasonable
%                                range: 0.05 (very aggressive) to 0.3 (quite lax). If you have lots
%                                of little glitches in a few channels that don't get entirely
%                                cleaned you might want to reduce this number so that they don't go
%                                into the calibration data. Default: 0.075.
%
%   BurstCriterionRefTolerances : These are the power tolerances outside of which a channel in a
%                                 given time window is considered "bad", in standard deviations
%                                 relative to a robust EEG power distribution (lower and upper
%                                 bound). Together with the previous parameter this determines how
%                                 ASR calibration data is be extracted from a recording. Can also be
%                                 specified as 'off' to achieve the same effect as in the previous
%                                 parameter. Default: [-Inf 5.5].
%
%   BurstRejection : 'on' or 'off'. If 'on' reject portions of data containing burst instead of 
%                    correcting them using ASR. Default is 'off'.
%
%   WindowCriterionTolerances : These are the power tolerances outside of which a channel in the final
%                               output data is considered "bad", in standard deviations relative
%                               to a robust EEG power distribution (lower and upper bound). Any time
%                               window in the final (repaired) output which has more than the
%                               tolerated fraction (set by the WindowCriterion parameter) of channel
%                               with a power outside of this range will be considered incompletely 
%                               repaired and will be removed from the output. This last stage can be
%                               skipped either by setting the WindowCriterion to 'off' or by taking
%                               the third output of this processing function (which does not include
%                               the last stage). Default: [-Inf 7].
%
%   FlatlineCriterion : Maximum tolerated flatline duration. In seconds. If a channel has a longer
%                       flatline than this, it will be considered abnormal. Default: 5
%
%   NoLocsChannelCriterion : Criterion for removing bad channels when no channel locations are
%                            present. This is a minimum correlation value that a given channel must
%                            have w.r.t. a fraction of other channels. A higher value makes the
%                            criterion more aggressive. Reasonable range: 0.4 (very lax) - 0.6
%                            (quite aggressive). Default: 0.45.
%
%   NoLocsChannelCriterionExcluded : The fraction of channels that must be sufficiently correlated with
%                                    a given channel for it to be considered "good" in a given time
%                                    window. Applies only to the NoLocsChannelCriterion. This adds
%                                    robustness against pairs of channels that are shorted or other
%                                    that are disconnected but record the same noise process.
%                                    Reasonable range: 0.1 (fairly lax) to 0.3 (very aggressive);
%                                    note that increasing this value requires the ChannelCriterion
%                                    to be relaxed in order to maintain the same overall amount of
%                                    removed channels. Default: 0.1.
%
%   MaxMem : The maximum amount of memory in MB used by the algorithm when processing. 
%            See function asr_process for more information. Default is 64.
%
% Out:
%   EEG : Final cleaned EEG recording.
%
%   HP : Optionally just the high-pass filtered data.
%
%   BUR : Optionally the data without final removal of "irrecoverable" windows.
%
% Examples:
%   % Load a recording, clean it, and visualize the difference (using the defaults)
%   raw = pop_loadset(...);
%   clean = clean_artifacts(raw);
%   vis_artifacts(clean,raw);
%
%   % Use a more aggressive threshold (passing the parameters in by position)
%   raw = pop_loadset(...);
%   clean = clean_artifacts(raw,[],2.5);
%   vis_artifacts(clean,raw);
%
%   % Passing some parameter by name (here making the WindowCriterion setting less picky)
%   raw = pop_loadset(...);
%   clean = clean_artifacts(raw,'WindowCriterion',0.25);
%
%   % Disabling the WindowCriterion and ChannelCriterion altogether
%   raw = pop_loadset(...);
%   clean = clean_artifacts(raw,'WindowCriterion','off','ChannelCriterion','off');
%
%
%                                Christian Kothe, Swartz Center for Computational Neuroscience, UCSD
%                                2012-09-04

% Copyright (C) Christian Kothe, SCCN, 2012, ckothe@ucsd.edu
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

hlp_varargin2struct(varargin,...
    {'chancorr_crit','ChannelCorrelationCriterion','ChannelCriterion'}, 0.8, ...
    {'line_crit','LineNoiseCriterion'}, 4, ...
    {'burst_crit','BurstCriterion'}, 5, ...
    {'window_crit','WindowCriterion'}, 0.25, ...
    {'highpass_band','Highpass'}, [0.25 0.75], ...
    {'channel_crit_maxbad_time','ChannelCriterionMaxBadTime'}, 0.5, ...
    {'burst_crit_refmaxbadchns','BurstCriterionRefMaxBadChns'}, 0.075, ...
    {'burst_crit_reftolerances','BurstCriterionRefTolerances'}, [-inf 5.5], ...
    {'distance2','Distance'}, 'euclidian', ...
    {'window_crit_tolerances','WindowCriterionTolerances'},[-inf 7], ...
    {'burst_rejection','BurstRejection'},'off', ...
    {'nolocs_channel_crit','NoLocsChannelCriterion'}, 0.45, ...
    {'nolocs_channel_crit_excluded','NoLocsChannelCriterionExcluded'}, 0.1, ...
    {'max_mem','MaxMem'}, 64, ...
    {'flatline_crit','FlatlineCriterion'}, 5);

% remove flat-line channels
if ~strcmp(flatline_crit,'off')
    EEG = clean_flatlines(EEG,flatline_crit); 
end

% high-pass filter the data
if ~strcmp(highpass_band,'off')
    EEG = clean_drifts(EEG,highpass_band); 
end
if nargout > 1
    HP = EEG; 
end

% remove noisy channels by correlation and line-noise thresholds
if ~strcmp(chancorr_crit,'off') || ~strcmp(line_crit,'off') %#ok<NODEF>
    if strcmp(chancorr_crit,'off')
        chancorr_crit = 0; end
    if strcmp(line_crit,'off')
        line_crit = 100; end    
    try 
        [EEG,removed_channels] = clean_channels(EEG,chancorr_crit,line_crit,[],channel_crit_maxbad_time); 
    catch e
%         if strcmp(e.identifier,'clean_channels:bad_chanlocs')
            disp('Your dataset appears to lack correct channel locations; using a location-free channel cleaning method.');
            [EEG,removed_channels] = clean_channels_nolocs(EEG,nolocs_channel_crit,nolocs_channel_crit_excluded,[],channel_crit_maxbad_time); 
%         else
%             rethrow(e);
%         end
    end
end

% repair bursts by ASR
if ~strcmp(burst_crit,'off')
    if ~strcmpi(distance2, 'euclidian')    
        BUR = clean_asr(EEG,burst_crit,[],[],[],burst_crit_refmaxbadchns,burst_crit_reftolerances,[], [], true, max_mem); 
    else
        BUR = clean_asr(EEG,burst_crit,[],[],[],burst_crit_refmaxbadchns,burst_crit_reftolerances,[], [], false, max_mem); 
    end

    if strcmp(burst_rejection,'on')
        % portion of data which have changed
        sample_mask = sum(abs(EEG.data-BUR.data),1) < 1e-10;

        % find latency of regions
        retain_data_intervals = reshape(find(diff([false sample_mask false])),2,[])';
        retain_data_intervals(:,2) = retain_data_intervals(:,2)-1;

        % reject regions
        EEG = pop_select(EEG, 'point', retain_data_intervals);
        EEG.etc.clean_sample_mask = sample_mask;
    else
        EEG = BUR;
    end
end

if nargout > 2
    BUR = EEG; end

% remove irrecoverable time windows based on power
if ~strcmp(window_crit,'off') && ~strcmp(window_crit_tolerances,'off')
    disp('Now doing final post-cleanup of the output.');
    EEG = clean_windows(EEG,window_crit,window_crit_tolerances); 
end
disp('Use vis_artifacts to compare the cleaned data to the original.');
