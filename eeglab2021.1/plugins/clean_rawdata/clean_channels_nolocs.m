function [signal,removed_channels] = clean_channels_nolocs(signal,min_corr,ignored_quantile,window_len,max_broken_time,linenoise_aware)
% Remove channels with abnormal data from a continuous data set.
% Signal = clean_channels_nolocs(Signal,MinCorrelation,IgnoredQuantile,WindowLength,MaxBrokenTime,LineNoiseAware)
%
% This is an automated artifact rejection function which ensures that the data contains no channels
% that record only noise for extended periods of time. If channels with control signals are
% contained in the data these are usually also removed. The criterion is based on correlation: if a
% channel is decorrelated from all others (pairwise correlation < a given threshold), excluding a
% given fraction of most correlated channels -- and if this holds on for a sufficiently long fraction 
% of the data set -- then the channel is removed.
%
% In:
%   Signal          : Continuous data set, assumed to be appropriately high-passed (e.g. >0.5Hz or
%                     with a 0.5Hz - 2.0Hz transition band).
%
%   MinCorrelation  : Minimum correlation between a channel and any other channel (in a short period 
%                     of time) below which the channel is considered abnormal for that time period.
%                     Reasonable range: 0.4 (very lax) to 0.6 (quite aggressive). The default is 0.45. 
%                     
%
%   The following are detail parameters that usually do not have to be tuned. If you cannot get
%   the function to do what you want, you might consider adapting these to your data.
%   
%   IgnoredQuantile : Fraction of channels that need to have at least the given MinCorrelation value
%                     w.r.t. the channel under consideration. This allows to deal with channels or
%                     small groups of channels that measure the same noise source, e.g. if they are
%                     shorted. If many channels can be disconnected during an experiment and you
%                     have strong noise in the room, you might increase this fraction, but consider
%                     that this a) requires you to decrease the MinCorrelation appropriately and b)
%                     this can make the correlation measure more brittle. Reasonable range: 0.05 (rather
%                     lax) to 0.2 (very tolerant re disconnected/shorted channels).The default is
%                     0.1.
%
%   WindowLength    : Length of the windows (in seconds) for which correlation is computed; ideally
%                     short enough to reasonably capture periods of global artifacts (which are
%                     ignored), but not shorter (for statistical reasons). Default: 2.
% 
%   MaxBrokenTime : Maximum time (either in seconds or as fraction of the recording) during which a 
%                   retained channel may be broken. Reasonable range: 0.1 (very aggressive) to 0.6
%                   (very lax). The default is 0.5.
%
%   LineNoiseAware : Whether the operation should be performed in a line-noise aware manner. If enabled,
%                    the correlation measure will not be affected by the presence or absence of line 
%                    noise (using a temporary notch filter). Default: true.
%
% Out:
%   Signal : data set with bad channels removed
%
% Notes:
%   This function requires the Signal Processing toolbox.
%
%                                Christian Kothe, Swartz Center for Computational Neuroscience, UCSD
%                                2010-07-06

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

if ~exist('min_corr','var') || isempty(min_corr) min_corr = 0.45; end
if ~exist('ignored_quantile','var') || isempty(ignored_quantile) ignored_quantile = 0.1; end
if ~exist('window_len','var') || isempty(window_len) window_len = 2; end
if ~exist('max_broken_time','var') || isempty(max_broken_time) max_broken_time = 0.5; end
if ~exist('linenoise_aware','var') || isempty(linenoise_aware) linenoise_aware = true; end


% flag channels
if max_broken_time > 0 && max_broken_time < 1  %#ok<*NODEF>
    max_broken_time = size(signal.data,2)*max_broken_time;
else
    max_broken_time = signal.srate*max_broken_time;
end

signal.data = double(signal.data);
[C,S] = size(signal.data);
window_len = window_len*signal.srate;
wnd = 0:window_len-1;
offsets = 1:window_len:S-window_len;
W = length(offsets);
retained = 1:(C-ceil(C*ignored_quantile));

% optionally ignore both 50 and 60 Hz spectral components...
if linenoise_aware
    Bwnd = design_kaiser(2*45/signal.srate,2*50/signal.srate,60,true);
    if signal.srate <= 130
        B = design_fir(length(Bwnd)-1,[2*[0 45 50 55]/signal.srate 1],[1 1 0 1 1],[],Bwnd);
    else
        B = design_fir(length(Bwnd)-1,[2*[0 45 50 55 60 65]/signal.srate 1],[1 1 0 1 0 1 1],[],Bwnd);
    end
    for c=signal.nbchan:-1:1
        X(:,c) = filtfilt_fast(B,1,signal.data(c,:)'); end
else
    X = signal.data';
end

% for each window, flag channels with too low correlation to any other channel (outside the
% ignored quantile)
flagged = zeros(C,W);
for o=1:W
    sortcc = sort(abs(corrcoef(X(offsets(o)+wnd,:))));
    flagged(:,o) = all(sortcc(retained,:) < min_corr);
end

% mark all channels for removal which have more flagged samples than the maximum number of
% ignored samples
removed_channels = sum(flagged,2)*window_len > max_broken_time;

% apply removal
if all(removed_channels)
    disp('Warning: all channels are flagged bad according to the used criterion: not removing anything.');
elseif any(removed_channels)
    disp('Now removing bad channels...');
    try
        signal = pop_select(signal,'nochannel',find(removed_channels));
    catch e
        if ~exist('pop_select','file')
            disp('Apparently you do not have EEGLAB''s pop_select() on the path.');
        else
            disp('Could not select channels using EEGLAB''s pop_select(); details: ');
            hlp_handleerror(e,1);
        end
        disp('Falling back to a basic substitute and dropping signal meta-data.');
        if length(signal.chanlocs) == size(signal.data,1)
            signal.chanlocs = signal.chanlocs(~removed_channels); end
        signal.data = signal.data(~removed_channels,:);
        signal.nbchan = size(signal.data,1);
        [signal.icawinv,signal.icasphere,signal.icaweights,signal.icaact,signal.stats,signal.specdata,signal.specicaact] = deal([]);
    end
    if isfield(signal.etc,'clean_channel_mask')
        signal.etc.clean_channel_mask(signal.etc.clean_channel_mask) = ~removed_channels;
    else
        signal.etc.clean_channel_mask = ~removed_channels;
    end
end
