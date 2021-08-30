function [signal,sample_mask] = clean_windows(signal,max_bad_channels,zthresholds,window_len,window_overlap,max_dropout_fraction,min_clean_fraction,truncate_quant,step_sizes,shape_range)
% Remove periods with abnormally high-power content from continuous data.
% [Signal,Mask] = clean_windows(Signal,MaxBadChannels,PowerTolerances,WindowLength,WindowOverlap,MaxDropoutFraction,Min)
%
% This function cuts segments from the data which contain high-power artifacts. Specifically,
% only windows are retained which have less than a certain fraction of "bad" channels, where a channel
% is bad in a window if its power is above or below a given upper/lower threshold (in standard 
% deviations from a robust estimate of the EEG power distribution in the channel).
%
% In:
%   Signal         : Continuous data set, assumed to be appropriately high-passed (e.g. >1Hz or
%                    0.5Hz - 2.0Hz transition band)
%
%   MaxBadChannels : The maximum number or fraction of bad channels that a retained window may still
%                    contain (more than this and it is removed). Reasonable range is 0.05 (very clean
%                    output) to 0.3 (very lax cleaning of only coarse artifacts). Default: 0.2.
%
%   PowerTolerances: The minimum and maximum standard deviations within which the power of a channel
%                    must lie (relative to a robust estimate of the clean EEG power distribution in 
%                    the channel) for it to be considered "not bad". Default: [-3.5 5].
%
%
%   The following are detail parameters that usually do not have to be tuned. If you can't get
%   the function to do what you want, you might consider adapting these to your data.
%
%   WindowLength    : Window length that is used to check the data for artifact content. This is 
%                     ideally as long as the expected time scale of the artifacts but not shorter 
%                     than half a cycle of the high-pass filter that was used. Default: 1.
%
%   WindowOverlap : Window overlap fraction. The fraction of two successive windows that overlaps.
%                   Higher overlap ensures that fewer artifact portions are going to be missed (but
%                   is slower). (default: 0.66)
% 
%   MaxDropoutFraction : Maximum fraction that can have dropouts. This is the maximum fraction of
%                        time windows that may have arbitrarily low amplitude (e.g., due to the
%                        sensors being unplugged). (default: 0.1)
%
%   MinCleanFraction : Minimum fraction that needs to be clean. This is the minimum fraction of time
%                      windows that need to contain essentially uncontaminated EEG. (default: 0.25)
%
%   
%   The following are expert-level parameters that you should not tune unless you fully understand
%   how the method works.
%
%   TruncateQuantile : Truncated Gaussian quantile. Quantile range [upper,lower] of the truncated
%                      Gaussian distribution that shall be fit to the EEG contents. (default: [0.022 0.6])
%
%   StepSizes : Grid search stepping. Step size of the grid search, in quantiles; separately for
%               [lower,upper] edge of the truncated Gaussian. The lower edge has finer stepping
%               because the clean data density is assumed to be lower there, so small changes in
%               quantile amount to large changes in data space. (default: [0.01 0.01])
%
%   ShapeRange : Shape parameter range. Search range for the shape parameter of the generalized
%                Gaussian distribution used to fit clean EEG. (default: 1.7:0.15:3.5)
%
% Out:
%   Signal : data set with bad time periods removed.
%
%   Mask   : mask of retained samples (logical array)
%
%                                Christian Kothe, Swartz Center for Computational Neuroscience, UCSD
%                                2010-07-06

% Copyright (C) Christian Kothe, SCCN, 2010, ckothe@ucsd.edu
%
% History
% 04/26/2017 Makoto. Changed action When EEG.etc.clean_sample_mask is present,
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

% handle inputs
if ~exist('max_bad_channels','var') || isempty(max_bad_channels) max_bad_channels = 0.2; end
if ~exist('zthresholds','var') || isempty(zthresholds) zthresholds = [-3.5 5]; end
if ~exist('window_len','var') || isempty(window_len) window_len = 1; end
if ~exist('window_overlap','var') || isempty(window_overlap) window_overlap = 0.66; end
if ~exist('max_dropout_fraction','var') || isempty(max_dropout_fraction) max_dropout_fraction = 0.1; end
if ~exist('min_clean_fraction','var') || isempty(min_clean_fraction) min_clean_fraction = 0.25; end
if ~exist('truncate_quant','var') || isempty(truncate_quant) truncate_quant = [0.022 0.6]; end
if ~exist('step_sizes','var') || isempty(step_sizes) step_sizes = [0.01 0.01]; end
if ~exist('shape_range','var') || isempty(shape_range) shape_range = 1.7:0.15:3.5; end
if ~isempty(max_bad_channels) && max_bad_channels > 0 && max_bad_channels < 1 %#ok<*NODEF>
    max_bad_channels = round(size(signal.data,1)*max_bad_channels); end

signal.data = double(signal.data);
[C,S] = size(signal.data);
N = window_len*signal.srate;
wnd = 0:N-1;
offsets = round(1:N*(1-window_overlap):S-N);

fprintf('Determining time window rejection thresholds...');
% for each channel...
for c = C:-1:1
    % compute RMS amplitude for each window...
    X = signal.data(c,:).^2;
    X = sqrt(sum(X(bsxfun(@plus,offsets,wnd')))/N);
    % robustly fit a distribution to the clean EEG part
    [mu,sig] = fit_eeg_distribution(X, ...
        min_clean_fraction, max_dropout_fraction, ...
        truncate_quant, step_sizes,shape_range);
    % calculate z scores relative to that
    wz(c,:) = (X - mu)/sig;
end
disp('done.');

% sort z scores into quantiles
swz = sort(wz);
% determine which windows to remove
remove_mask = false(1,size(swz,2));
if max(zthresholds)>0
    remove_mask(swz(end-max_bad_channels,:) > max(zthresholds)) = true; end
if min(zthresholds)<0
    remove_mask(swz(1+max_bad_channels,:) < min(zthresholds)) = true; end
removed_windows = find(remove_mask);

% find indices of samples to remove
removed_samples = repmat(offsets(removed_windows)',1,length(wnd))+repmat(wnd,length(removed_windows),1);
% mask them out
sample_mask = true(1,S); 
sample_mask(removed_samples(:)) = false;
fprintf('Keeping %.1f%% (%.0f seconds) of the data.\n',100*(mean(sample_mask)),nnz(sample_mask)/signal.srate);
% determine intervals to retain
retain_data_intervals = reshape(find(diff([false sample_mask false])),2,[])';
retain_data_intervals(:,2) = retain_data_intervals(:,2)-1;

% apply selection
try
    signal = pop_select(signal, 'point', retain_data_intervals);
catch e
    if ~exist('pop_select','file')
        disp('Apparently you do not have EEGLAB''s pop_select() on the path.');
    else
        disp('Could not select time windows using EEGLAB''s pop_select(); details: ');
        hlp_handleerror(e,1);
    end
    %disp('Falling back to a basic substitute and dropping signal meta-data.');
    warning('Falling back to a basic substitute and dropping signal meta-data.');
    signal.data = signal.data(:,sample_mask);
    signal.pnts = size(signal.data,2);
    signal.xmax = signal.xmin + (signal.pnts-1)/signal.srate;    
    [signal.event,signal.urevent,signal.epoch,signal.icaact,signal.reject,signal.stats,signal.specdata,signal.specicaact] = deal(signal.event([]),signal.urevent([]),[],[],[],[],[],[]);
end
% if isfield(signal.etc,'clean_sample_mask')
%     signal.etc.clean_sample_mask(signal.etc.clean_sample_mask) = sample_mask;
% else
%     signal.etc.clean_sample_mask = sample_mask;
% end
if isfield(signal.etc,'clean_sample_mask')
    oneInds = find(signal.etc.clean_sample_mask == 1);
    if length(oneInds) == length(sample_mask)
        signal.etc.clean_sample_mask(oneInds) = sample_mask;
    else
        warning('EEG.etc.clean_sample is present. It is overwritten.');
    end
else
    signal.etc.clean_sample_mask = sample_mask;
end
