function [signal,removed_channels] = clean_channels(signal,corr_threshold,noise_threshold,window_len,max_broken_time,num_samples,subset_size)
% Remove channels with abnormal data from a continuous data set.
% Signal = clean_channels(Signal,CorrelationThreshold,WindowLength,MaxBrokenTime,NumSamples,SubsetSize,UseGPU)
%
% This is an automated artifact rejection function which ensures that the data contains no channels
% that record only noise for extended periods of time. If channels with control signals are
% contained in the data these are usually also removed. The criterion is based on correlation: if a
% channel has lower correlation to its robust estimate (based on other channels) than a given threshold
% for a minimum period of time (or percentage of the recording), it will be removed.
%
% In:
%   Signal          : Continuous data set, assumed to be appropriately high-passed (e.g. >0.5Hz or
%                     with a 0.5Hz - 2.0Hz transition band).
%
%   CorrelationThreshold : Correlation threshold. If a channel is correlated at less than this value
%                          to its robust estimate (based on other channels), it is considered abnormal in
%                          the given time window. Default: 0.85.
%                     
%   LineNoiseThreshold : If a channel has more line noise relative to its signal than this value, in
%                        standard deviations from the channel population mean, it is considered abnormal.
%                        Default: 4.
%
%
%   The following are detail parameters that usually do not have to be tuned. If you cannot get
%   the function to do what you want, you might consider adapting these to your data.
%   
%   WindowLength    : Length of the windows (in seconds) for which correlation is computed; ideally
%                     short enough to reasonably capture periods of global artifacts or intermittent 
%                     sensor dropouts, but not shorter (for statistical reasons). Default: 5.
% 
%   MaxBrokenTime : Maximum time (either in seconds or as fraction of the recording) during which a 
%                   retained channel may be broken. Reasonable range: 0.1 (very aggressive) to 0.6
%                   (very lax). The default is 0.4.
%
%   NumSamples : Number of RANSAC samples. This is the number of samples to generate in the random
%                sampling consensus process. The larger this value, the more robust but also slower 
%                the processing will be. Default: 50.
%
%   SubsetSize : Subset size. This is the size of the channel subsets to use for robust reconstruction, 
%                as a fraction of the total number of channels. Default: 0.25.
%
% Out:
%   Signal : data set with bad channels removed
%
%                                Christian Kothe, Swartz Center for Computational Neuroscience, UCSD
%                                2014-05-12

% Copyright (C) Christian Kothe, SCCN, 2014, christian@sccn.ucsd.edu
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

if ~exist('corr_threshold','var') || isempty(corr_threshold) corr_threshold = 0.8; end
if ~exist('noise_threshold','var') || isempty(noise_threshold) noise_threshold = 4; end
if ~exist('window_len','var') || isempty(window_len) window_len = 5; end
if ~exist('max_broken_time','var') || isempty(max_broken_time) max_broken_time = 0.4; end
if ~exist('num_samples','var') || isempty(num_samples) num_samples = 50; end
if ~exist('subset_size','var') || isempty(subset_size) subset_size = 0.25; end

subset_size = round(subset_size*size(signal.data,1)); 

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

fprintf('Scanning for bad channels...\n');

if signal.srate > 100
    % remove signal content above 50Hz
    B = design_fir(100,[2*[0 45 50]/signal.srate 1],[1 1 0 0]);
    for c=signal.nbchan:-1:1
        X(:,c) = filtfilt_fast(B,1,signal.data(c,:)'); end
    % determine z-scored level of EM noise-to-signal ratio for each channel
    noisiness = mad(signal.data'-X)./mad(X,1);
    znoise = (noisiness - median(noisiness)) ./ (mad(noisiness,1)*1.4826);        
    % trim channels based on that
    noise_mask = znoise > noise_threshold;
else
    X = signal.data';
    noise_mask = false(C,1)'; % transpose added. Otherwise gives an error below at removed_channels = removed_channels | noise_mask';  (by Ozgur Balkan)
end

if ~(isfield(signal.chanlocs,'X') && isfield(signal.chanlocs,'Y') && isfield(signal.chanlocs,'Z') && all([length([signal.chanlocs.X]),length([signal.chanlocs.Y]),length([signal.chanlocs.Z])] > length(signal.chanlocs)*0.5))
    error('clean_channels:bad_chanlocs','To use this function most of your channels should have X,Y,Z location measurements.'); end

% get the matrix of all channel locations [3xN]
[x,y,z] = deal({signal.chanlocs.X},{signal.chanlocs.Y},{signal.chanlocs.Z});
usable_channels = find(~cellfun('isempty',x) & ~cellfun('isempty',y) & ~cellfun('isempty',z));
locs = [cell2mat(x(usable_channels));cell2mat(y(usable_channels));cell2mat(z(usable_channels))];
X = X(:,usable_channels);
  
% caculate all-channel reconstruction matrices from random channel subsets   
if exist('OCTAVE_VERSION', 'builtin') == 0
    P = hlp_microcache('cleanchans',@calc_projector,locs,num_samples,subset_size);
else
    P = calc_projector(locs,num_samples,subset_size);
end
corrs = zeros(length(usable_channels),W);
        
% calculate each channel's correlation to its RANSAC reconstruction for each window
timePassedList = zeros(W,1);
for o=1:W
    tic; % makoto
    XX = X(offsets(o)+wnd,:);
    YY = sort(reshape(XX*P,length(wnd),length(usable_channels),num_samples),3);
    YY = YY(:,:,round(end/2));
	corrs(:,o) = sum(XX.*YY)./(sqrt(sum(XX.^2)).*sqrt(sum(YY.^2)));
    timePassedList(o) = toc; % makoto
    medianTimePassed = median(timePassedList(1:o));
    fprintf('clean_channel: %3.0d/%d blocks, %.1f minutes remaining.\n', o, W, medianTimePassed*(W-o)/60); % makoto
end
        
flagged = corrs < corr_threshold;
        
% mark all channels for removal which have more flagged samples than the maximum number of
% ignored samples
removed_channels = false(C,1);
removed_channels(usable_channels) = sum(flagged,2)*window_len > max_broken_time;
removed_channels = removed_channels | noise_mask';

% apply removal
if mean(removed_channels) > 0.75
    error('clean_channels:bad_chanlocs','More than 75%% of your channels were removed -- this is probably caused by incorrect channel location measurements (e.g., wrong cap design).');
elseif any(removed_channels)
    try
        signal = pop_select(signal,'nochannel',find(removed_channels));
    catch e
        if ~exist('pop_select','file')
            disp('Apparently you do not have EEGLAB''s pop_select() on the path.');
        else
            disp('Could not select channels using EEGLAB''s pop_select(); details: ');
            hlp_handleerror(e,1);
        end
        fprintf('Removing %i channels and dropping signal meta-data.\n',nnz(removed_channels));
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



% calculate a bag of reconstruction matrices from random channel subsets
function P = calc_projector(locs,num_samples,subset_size)
%stream = RandStream('mt19937ar','Seed',435656);
rand_samples = {};
for k=num_samples:-1:1
    tmp = zeros(size(locs,2));
    subset = randsample(1:size(locs,2),subset_size);
%    subset = randsample(1:size(locs,2),subset_size,stream);
    tmp(subset,:) = real(sphericalSplineInterpolate(locs(:,subset),locs))';
    rand_samples{k} = tmp;
end
P = horzcat(rand_samples{:});


function Y = randsample(X,num)
Y = [];
while length(Y)<num
    pick = round(1 + (length(X)-1).*rand());
    Y(end+1) = X(pick);
    X(pick) = [];
end
% 
% function Y = randsample(X,num,stream)
% Y = [];
% while length(Y)<num
%     pick = round(1 + (length(X)-1).*rand(stream));
%     Y(end+1) = X(pick);
%     X(pick) = [];
% end

function Y = mad(X,flag) %#ok<INUSD>
Y = median(abs(bsxfun(@minus,X,median(X))));
