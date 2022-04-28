% The files 'constants-raw.mat', 'psdmed-raw.mat',
% 'psd-step-by-step-raw.mat' and 'psd-raw.mat' were obtained from the
% sample EEGLAB dataset 'sample-raw.set'.

% 'constants-raw.mat' --------------------------
% sha1: 72f7f1260a5b294f9b79b9d65c4a83f207a71305
% ----------------------------------------------

% 'psd-step-by-step-raw.mat' -------------------
% sha1: 9266c1002174a22ec5237f5d892e2fde937fc552
% ----------------------------------------------

% 'psd-raw.mat' --------------------------------
% sha1: 9266c1002174a22ec5237f5d892e2fde937fc552
% ----------------------------------------------


% Load
EEG = pop_loadset('sample-raw.set');
EEG = eeg_checkset(EEG);

% Calculate ICA activations
EEG.icaact = eeg_getica(EEG);
EEG.icaact = double(EEG.icaact);


%% Calculate constants from eeg_rpsd.m
% clean input cutoff freq
nyquist = floor(EEG.srate / 2);
nfreqs = 100;
if nfreqs > nyquist
    nfreqs = nyquist;
end
pct_data = 100;  % useless variable

% setup constants
ncomp = size(EEG.icaweights, 1);
n_points = min(EEG.pnts, EEG.srate);
window = windows('hamming', n_points, 0.54)';
cutoff = floor(EEG.pnts / n_points) * n_points;
index = bsxfun(@plus, ceil(0:n_points / 2:cutoff - n_points), (1:n_points)');
if ~exist('OCTAVE_VERSION', 'builtin')
    rng('default');
else
    rand('state', 0);
end
n_seg = size(index, 2) * EEG.trials;
subset = randperm(n_seg, ceil(n_seg * pct_data / 100));
if exist('OCTAVE_VERSION', 'builtin') == 0
    rng('shuffle');
end


%% Export constants from eeg_rpsd.m
constants = struct(...
    'ncomp', ncomp, ...
    'nfreqs', nfreqs, ...
    'n_points', n_points, ...
    'nyquist', nyquist, ...
    'index', index, ...
    'window', window, ...
    'subset', subset);
save('constants-raw', 'constants');


%% calculate windowed spectrums
psdmed = zeros(ncomp, nfreqs);
for it = 1:ncomp
    temp = reshape(EEG.icaact(it, index, :), [1 size(index) .* [1 EEG.trials]]);
    temp = bsxfun(@times, temp(:, :, subset), window);
    temp = fft(temp, n_points, 2);
    temp = temp .* conj(temp);
    temp = temp(:, 2:nfreqs + 1, :) * 2 / (EEG.srate*sum(window.^2));
    if nfreqs == nyquist
        temp(:, end, :) = temp(:, end, :) / 2;
    end

    psdmed(it, :) = 20 * log10(median(temp, 3));
end


%% Format and undo notch filter
% extrapolate or prune as needed
nfreq = size(psdmed, 2);
if nfreq < 100
    psdmed = [psdmed, repmat(psdmed(:, end), 1, 100 - nfreq)];
end

% undo notch filter
for linenoise_ind = [50, 60]
    linenoise_around = [linenoise_ind - 1, linenoise_ind + 1];
    difference = bsxfun(@minus, psdmed(:, linenoise_around), ...
        psdmed(:, linenoise_ind));
    notch_ind = all(difference > 5, 2);
    if any(notch_ind)
        psdmed(notch_ind, linenoise_ind) = ...
            mean(psdmed(notch_ind, linenoise_around), 2);
    end
end

% normalize
psd = bsxfun(@rdivide, psdmed, max(abs(psdmed), [], 2));

% reshape and cast
psd = single(permute(psd, [3 2 4 1]));


%% Export psd
save('psd-step-by-step-raw', 'psd')


%% Re-create psd feature by using 'eeg_rpsd' directly

% Load
EEG = pop_loadset('sample-raw.set');
EEG = eeg_checkset(EEG);

% Calculate ICA activations
EEG.icaact = eeg_getica(EEG);
EEG.icaact = double(EEG.icaact);

% Compute PSD
psd = eeg_rpsd(EEG, 100);

% extrapolate or prune as needed
nfreq = size(psd, 2);
if nfreq < 100
    psd = [psd, repmat(psd(:, end), 1, 100 - nfreq)];
end

% undo notch filter
for linenoise_ind = [50, 60]
    linenoise_around = [linenoise_ind - 1, linenoise_ind + 1];
    difference = bsxfun(@minus, psd(:, linenoise_around), ...
        psd(:, linenoise_ind));
    notch_ind = all(difference > 5, 2);
    if any(notch_ind)
        psd(notch_ind, linenoise_ind) = mean(psd(notch_ind, linenoise_around), 2);
    end
end

% normalize
psd = bsxfun(@rdivide, psd, max(abs(psd), [], 2));

% reshape and cast
psd = single(permute(psd, [3 2 4 1]));


%% Export psd
save('psd-raw', 'psd')
