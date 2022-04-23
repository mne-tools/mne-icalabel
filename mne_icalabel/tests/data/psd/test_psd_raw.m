% The files 'constants-raw.mat' and 'psdmed-raw.mat' were obtained from
% the sample EEGLAB dataset 'sample-raw.set'.

% 'constants-raw.mat' --------------------------
% sha1:
% ----------------------------------------------

% 'psdmed-raw.mat' -----------------------------
% sha1:
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

%% Export psdmed from eeg_rpsd.m
save('psdmed-raw', 'psdmed');
