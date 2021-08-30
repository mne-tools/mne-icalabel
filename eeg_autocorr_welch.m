function ac = eeg_autocorr_welch(EEG, pct_data)

% clean input cutoff freq
if ~exist('pct_data', 'var') || isempty(pct_data)
    pct_data = 100;
end

% setup constants
ncomp = size(EEG.icaweights, 1);
n_points = min(EEG.pnts, EEG.srate * 3);
nfft = 2^nextpow2(n_points * 2 - 1);
cutoff = floor(EEG.pnts / n_points) * n_points;
index = bsxfun(@plus, ceil(0:n_points / 2:cutoff - n_points), (1:n_points)');

% separate data segments
if pct_data ~=100
    rng(0)
    n_seg = size(index, 2) * EEG.trials;
    subset = randperm(n_seg, ceil(n_seg * pct_data / 100)); % need to find a better way to take subset
    rng('shuffle') % remove duplicate data first (from 50% overlap)
    temp = reshape(EEG.icaact(:, index, :), [ncomp size(index) .* [1 EEG.trials]]);
    segments = temp(:, :, subset);
else
    segments = reshape(EEG.icaact(:, index, :), [ncomp size(index) .* [1 EEG.trials]]);
end

%% calc autocorrelation
fftpow = abs(fft(segments, nfft, 2)).^2;
ac = ifft(mean(fftpow, 3), [], 2);

% normalize
if EEG.pnts < EEG.srate
    ac = [ac(:, 1:EEG.pnts, :) ./ (ac(:, 1) * (n_points:-1:1) / (n_points)) ...
        zeros(ncomp, EEG.srate - n_points + 1)];
else
    ac = ac(:, 1:EEG.srate + 1, :) ./ ...
        (ac(:, 1) * [(n_points:-1:n_points - EEG.srate + 1) ...
        max(1, n_points - EEG.srate)] / (n_points));
end

% resample to 1 second at 100 samples/sec
ac = resample(ac', 100, EEG.srate)';
ac(:, 1) = [];