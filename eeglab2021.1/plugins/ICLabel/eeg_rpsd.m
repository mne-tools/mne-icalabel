function psdmed = eeg_rpsd(EEG, nfreqs, pct_data)

% clean input cutoff freq
nyquist = floor(EEG.srate / 2);
if ~exist('nfreqs', 'var') || isempty(nfreqs)
    nfreqs = nyquist;
elseif nfreqs > nyquist
    nfreqs = nyquist;
end
if ~exist('pct_data', 'var') || isempty(pct_data)
    pct_data = 100;
end

% setup constants
ncomp = size(EEG.icaweights, 1);
n_points = min(EEG.pnts, EEG.srate);
window = windows('hamming', n_points, 0.54)';
cutoff = floor(EEG.pnts / n_points) * n_points;
index = bsxfun(@plus, ceil(0:n_points / 2:cutoff - n_points), (1:n_points)');
if ~exist('OCTAVE_VERSION', 'builtin')
    rng('default')
else
    rand('state', 0);
end
n_seg = size(index, 2) * EEG.trials;
subset = randperm(n_seg, ceil(n_seg * pct_data / 100)); % need to improve this
if exist('OCTAVE_VERSION', 'builtin') == 0
    rng('shuffle')
end

% calculate windowed spectrums
psdmed = zeros(ncomp, nfreqs);
for it = 1:ncomp
    temp = reshape(EEG.icaact(it, index, :), [1 size(index) .* [1 EEG.trials]]);
    temp = bsxfun(@times, temp(:, :, subset), window);
    temp = fft(temp, n_points, 2);
    temp = temp .* conj(temp);
    temp = temp(:, 2:nfreqs + 1, :) * 2 / (EEG.srate*sum(window.^2));
    if nfreqs == nyquist
        temp(:, end, :) = temp(:, end, :) / 2; end

    psdmed(it, :) = 20 * log10(median(temp, 3));
end
end
