function [psdmed, psdvar, psdkrt] = eeg_rpsd(EEG, nfreqs, pct_data)

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
window = hamming(n_points)';
cutoff = floor(EEG.pnts / n_points) * n_points;
index = bsxfun(@plus, ceil(0:n_points / 2:cutoff - n_points), (1:n_points)');
rng(0)
n_seg = size(index, 2) * EEG.trials;
subset = randperm(n_seg, ceil(n_seg * pct_data / 100)); % need to improve this
rng('shuffle')

try
    %% slightly faster but high memory use
    % calculate windowed spectrums
    temp = reshape(EEG.icaact(:, index, :), [ncomp size(index) .* [1 EEG.trials]]);
    temp = bsxfun(@times, temp(:, :, subset), window);
    temp = fft(temp, n_points, 2);
    temp = abs(temp).^2;
    temp = temp(:, 2:nfreqs + 1, :) * 2 / (EEG.srate*sum(window.^2));
    if nfreqs == nyquist
        temp(:, end, :) = temp(:, end, :) / 2; end

    % calculate outputs
    psdmed = db(median(temp, 3));
    psdvar = db(var(temp, [], 3), 'power');
    psdkrt = db(kurtosis(temp, [], 3)/4);

catch
    %% lower memory but slightly slower
    psdmed = zeros(ncomp, nfreqs);
    psdvar = zeros(ncomp, nfreqs);
    psdkrt = zeros(ncomp, nfreqs);
    for it = 1:ncomp
        % calculate windowed spectrums
        temp = reshape(EEG.icaact(it, index, :), [1 size(index) .* [1 EEG.trials]]);
        temp = bsxfun(@times, temp(:, :, subset), window);
        temp = fft(temp, n_points, 2);
        temp = temp .* conj(temp);
        temp = temp(:, 2:nfreqs + 1, :) * 2 / (EEG.srate*sum(window.^2));
        if nfreqs == nyquist
            temp(:, end, :) = temp(:, end, :) / 2; end

        psdmed(it, :) = db(median(temp, 3));
        psdvar(it, :) = db(var(temp, [], 3), 'power');
        psdkrt(it, :) = db(kurtosis(temp, [], 3)/4);
    end
end
end
