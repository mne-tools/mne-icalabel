function resamp = eeg_autocorr(EEG, pct_data)

if ~exist('pct_data', 'var') || isempty(pct_data)
    pct_data = 100;
end

ncomp = size(EEG.icaact, 1);
nfft = 2^nextpow2(2*EEG.pnts-1);

% calc autocorrelation
c = zeros(ncomp, nfft);
for it = 1:ncomp
    X = mean(abs(fft(EEG.icaact(it, :), nfft, 2)).^2, 3);
    c(it, :) = ifft(X, [], 2);
end
% X = fft(EEG.icaact, 2^nextpow2(2*EEG.pnts-1), 2);
% c = ifft(mean(abs(X).^2, 3), [], 2);
if EEG.pnts < EEG.srate
    ac = [c(:, 1:EEG.pnts, :) zeros(size(c, 1), EEG.srate - EEG.pnts + 1)];
else
    ac = c(:, 1:EEG.srate + 1, :);
end

% normalize by 0-tap autocorrelation
ac = bsxfun(@rdivide, ac, ac(:, 1)); 

% resample to 1 second at 100 samples/sec
resamp = resample(ac', 100, EEG.srate)';
resamp(:, 1) = [];