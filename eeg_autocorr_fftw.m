function [ac2,resamp] = eeg_autocorr_fftw(icaact, trials, srate, pnts, pct_data)
if ~exist('pct_data', 'var') || isempty(pct_data)
    pct_datas = 100;
end

nfft = 2^nextpow2(2*pnts-1);

% calc autocorrelation
fftw('planner', 'hybrid');
ac = zeros(size(icaact, 1), nfft, trials);

for it = 1:size(icaact, 1)
    X = fft(icaact(it, :, :), nfft, 2);
    %if it == 1
    %    ac2 = X(:,:,:);
    %end
    ac(it, :, :) = abs(X).^2;
end
ac2 = ac(:,:,:);
ac = ifft(mean(ac, 3), [], 2);

if pnts < srate
    ac = [ac(:, 1:pnts) zeros(size(ac, 1), srate - pnts + 1)];
else
    ac = ac(:, 1:srate + 1);
end

% normalize by 0-tap autocorrelation
ac = bsxfun(@rdivide, ac(:, 1:srate + 1, :), ac(:, 1)); 

% resample to 1 second at 100 samples/sec
resamp = resample(ac', 100, srate)';
resamp(:, 1) = [];