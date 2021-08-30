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
ac = zeros(ncomp, nfft);
for it = 1:ncomp
    fftpow = mean(abs(fft(segments(it, :, :), nfft, 2)).^2, 3);
    ac(it, :) = ifft(fftpow, [], 2);
end
% fftpow = abs(fft(segments, nfft, 2)).^2;
% ac = ifft(mean(fftpow, 3), [], 2);

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
if ~exist('OCTAVE_VERSION', 'builtin') && exist('resample')
    ac = resample(double(ac'), 100, EEG.srate)';
else
    ac = myresample(double(ac'), 100, EEG.srate)';
end
ac = ac(:, 2:101);

% resample if resample is not present
% -----------------------------------
function tmpeeglab = myresample(data, p, q)

% Default cutoff frequency (pi rad / smp)
fc = 0.9;

% Default transition band width (pi rad / smp)
df = 0.2;

% anti-alias filter
% -----------------
%        data         = eegfiltfft(data', 256, 0, 128*pnts/new_pnts); % Downsample from 256 to 128 times the ratio of freq.
%                                                                      % Code was verified by Andreas Widdman March  2014

%                                                                      % No! Only cutoff frequency for downsampling was confirmed.
%                                                                      % Upsampling doesn't work and FFT filtering introduces artifacts.
%                                                                      % Also see bug 1757. Replaced May 05, 2015, AW

if p < q, nyq = p / q; else nyq = q / p; end
fc = fc * nyq; % Anti-aliasing filter cutoff frequency
df = df * nyq; % Anti-aliasing filter transition band width
m = pop_firwsord('kaiser', 2, df, 0.002); % Anti-aliasing filter kernel
b = firws(m, fc, windows('kaiser', m + 1, 5)); % Anti-aliasing filter kernel
%         figure; freqz(b, 1, 2^14, 1000) % Debugging only! Sampling rate hardcoded as it is unknown in this context. Manually adjust for debugging!

if p < q % Downsampling, anti-aliasing filter
    data = fir_filterdcpadded(b, 1, data, 0);
end

% spline interpolation
% --------------------
%         X            = [1:length(data)];
%         nbnewpoints  = length(data)*p/q;
%         nbnewpoints2 = ceil(nbnewpoints);
%         lastpointval = length(data)/nbnewpoints*nbnewpoints2;
%         XX = linspace( 1, lastpointval, nbnewpoints2);

% New time axis scaling, May 06, 2015, AW
X = 0:length(data) - 1;
newpnts  = ceil(length(data) * p / q);
XX = (0:newpnts - 1) / (p / q);

cs = spline( X, data');
tmpeeglab = ppval(cs, XX)';

if p > q % Upsampling, anti-imaging filter
    tmpeeglab = fir_filterdcpadded(b, 1, tmpeeglab, 0);
end
