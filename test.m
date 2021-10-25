if isempty(EEG.icaact)
    EEG.icaact = eeg_getica(EEG);
end
EEG.icaact = double(EEG.icaact);

[~, ~, Th, Rd, ~] = readlocs(EEG.chanlocs(EEG.icachansind));

icaact = EEG.icaact;
icaweights = EEG.icaweights;
srate = EEG.srate;
pnts = EEG.pnts;
icawinv = EEG.icawinv;
trials = EEG.trials;

ncomp = size(EEG.icawinv, 2);

topo = zeros(32, 32, 1, ncomp);
for it = 1:ncomp
    if ~exist('OCTAVE_VERSION', 'builtin') 
        [~, temp_topo, ~] = ...
            topoplotFast(EEG.icawinv(:, it), EEG.chanlocs(EEG.icachansind), ...
            'noplot', 'on');
    else
        [~, temp_topo, ~] = ...
            topoplot(EEG.icawinv(:, it), EEG.chanlocs(EEG.icachansind), ...
            'noplot', 'on', 'gridscale', 32);
    end
    temp_topo(isnan(temp_topo)) = 0;
    topo(:, :, 1, it) = temp_topo / max(abs(temp_topo(:)));
end

plotchans = [1 3 4 5 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32];

% %% calc psd
% psd = eeg_rpsd(EEG, 100);
% subset = psd;
% 
% % extrapolate or prune as needed
% nfreq = size(psd, 2);
% if nfreq < 100
%     psd = [psd, repmat(psd(:, end), 1, 100 - nfreq)];
% end
% 
% % undo notch filter
% for linenoise_ind = [50, 60]
%     linenoise_around = [linenoise_ind - 1, linenoise_ind + 1];
%     difference = bsxfun(@minus, psd(:, linenoise_around), ...
%         psd(:, linenoise_ind));
%     notch_ind = all(difference > 5, 2);
%     if any(notch_ind)
%         psd(notch_ind, linenoise_ind) = mean(psd(notch_ind, linenoise_around), 2);
%     end
% end
% 
% psd = bsxfun(@rdivide, psd, max(abs(psd), [], 2));

% psd = single(permute(psd, [3 2 4 1]));

features = ICL_feature_extractor(EEG, true);

size(features{2})

save test_data/full_data.mat topo icawinv Rd Th plotchans subset icaact icaweights psd trials pnts srate features


