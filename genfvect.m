function [vec_feature, vec_label] = genfvect(EEG, study_ID, dataset_ID, ICnum)
% check input
if ~exist('EEG', 'var') || isempty(EEG);
    study_ID = 1;
    dataset_ID = 1;
    ICnum = 1;
    EEG = gen_eeg;
end

% Getting a couple more features
dip_single = [EEG.dipfit.model(ICnum).posxyz EEG.dipfit.model(ICnum).momxyz];
dip_double1 = [EEG.dipfit2.model(ICnum).posxyz(1, :) EEG.dipfit2.model(ICnum).momxyz(1, :)];
dip_double2 = [EEG.dipfit2.model(ICnum).posxyz(2, :) EEG.dipfit2.model(ICnum).momxyz(2, :)];

vec_feature = [];
vec_label = {};
index = 0;

% book keeping
update_fvec([study_ID, dataset_ID, ICnum], {'Study ID', 'Dataset ID', 'IC#'}, 3)

% scalp topography
update_fvec(EEG.reject.SASICA.topo(ICnum, :), 'topo image', 740)

% SASICA
update_fvec(EEG.reject.SASICA.icaautocorr(ICnum),'SASICA autocorrelation', 1)
update_fvec(EEG.reject.SASICA.icafocalcomp(ICnum),'SASICA focal topo', 1)
update_fvec(EEG.reject.SASICA.icaSNR(ICnum),'SASICA snr', 1)
update_fvec(EEG.reject.SASICA.var(ICnum),'SASICA ic variance', 1)

% ADJUST
update_fvec(EEG.reject.SASICA.icaADJUST.diff_var(ICnum),'ADJUST diff_var', 1)
update_fvec(EEG.reject.SASICA.icaADJUST.meanK(ICnum),'ADJUST Temporal Kurtosis', 1)
update_fvec(EEG.reject.SASICA.icaADJUST.SED(ICnum),'ADJUST Spatial Eye Difference', 1)
update_fvec(EEG.reject.SASICA.icaADJUST.SAD(ICnum),'ADJUST spatial average difference', 1)
update_fvec(EEG.reject.SASICA.icaADJUST.GDSF(ICnum),'ADJUST General Discontinuity Spatial Feature', 1)
update_fvec(EEG.reject.SASICA.icaADJUST.nuovaV(ICnum),'ADJUST maxvar/meanvar', 1) % should remove

% FASTER
% 1 Median gradient value, for high frequency stuff
% 2 (NOT USED) Mean slope around the LPF band (spectral)
% 3 Kurtosis of spatial map
% 4 Hurst exponent
% 5 (NOT USED) Eyeblink correlations
label = {'FASTER Median gradient value','FASTER Kurtosis of spatial map','FASTER Hurst exponent'};
update_fvec(EEG.reject.SASICA.icaFASTER.listprops(ICnum,[1 3 4]),label, 3)


% basic info
update_fvec(EEG.nbchan,'number of channels', 1)
nic = size(EEG.icaact,1);
update_fvec(nic,'number of ICs', 1)
update_fvec(EEG.reject.SASICA.plotrad(ICnum),'topoplot plot radius', 1)
update_fvec(EEG.trials>1,'epoched?', 1)
update_fvec(EEG.srate,'sampling rate', 1)
update_fvec(EEG.pnts,'number of data points', 1)

% dipole info
update_fvec(EEG.dipfit.model(ICnum).rv,'dip1 rv (SASICA)', 1)
label = {'dip1 posx', 'dip1 posy', 'dip1 posz', 'dip1 momx', 'dip1 momy', 'dip1 momz'};
update_fvec(dip_single,label, 6)
update_fvec(EEG.dipfit2.model(ICnum).rv,'dip2 rv', 1)
label = {'dip2_1 posx', 'dip2_1 posy', 'dip2_1 posz', 'dip2_1 momx', 'dip2_1 momy', 'dip2_1 momz'};
update_fvec(dip_double1,label, 6)
label = {'dip2_2 posx', 'dip2_2 posy', 'dip2_2 posz', 'dip2_2 momx', 'dip2_2 momy', 'dip2_2 momz'};
update_fvec(dip_double2,label, 6)

% spectrum info
update_fvec(EEG.reject.SASICA.spec(ICnum, :), 'psd_med', 100)
update_fvec(EEG.reject.SASICA.specvar(ICnum, :), 'psd_var', 100)
update_fvec(EEG.reject.SASICA.speckrt(ICnum, :), 'psd_kurt', 100)

% autocorrelation function
update_fvec(EEG.reject.SASICA.autocorr(ICnum, :), 'autocorr', 100)


function update_fvec(val, label, len)
    vec_feature(index + (1:len)) = val;
    if iscellstr(label)
        vec_label(index + (1:len)) = label;
    elseif ischar(label)
        [vec_label{index + (1:len)}] = deal(label);
    end
    index = index + len;
end

end

function EEG = gen_eeg
EEG = eeg_emptyset;

EEG.reject.SASICA.icaresvar = 1;
EEG.reject.SASICA.icaautocorr = 1;
EEG.reject.SASICA.icafocalcomp = 1;
EEG.reject.SASICA.icaSNR = 1;
EEG.reject.SASICA.var = 1;
EEG.reject.SASICA.icaADJUST.diff_var = 1;
EEG.reject.SASICA.icaADJUST.meanK = 1;
EEG.reject.SASICA.icaADJUST.SED = 1;
EEG.reject.SASICA.icaADJUST.SAD = 1;
EEG.reject.SASICA.icaADJUST.GDSF = 1;
EEG.reject.SASICA.icaADJUST.nuovaV = 1;

EEG.dipfit.model.rv = 0;
EEG.dipfit.model.posxyz = zeros(1,3);
EEG.dipfit.model.momxyz = zeros(1,3);
EEG.dipfit2.model.rv = 0;
EEG.dipfit2.model.posxyz(1,:) = zeros(1,3);
EEG.dipfit2.model.momxyz(1,:) = zeros(1,3);
EEG.dipfit2.model.posxyz(2,:) = zeros(1,3);
EEG.dipfit2.model.momxyz(2,:) = zeros(1,3);

EEG.reject.SASICA.topo = zeros(1,740);
EEG.reject.SASICA.spec = zeros(1,100);
EEG.reject.SASICA.specvar = zeros(1,100);
EEG.reject.SASICA.speckrt = zeros(1,100);
EEG.reject.SASICA.autocorr = zeros(1,100);
EEG.reject.SASICA.icaFASTER.listprops = zeros(1,5);

EEG.nbchan = 1;
EEG.icaact = ones(2);
EEG.reject.SASICA.plotrad = 1;
EEG.pnts = 1;
end