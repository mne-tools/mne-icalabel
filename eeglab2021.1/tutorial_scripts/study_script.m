% check folder
eeglab
if ~exist('n400.study', 'file')
    error([ 'You must change the path to the folder containing the data to run this script' 10 ...
        'Download the data from https://eeglab.org/tutorials/tutorial_data.html (5 subject study)' ]);
else
    filepath = fileparts(which('n400.study'));
end

%% import data and create study
[STUDY, ALLEEG] = std_editset( [], [], 'name','N400STUDY',...
        'task', 'Auditory task: Synonyms Vs. Non-synonyms, N400',...
        'filename', 'N400empty.study','filepath', './',...
        'commands', { ...
        { 'index' 1 'load' fullfile(filepath, 's02','syn02-s253-clean.set') 'subject' 'S02' 'condition' 'synonyms' }, ...
        { 'index' 2 'load' fullfile(filepath, 's05', 'syn05-s253-clean.set') 'subject' 'S05' 'condition' 'synonyms' }, ...
        { 'index' 3 'load' fullfile(filepath, 's07', 'syn07-s253-clean.set') 'subject' 'S07' 'condition' 'synonyms' }, ...
        { 'index' 4 'load' fullfile(filepath, 's08', 'syn08-s253-clean.set') 'subject' 'S08' 'condition' 'synonyms' }, ...
        { 'index' 5 'load' fullfile(filepath, 's10', 'syn10-s253-clean.set') 'subject' 'S10' 'condition' 'synonyms' }, ...
        { 'index' 6 'load' fullfile(filepath, 's02', 'syn02-s254-clean.set') 'subject' 'S02' 'condition' 'non-synonyms' }, ...
        { 'index' 7 'load' fullfile(filepath, 's05', 'syn05-s254-clean.set') 'subject' 'S05' 'condition' 'non-synonyms' }, ...
        { 'index' 8 'load' fullfile(filepath, 's07', 'syn07-s254-clean.set') 'subject' 'S07' 'condition' 'non-synonyms' }, ...
        { 'index' 9 'load' fullfile(filepath, 's08', 'syn08-s254-clean.set') 'subject' 'S08' 'condition' 'non-synonyms' }, ...
        { 'index' 10 'load' fullfile(filepath, 's10', 'syn10-s254-clean.set') 'subject' 'S10' 'condition' 'non-synonyms' }, ...
    { 'dipselect' 0.15 } });

%% update graphical interface
CURRENTSTUDY = 1; EEG = ALLEEG;
eeglab redraw;

%% Computing and plotting channel measures
[STUDY, ALLEEG] = std_precomp(STUDY, ALLEEG, 'channels', 'erp', 'on', 'erpparams', {'rmbase' [-200 0]});

%% plotting components
STUDY = std_erpplot(STUDY, ALLEEG, 'channels', {'Oz'});
[STUDY, erpdata, erptimes] = std_erpplot(STUDY, ALLEEG, 'channels', {'Oz'}, 'timerange', [-200 1000]);
std_plotcurve(erptimes, erpdata, 'plotconditions', 'together', 'plotstderr', 'on', 'figure', 'on', 'filter', 30);

%% Retreiving measures
STUDY = std_erpplot(STUDY,ALLEEG,'channels',{ 'FP1'});
[STUDY, erpdata, erptimes] = std_erpplot(STUDY, ALLEEG, 'channels', {'Oz'}, 'timerange', [-200 1000]);
std_plotcurve(erptimes, erpdata, 'plotconditions', 'together', 'plotstderr', 'on', 'figure', 'on');

%% Getting command output
STUDY = std_erpplot(STUDY,ALLEEG,'channels',{ 'FP1'});
[STUDY, erpdata, erptimes] = std_erpplot(STUDY,ALLEEG,'channels',{ 'FP1'}, 'noplot', 'on');
figure; plot(erptimes, erpdata{2});

%% Comparing newtimef output in single dataset and study
% Compute newtimef on first dataset for channel 1
options = {'freqscale', 'linear', 'freqs', [3 25], 'nfreqs', 20, 'ntimesout', 60, 'padratio', 1,'winsize',64,'baseline', 0};
TMPEEG = eeg_checkset(ALLEEG(1), 'loaddata');
figure; X = pop_newtimef( TMPEEG, 1, 1, [TMPEEG.xmin TMPEEG.xmax]*1000, [3 0.8] , 'topovec', 1, 'elocs', TMPEEG.chanlocs, 'chaninfo', TMPEEG.chaninfo, 'plotphase', 'off', options{:},'title',TMPEEG.setname, 'erspmax ',6.6);

% Compute newtimef for all datasets and plot first channel of first dataset
[STUDY, ALLEEG] = std_precomp(STUDY, ALLEEG, 'channels','recompute','on','ersp','on','erspparams',{'cycles' [3 0.8] 'parallel' 'on' options{:} },'itc','on');
STUDY = std_erspplot(STUDY,ALLEEG,'channels',{TMPEEG.chanlocs(1).labels}, 'subject', 'S02', 'design', 1 );

%% Computing component measures
[STUDY, ALLEEG] = std_precomp(STUDY, ALLEEG, 'components',...
    'erp','on','erpparams',{'rmbase' [-200 0] },...
    'scalp','on',...
    'spec','on','specparams',{'freqrange' [3 50] 'specmode' 'fft' 'logtrials' 'off'},...
    'ersp','on','erspparams',{'cycles' [3 0.8] 'nfreqs' 20 'ntimesout' 60},...
    'itc','on');

%% Cluster components
[STUDY, ALLEEG] = std_preclust(STUDY, ALLEEG, 1,...
        {'spec' 'npca' 10 'weight' 1 'freqrange' [3 25] },...
        {'erp' 'npca' 10 'weight' 1 'timewindow' [100 600]  'erpfilter' '20'},...
        {'dipoles' 'weight' 10},...
        {'ersp' 'npca' 10 'freqrange' [3 25]  'timewindow' [-1600 1495]  'weight' 1 'norm' 1 'weight' 1});
[STUDY] = pop_clust(STUDY, ALLEEG, 'algorithm','kmeanscluster', 'clus_num', 10);

%% Plot component clusters
[STUDY] = pop_clust(STUDY, ALLEEG, 'algorithm','kmeanscluster', 'clus_num', 10);
[STUDY] = std_topoplot(STUDY, ALLEEG, 'clusters', 2, 'mode', 'together');
[STUDY] = std_topoplot(STUDY, ALLEEG, 'clusters', 2, 'mode', 'apart');
[STUDY] = std_topoplot(STUDY, ALLEEG, 'clusters', 2, 'comps', 1);

%% Plot statistics
STUDY = pop_statparams(STUDY, 'condstats', 'on');
[STUDY, erpdata, erptimes, pgroup, pcond, pinter] = std_erpplot(STUDY,ALLEEG,'channels',{ 'FP1'});
[STUDY, erpdata, erptimes, pgroup, pcond, pinter] = std_erpplot(STUDY,ALLEEG,'clusters', 1);

%% Custom calculation
std_precomp(STUDY, ALLEEG, 'channels', 'customfunc', @(data)bsxfun(@minus, data, mean(data(:,1:410,:),2)), 'interp', 'on');
std_precomp(STUDY, ALLEEG, 'channels', 'customfunc', @(data)reshape(eegfilt(data(:,:), EEG(1).srate, 0,10,EEG(1).pnts,60,0,'fir1'), size(data)), 'interp', 'on');

[~, customdata] = std_readdata(STUDY, ALLEEG, 'channels', {ALLEEG(1).chanlocs.labels }, 'design', 1, 'datatype', 'custom');
[~, erpdata] = std_readdata(STUDY, ALLEEG, 'channels', {ALLEEG(1).chanlocs.labels }, 'design', 1, 'datatype', 'erp');

%% Custom plotting
std_plotcurve(EEG(1).times, erpdata, 'chanlocs', ALLEEG(1).chanlocs);
customdata = cellfun(@squeeze, customdata, 'uniformoutput', false);
std_plotcurve(EEG(1).times, customdata, 'chanlocs', ALLEEG(1).chanlocs);

figure;
nCond = length(STUDY.design.variable(1).value);
for iCond = 1:nCond
  rms = sqrt(mean(mean(customdata{iCond},3).^2,2));
  hold on; plot(EEG(1).times, rms);
end
legend(STUDY.design.variable(1).value)
setfont(gcf, 'fontsize', 16); % change font size

%% Custom statistics
std_stat(erpdata, 'condstats', 'on', 'mcorrect', 'fdr', 'method', 'permutation')
std_stat(erpdata, 'condstats', 'on', 'fieldtripmcorrect', 'cluster', 'fieldtripmethod', 'montecarlo', 'mode', 'fieldtrip')
res = statcond(erpdata); size(res)

res = statcond(erpdata); size(res)
