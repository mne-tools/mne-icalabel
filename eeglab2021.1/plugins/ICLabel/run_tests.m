% test the ICLabel EEGLAB plugin
function run_tests()

%% set up
% load test data
EEG = pop_loadset('tests/eeglab_data.set');

% calculate sphereing matrix
try
    icasphere = pca(EEG.data')';
catch
    % BCILAB (unfortunately) shadows pca from the stats toolbox.
    flist = which('pca.m', '-all');
    fpath = flist{find(strncmp(flist, matlabroot, length(matlabroot)), 1)};
    cwd = pwd;
    cd(fileparts(fpath))
    icasphere = pca(EEG.data')';
    cd(cwd)
end

% make sample ica matrix and test legacy rng
w = warning;
warning('off', 'MATLAB:RandStream:ActivatingLegacyGenerators')
rand('state', 11)
warning(w);
icaweights = randn(size(EEG.data, 1));
icaweights = bsxfun(@rdivide, icaweights, sqrt(sum(icaweights.^2)));

%% begin testing: first unepoched then epoched

for it = 1:2
    
    % epoch on second run through
    if it == 2
        EEG = pop_epoch(EEG, {'square'}, [-1 1]); 
    end

    %% test complete ICA with data as singles
    
    EEG_temp = setup(EEG, icasphere, icaweights, 1:32);
    EEG_temp.data = single(EEG_temp.data);
    EEG_temp.icaact = single(EEG_temp.icaact);
    
    test_iclabel(EEG_temp)
    

    %% test ICA with channels removed (A)
    
    remove = [5 10 15 20 25 30];
    icachansind = setdiff(1:32, remove);
    icasphere_temp = icasphere(icachansind, icachansind);
    icaweights_temp = icaweights(icachansind, icachansind);
    EEG_temp = setup(EEG, icasphere_temp, icaweights_temp, icachansind);
    
    test_iclabel(EEG_temp)
    

    %% test ICA with components removed (B)
    
    EEG_temp = setup(EEG, icasphere, icaweights, 1:32);
    remove = [5 10 15 20 25 30];
    EEG_temp = pop_subcomp(EEG_temp, remove);
    EEG_temp.icaact = eeg_getica(EEG_temp);
    
    test_iclabel(EEG_temp)

    
    %% test PCA-reduced ICA (C)
    
    EEG_temp = setup(EEG, icasphere(1:20, :), icaweights(1:20, 1:20), 1:32);
    
    test_iclabel(EEG_temp)
    

    %% test ABC
    
    remove = [5 10 15 20 25 30];
    icachansind = setdiff(1:32, remove);
    pca_subset = 1:20;
    icasphere_temp = icasphere(pca_subset, icachansind);
    icaweights_temp = icaweights(pca_subset, pca_subset);
    EEG_temp = setup(EEG, icasphere_temp, icaweights_temp, icachansind);
    remove = [5 10 15];
    EEG_temp = pop_subcomp(EEG_temp, remove);
    EEG_temp.icaact = eeg_getica(EEG_temp);
    
    test_iclabel(EEG_temp)
    
    
    %% test ABC also missing IC activations (D)
    
    EEG_temp.icaact = [];
    
    test_iclabel(EEG_temp)
   
    
end


    % testing for a single run of iclabel
    function test_iclabel(EEG)
        
    nic = size(EEG.icaweights, 1);
    
    % test all three versions
    for version = {'default', 'lite', 'beta'}
        out = pop_iclabel(EEG, version);
        assert(all(size(out.etc.ic_classification.ICLabel.classifications) == [nic, 7]))
        assert(strcmp(out.etc.ic_classification.ICLabel.version, version))
    end
    
    
    % prepare a dataset for testing
    function EEG = setup(EEG, icasphere, icaweights, icachansind)
    EEG.icasphere = icasphere;
    EEG.icaweights = icaweights;
    EEG.icawinv = pinv(EEG.icaweights * EEG.icasphere);
    EEG.icachansind = icachansind;
    EEG.icaact = eeg_getica(EEG);
        
    
    
    
    
    
    
    