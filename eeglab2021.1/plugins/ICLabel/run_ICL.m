function labels = run_ICL(version, images, psds, autocorrs)
%% check inputs
if any(strcmpi(version, {[], 'default'}))
    version = '';
elseif any(strcmpi(version, 'lite'))
    version = '_lite';
elseif any(strcmpi(version, 'beta'))
    version = '_beta';
else
    error(['Invalid network version choice. '...
           'Must be one of the following: ' ...
           '''default'' (alternatively ''[] or  ''''), '...
           '''lite'', or ''beta''.'])
end

if ~exist('autocorrs', 'var') || isempty(autocorrs)
    flag_autocorr = false;
else
    flag_autocorr = true;
end

%% get path information and activate matconvnet
pluginpath = activate_matconvnet();

%% load network
netStruct = load(fullfile(pluginpath, ['netICL' version]));
try
    net = dagnn.DagNN.loadobj(netStruct);
catch
    net = dagnn_bc.DagNN.loadobj(netStruct);
end
clear netStruct;

%% format network inputs
images = cat(4, images, -images, images(:, end:-1:1, :, :), -images(:, end:-1:1, :, :));
psds = repmat(psds, [1 1 1 4]);
input = {
    'in_image', single(images), ...
    'in_psdmed', single(psds)
};
if flag_autocorr
    autocorrs = repmat(autocorrs, [1 1 1 4]);
    input = [input {'in_autocorr', single(autocorrs)}];
end

% check path (sometimes mex file not first which create a problem)
path2vl_nnconv = which('-all', 'vl_nnconv');
if isempty(findstr('mex', path2vl_nnconv{1})) && length(path2vl_nnconv) > 1
    addpath(fileparts(path2vl_nnconv{2}));
end

%% inference
try
    % run with mex-files
    net.eval(input);
catch
    % failed, try to recompile mex-files
    disp 'Failed to run ICLabel. Trying to compile MEX-files.'
    curr_path = pwd;
    cd(fileparts(which('vl_compilenn')));
    try
        vl_compilenn
        cd(curr_path)
        disp(['MEX-files successfully compiled. Attempting to run ICLabel again. ' ...
            'Please consider emailing Luca Pion-Tonachini at lpionton@ucsd.edu to ' ...
            'share the compiled MEX-files. They will likely help other EEGLAB users ' ...
            'with similar computers as yourself.'])
        net.eval(input);
    catch
        % could not recompile. running natively
        % ~80x slower than using mex-files
        cd(curr_path)
        disp(['MEX-file compilation failed. Further instructions on compiling ' ...
              'the MEX-files can be found at http://www.vlfeat.org/matconvnet/install/. ' ...
              'Further, you may contact Luca Pion-Tonachini at lpionton@ucsd.edu for help. ' ...
              'If you solve this issue without help, please consider emailing Luca as the ' ...
              'compiled files will likely be useful to other EEGLAB users with similar ' ...
              'computers as yourself.'])
        warning('ICLabel: defaulting to uncompiled matlab code (about 80x slower)')
        net = uncompiled_network_evaluation(net, input);
    end
end

%% extract result
labels = squeeze(net.getVar(net.getOutputs()).value)';
labels = reshape(mean(reshape(labels', [], 4), 2), 7, [])';
