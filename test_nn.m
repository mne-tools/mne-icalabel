clear;

netStruct = load('net_conv1_psds.mat');
% netStruct = load('netICL_conv.mat');
% netStruct = load('/Users/jacobfeitelberg/Desktop/sarma/iclabel-python/eeglab2021.1/plugins/ICLabel/netICL.mat');

net = dagnn.DagNN.loadobj(netStruct);
% load('/Users/jacobfeitelberg/Desktop/sarma/iclabel-python/test_data/full_data.mat');
load('/Users/jacobfeitelberg/Desktop/sarma/iclabel-python/features.mat');
images = features{1};
psds = features{2};
autocorrs = features{3};

images = cat(4, images, -images, images(:, end:-1:1, :, :), -images(:, end:-1:1, :, :));
psds = repmat(psds, [1 1 1 4]);
autocorrs = repmat(autocorrs, [1 1 1 4]);

% input = {'in_image', single(images)};
input = {'in_psdmed', single(psds)};
% input = {'in_autocorr', single(autocorrs)};
% input = {
%     'in_image', single(images), ...
%     'in_psdmed', single(psds), ...
%     'in_autocorr', single(autocorrs)
% };

path2vl_nnconv = which('-all', 'vl_nnconv');
if isempty(findstr('mex', path2vl_nnconv{1})) && length(path2vl_nnconv) > 1
    addpath(fileparts(path2vl_nnconv{2}));
end

net.eval(input);

out = net.getVar(net.getOutputs()).value;

% % Labels
% labels = squeeze(net.getVar(net.getOutputs()).value)';
% labels = reshape(mean(reshape(labels', [], 4), 2), 7, [])';
