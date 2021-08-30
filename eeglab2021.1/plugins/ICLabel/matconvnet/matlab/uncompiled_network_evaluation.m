function net = uncompiled_network_evaluation(net, inputs)

% %  get path information
% pluginpath = fileparts(which('pop_iclabel'));
% 
% % activate matconvnet
% run(fullfile(pluginpath, 'matconvnet', 'matlab', 'vl_setupnn'))
% 
% % load network
% netStruct = load(fullfile(pluginpath, 'netICL.mat'));
% net = dagnn.DagNN.loadobj(netStruct);
% clear netStruct;
% 
% % feed inputs to network
% inputs = {
%     'in_image', single(randn(32, 32, 1, 71)), ...
%     'in_psdmed', single(randn(1, 100, 1, 71))
% };
for it = 1:length(inputs) / 2
    net.vars(net.getVarIndex(inputs{it*2 - 1})).value = inputs{it*2};
end

% evaluate network
for layer_index = net.getLayerExecutionOrder
    % get layer
    layer = net.layers(layer_index);
    % get inputs
    inputs = {net.vars(layer.inputIndexes).value};
    % get parameters
    params = {net.params(layer.paramIndexes).value};
    % calc output
    block = layer.block;
    if block_type(block, 'Conv')
        input = inputs{1};
        % pad input
        input = [zeros(block.pad(1), size(input, 2), size(input, 3), size(input, 4), 'single'); ...
            input; zeros(block.pad(2), size(input, 2), size(input, 3), size(input, 4), 'single')];
        input = [zeros(size(input, 1), block.pad(3), size(input, 3), size(input, 4), 'single') ...
            input zeros(size(input, 1), block.pad(4), size(input, 3), size(input, 4), 'single')];
        
        % determine number of convolutions
        %     YH = floor((H + (PADTOP+PADBOTTOM) - FH)/STRIDEY) + 1,
        %     YW = floor((W + (PADLEFT+PADRIGHT) - FW)/STRIDEX) + 1.
        input_size = size(input);
        param_size = size(params{1});
        if numel(param_size) == 3
            param_size(4) = 1; end
        combined_size = (input_size(1:2) - param_size(1:2));
        output_size = floor(combined_size ./ block.stride) + 1;
        output = zeros([output_size param_size(4) input_size(4)], 'single');
        % convolve by looping over patches
        hind = 1:block.stride(1):combined_size(1)+1;
        vind = 1:block.stride(2):combined_size(2)+1;
        for it1 = 1:length(hind)
            for it2 = 1:length(vind)
                patch = input(hind(it1):hind(it1) + param_size(1) - 1, ...
                              vind(it2):vind(it2) + param_size(2) - 1, :, :);
                patch_size = size(patch);
                output(it1, it2, :, :) = ...
                    sum(reshape(bsxfun(@times, reshape(patch, ...
                    [patch_size(1:3) 1 patch_size(4)]), params{1}), ...
                    [], 1, param_size(4), patch_size(4)));
            end
        end
        % add bias if relevant
        if block.hasBias
            output = bsxfun(@plus, output, ...
                reshape(params{2}, [1 1 length(params{2})]));
        end
    elseif block_type(block, 'ReLU')
        output = max(inputs{1}, inputs{1} * block.leak);
    elseif block_type(block, 'Reshape')
        input_size = size(inputs{1});
        output = reshape(inputs{1}, [block.size input_size(end)]);
    elseif block_type(block, 'Concat')
        output = cat(block.dim, inputs{:});
    elseif block_type(block, 'SoftMax')
        maxval = max(inputs{1}, [], 3);
        output = exp(bsxfun(@minus, inputs{1}, maxval));
        output = bsxfun(@rdivide, output, sum(output, 3));
    end
    % save output
    net.vars(layer.outputIndexes).value = output;
    % delete input
    for it = layer.inputIndexes
        net.vars(it).value = [];
    end
end

function match = block_type(block, cls)
 match = isa(block, ['dagnn.' cls]) || isa(block, ['dagnn_bc.' cls]);
