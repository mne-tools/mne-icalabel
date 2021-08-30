function y = vl_nnreshape(inputs, inSize, dzdy, varargin)
%VL_NNCONCAT CNN concatenate multiple inputs.
%  Y = VL_NNCONCAT(INPUTS, DIM) concatenates the inputs in the cell
%  array INPUTS along dimension DIM generating an output Y.
%
%  DZDINPUTS = VL_NNCONCAT(INPUTS, DIM, DZDY) computes the derivatives
%  of the block projected onto DZDY. DZDINPUTS has one element for
%  each element of INPUTS, each of which is an array that has the same
%  dimensions of the corresponding array in INPUTS.

% Copyright (C) 2015 Karel Lenc and Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

% opts.inputSizes = [] ;
% opts = vl_argparse(opts, varargin, 'nonrecursive') ;

% if nargin < 2, reshape = 3; end;
if nargin < 3, dzdy = []; end;

if isempty(dzdy)
  y = reshape(inputs, inSize(1), inSize(2), inSize(3), []); %vectorize
else
  y = reshape(dzdy,1,1, prod(inSize), []);
end