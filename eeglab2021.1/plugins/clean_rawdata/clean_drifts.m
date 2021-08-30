function signal = clean_drifts(signal,transition,attenuation)
% Removes drifts from the data using a forward-backward high-pass filter.
% Signal = clean_drifts(Signal,Transition)
%
% This removes drifts from the data using a forward-backward (non-causal) filter.
% NOTE: If you are doing directed information flow analysis, do no use this filter but some other one.
%
% In:
%   Signal : the continuous data to filter
%
%   Transition : the transition band in Hz, i.e. lower and upper edge of the transition
%                (default: [0.5 1])
%
%   Attenuation : stop-band attenuation, in db (default: 80)
%
% Out:
%   Signal : the filtered signal
%
% Notes:
%   This function requires the Signal Processing toolbox.
%
%                                Christian Kothe, Swartz Center for Computational Neuroscience, UCSD
%                                2012-09-01

% Copyright (C) Christian Kothe, SCCN, 2012, ckothe@ucsd.edu
%
% This program is free software; you can redistribute it and/or modify it under the terms of the GNU
% General Public License as published by the Free Software Foundation; either version 2 of the
% License, or (at your option) any later version.
%
% This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
% even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
% General Public License for more details.
%
% You should have received a copy of the GNU General Public License along with this program; if not,
% write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307
% USA

if ~exist('transition','var') || isempty(transition) transition = [0.5 1]; end
if ~exist('attenuation','var') || isempty(attenuation) attenuation = 80; end
signal.data = double(signal.data);

% design highpass FIR filter
transition = 2*transition/signal.srate;
wnd = design_kaiser(transition(1),transition(2),attenuation,true);
B = design_fir(length(wnd)-1,[0 transition 1],[0 0 1 1],[],wnd);

% apply it, channel by channel to save memory
signal.data = signal.data';
for c=1:signal.nbchan
    signal.data(:,c) = filtfilt_fast(B,1,signal.data(:,c)); end
signal.data = signal.data';
signal.etc.clean_drifts_kernel = B;
