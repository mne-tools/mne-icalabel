%firgauss() - Gaussian low-pass FIR filter
%
% Usage:
%   >> b = firgauss( fc, fs );
%
% Inputs:
%   fc    - scalar low-pass cutoff frequency (-6 dB)
%   fs    - scalar sampling frequency
%
% Output:
%   b - filter coefficients
%
% Example:
%   fs = 500; fc = 25
%   b  = firgauss( fc, fs ); 
%
% References:
%   http://en.wikipedia.org/wiki/Gaussian_filter
%
% Author: Andreas Widmann, University of Leipzig, 2015
%
% See also:
%   firws

%123456789012345678901234567890123456789012345678901234567890123456789012

% Copyright (C) 2015 Andreas Widmann, University of Leipzig, widmann@uni-leipzig.de
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 2 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
%
% $Id$

function [ b ] = firgauss( fc, fs )

if nargin < 2
    fs = 2;
end

sigf = fc / sqrt( 2 * log( 2 ) );
sigt = fs / ( 2 * pi * sigf );

order = ceil( 3 * sigt ) * 2; % Even for type 1 FIR;

x = -order / 2:order / 2;
b = ( 1 / ( sqrt( 2 * pi ) * sigt ) ) * exp( -x .^ 2 / ( 2 * sigt ^ 2 ) );

end
