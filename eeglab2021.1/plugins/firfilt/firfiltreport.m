% firfiltreport() - Reporting of filter parameters
%
% Usage:
%   >> firfiltreport('key1', value1, 'key2', value2, 'keyn', valuen);
%
% Inputs:
%   'func'    - string filter function
%   'family'  - string filter family
%   'type'    - string filter type
%   'dir'     - string filter direction/phase
%   'order'   - scalar integer filter order
%
% Optional inputs:
%   'fs'      - scalar sampling frequency
%   'fc'      - scalar or vector cutoff frequency(ies)
%   'df'      - scalar transition band width
%   'pbdev'   - scalar passband deviation
%   'sbatt'   - scalar stopband attenuation
%
% Author: Andreas Widmann, University of Leipzig, 2015
%
% See also:
%   pop_firws

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

function firfiltreport( varargin )

Arg = struct( varargin{ : } );

% Basic reporting
if ~all( isfield( Arg, { 'func', 'type', 'dir', 'order', 'family' } ) )
    error( 'Not enough input arguments.' )
end
if strncmp( Arg.dir, 'twopass', 7 )
    isTwopass = true;
    Arg.order = 2 * Arg.order;
    fcatt = -12;
else
    isTwopass = false;
    fcatt = -6;
end
reportArray{ 1 } = sprintf( '%s() - %s filtering data: %s, order %d, %s\n', Arg.func, Arg.type, Arg.dir, Arg.order, Arg.family );


% Detailed reporting
if all( isfield( Arg, { 'fs', 'fc', 'df', 'pbdev', 'sbatt' } ) ) && ~isempty( Arg.fs ) && ~isempty( Arg.fc ) && ~isempty( Arg.df ) && ~isempty( Arg.pbdev ) && ~isempty( Arg.sbatt )
    
    % Transition band edges
    fn = Arg.fs / 2;
    for iFc = 1:length( Arg.fc )
        dflim( :, iFc ) = [ max( [ Arg.fc(iFc) - Arg.df / 2, 0 ] ), min( [ Arg.fc(iFc) + Arg.df / 2, fn ] ) ]; %#ok<AGROW>
    end

    switch Arg.type
        case 'lowpass'
            reportArray{ 2 } = sprintf( '  cutoff (%d dB) %g Hz\n', fcatt, Arg.fc );
            reportArray{ 3 } = sprintf( '  transition width %.1f Hz, passband 0-%.1f Hz, stopband %.1f-%.0f Hz\n', Arg.df, dflim( : ), fn );
        case 'highpass'
            reportArray{ 2 } = sprintf( '  cutoff (%d dB) %g Hz\n', fcatt, Arg.fc );
            reportArray{ 3 } = sprintf( '  transition width %.1f Hz, stopband 0-%.1f Hz, passband %.1f-%.0f Hz\n', Arg.df, dflim( : ), fn );
        case 'bandpass'
            reportArray{ 2 } = sprintf( '  cutoff (%d dB) %g Hz and %g Hz\n', fcatt, Arg.fc );
            reportArray{ 3 } = sprintf( '  transition width %.1f Hz, stopband 0-%.1f Hz, passband %.1f-%.1f Hz, stopband %.1f-%.0f Hz\n', Arg.df, dflim( : ), fn );
        case 'bandstop'
            reportArray{ 2 } = sprintf( '  cutoff (%d dB) %g Hz and %g Hz\n', fcatt, Arg.fc );
            reportArray{ 3 } = sprintf( '  transition width %.1f Hz, passband 0-%.1f Hz, stopband %.1f-%.1f Hz, passband %.1f-%.0f Hz\n', Arg.df, dflim( : ), fn );
    end

    if isTwopass % Adjust deviation/ripple for twopass filtering
        Arg.pbdev = ( Arg.pbdev + 1 ) ^ 2 - 1;
        Arg.sbatt = Arg.sbatt ^ 2;
    end
    reportArray{ 4 } = sprintf( '  max. passband deviation %.4f (%.2f%%), stopband attenuation %.0f dB\n', Arg.pbdev, Arg.pbdev * 100, 20 * log10( Arg.sbatt ) );

end

% Fieldtrip or EEGLAB
if strncmp( 'ft_', Arg.func, 3 )
    for iLine = 1:length( reportArray )
        print_once( reportArray{ iLine } );
    end
else
    for iLine = 1:length( reportArray )
        fprintf( strrep( reportArray{ iLine }, '%', '%%' ) );
    end
end

end

