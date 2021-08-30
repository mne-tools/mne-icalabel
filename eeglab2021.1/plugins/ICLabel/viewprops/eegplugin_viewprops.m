function vers = eegplugin_viewprops( fig, try_strings, catch_strings )
%EEGLABPLUGIN_POP_PROP_EXTENDED View extended channel/component properties
%   Displays ERPimage, spectopo, topomap, activity scroll, dipole, and pvaf
vers = 'ViewProps1.5.4';
if nargin < 3
    error('eegplugin_viewprops requires 3 arguments');
end

plotmenu = findobj(fig, 'tag', 'plot');
uimenu( plotmenu, 'label', 'View extended channel properties', ...
    'callback', [try_strings.no_check 'LASTCOM = pop_viewprops(EEG, 1);' catch_strings.add_to_hist]);
uimenu( plotmenu, 'label', 'View extended component properties', ...
    'callback', [try_strings.no_check 'LASTCOM = pop_viewprops(EEG, 0);' catch_strings.add_to_hist]);
end

