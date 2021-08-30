function pluginpath = activate_matconvnet()

% get path information
pluginpath = fileparts(which('pop_iclabel'));

% activate matconvnet
folder = fullfile(pluginpath, 'matconvnet', 'matlab', 'mex');
path_cell = regexp(path, pathsep, 'split');
if ispc  % Windows is not case-sensitive
  flag = ~any(strcmpi(folder, path_cell));
else
  flag = ~any(strcmp(folder, path_cell));
end
if flag
    run(fullfile(pluginpath, 'matconvnet', 'matlab', 'vl_setupnn'))
end