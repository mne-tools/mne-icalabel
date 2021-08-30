% info
name = 'ICLabel';
filetext = fileread('eegplugin_iclabel.m');
version = regexp(filetext, 'vers = ''ICLabel(\d+\.\d+\.?\d*)'';', 'tokens');
version = version{1}{1};

% enumerate files/directories
files = dir();
file_names = {files.name}';
file_names = file_names(cellfun(@isempty, ...
    regexp(file_names,['^\.|\.md$|~$|^' name '.*\.zip$|zipper.m|tests'])));

% create zip file
zip([name version '.zip'], file_names)