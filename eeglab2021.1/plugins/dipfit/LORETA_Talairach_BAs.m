% The file "LORETA-Talairach-BAs.csv" was downloaded from the LORETA-KEY
% website. It was convreted using this script to a Matlab source model file
      
%       Original documentation:
%       The file "LORETA-Talairach-BAs.csv" is plain text (ascii), but suitable for excel.
%       On the first line are the column names, followed by 2394 lines.
%       The header (column names) is:
%       
%       x-min	y-mni	z-mni	x-tal	y-tal	z-tal	dist1	BA1	AnatA1	AnatB1	dist2	BA2	AnatA2	AnatB2	dist3	BA3	AnatA3	AnatB3
%       
%       So, first three numbers are MNI coordinates. Next three are corrected to Talairach coordinates. Then come three groups of 4 numbers each, i.e.:
%       (dist1	BA1	AnatA1	AnatB1)
%       (dist2	BA2	AnatA2	AnatB2)
%       (dist3	BA3	AnatA3	AnatB3)
%       
%       Where:
%       - dist1 is the distance from the loreta voxel to the closest matching talairach labeled point available to me in "TDresult1NearestGM.csv". You can find the file "TDresult1NearestGM.csv" under the folder where you installed the loreta-key software, under subfoler "500-LorSysData". If you change this file, better uninstall and then reinstall the whole loreta package (i.e. don't mess with this file!).
%       - BA1 its Brodmann area
%       - AnatA1 a neuroanatomical label
%       - AnatB1 another neuroanatomical label
%       
%       The next group is for the next best matching point, and the third group for the third best matching point.
%       
%       
%       All this information is based on the Talairach Daemon:
%       http://ric.uthscsa.edu/projects/talairachdaemon.html

% structure needed for Matlab
%     Vertices: [5003×3 double]
%        Faces: [9998×3 double]
%        Atlas: [1×1 struct]
% cortex.Atlas
%      Name: 'Desikan-Kiliany'
%    Scouts: [1×68 struct]
% cortex.Atlas.Scouts(1)
%       Label: 'bankssts L'
%    Vertices: [40×1 double]

tmp = loadtxt( 'LORETA-Talairach-BAs.csv', 'delim', ',');
cortex = [];
cortex.Vertices = [ [ tmp{2:end,1} ]; [ tmp{2:end,2} ];  [ tmp{2:end,3} ] ]';
cortex.Atlas.Name = 'LORETA-Talairach-BAs';
BAareas = unique(tmp(2:end,8));

for iBA = 1:length(BAareas)
    inds = strmatch(BAareas{iBA}, tmp(2:end,8), 'exact');
    
    inds2 = find(cortex.Vertices(inds,1) > 0); % no area has x=0
    cortex.Atlas.Scouts(2*iBA-1).Label = [ BAareas{iBA} 'R' ];
    cortex.Atlas.Scouts(2*iBA-1).Vertices = inds(inds2);
    
    inds2 = find(cortex.Vertices(inds,1) < 0); % no area has x=0
    cortex.Atlas.Scouts(2*iBA).Label = [ BAareas{iBA} 'L' ];
    cortex.Atlas.Scouts(2*iBA).Vertices = inds(inds2);
end
save('-mat', 'LORETA-Talairach-BAs.mat', '-struct', 'cortex');