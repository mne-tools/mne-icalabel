%% Simple 2-D movie
eeglab; close; % add path
eeglabp = fileparts(which('eeglab.m'));
EEG = pop_loadset(fullfile(eeglabp, 'sample_data', 'eeglab_data_epochs_ica.set'));

% Above, convert latencies in ms to data point indices
pnts1 = round(eeg_lat2point(-100/1000, 1, EEG.srate, [EEG.xmin EEG.xmax]));
pnts2 = round(eeg_lat2point( 600/1000, 1, EEG.srate, [EEG.xmin EEG.xmax]));
scalpERP = mean(EEG.data(:,pnts1:pnts2),3);

% Smooth data
for iChan = 1:size(scalpERP,1)
    scalpERP(iChan,:) = conv(scalpERP(iChan,:) ,ones(1,5)/5, 'same');
end

% 2-D movie
figure; [Movie,Colormap] = eegmovie(scalpERP, EEG.srate, EEG.chanlocs, 'framenum', 'off', 'vert', 0, 'startsec', -0.1, 'topoplotopt', {'numcontour' 0});
seemovie(Movie,-5,Colormap);

% save movie
if isunix && ~ismac
   vidObj = VideoWriter('erpmovie2d.avi', 'Uncompressed AVI');
else
   vidObj = VideoWriter('erpmovie2d.mp4', 'MPEG-4');
end
open(vidObj);
writeVideo(vidObj, Movie);
close(vidObj);

%% Simple 3-D movie
% Use the graphic interface to coregister your head model with your electrode positions
headplotparams1 = { 'meshfile', 'mheadnew.mat'       , 'transform', [0.664455     -3.39403     -14.2521  -0.00241453     0.015519     -1.55584           11      10.1455           12] };
headplotparams2 = { 'meshfile', 'colin27headmesh.mat', 'transform', [0          -13            0          0.1            0        -1.57         11.7         12.5           12] };
headplotparams  = headplotparams1; % switch here between 1 and 2

% set up the spline file
headplot('setup', EEG.chanlocs, 'STUDY_headplot.spl', headplotparams{:}); close
 
% check scalp topo and head topo
figure; headplot(scalpERP(:,end-50), 'STUDY_headplot.spl', headplotparams{:}, 'maplimits', 'absmax', 'lighting', 'on');
figure; topoplot(scalpERP(:,end-50), EEG.chanlocs);
figure('color', 'w'); [Movie,Colormap] = eegmovie( scalpERP, EEG.srate, EEG.chanlocs, 'framenum', 'off', 'vert', 0, 'startsec', -0.1, 'mode', '3d', 'headplotopt', { headplotparams{:}, 'material', 'metal'}, 'camerapath', [-127 2 30 0]); 
seemovie(Movie,-5,Colormap);

% save movie
if isunix && ~ismac
    vidObj = VideoWriter('erpmovie3d1.avi', 'Uncompressed AVI');
else
    vidObj = VideoWriter('erpmovie3d1.mp4', 'MPEG-4');
end
open(vidObj);
writeVideo(vidObj, Movie);
close(vidObj);

%% Using topoplot to make movie frames
if isunix && ~ismac
   vidObj = VideoWriter('erpmovietopoplot.avi', 'Uncompressed AVI');
else
   vidObj = VideoWriter('erpmovietopoplot.mp4', 'MPEG-4');
end
open(vidObj);
counter = 0;
for latency = -100:10:600 %-100 ms to 1000 ms with 10 time steps
    figure; pop_topoplot(EEG,1,latency, 'My movie', [] ,'electrodes', 'off'); % plot'
    currFrame = getframe(gcf);
    writeVideo(vidObj,currFrame);
    close;  % close current figure
end
close(vidObj);
