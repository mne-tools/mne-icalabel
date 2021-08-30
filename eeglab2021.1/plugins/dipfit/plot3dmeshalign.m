% plot3dmeshalign - align a first meshes or volume with a second mesh
%                   to check alignment.
%
% Usage:
%  pop_roi_connectplot(file1, file2, transform, color1, color2, region);
%
% Inputs:
%  file1     - [string] name of the first mesh/volume
%  file2     - [string] name of the second mesh
%  transform - [real array] homogenous transformation matrix to put mesh2
%              into the same space as mesh1/volume1
%  color1    - color for mesh1
%  color2    - color for mesh2
%  region    - region to load (volume data only)
%
% Author: Arnaud Delorme, 2020

% Copyright (C) Arnaud Delorme, arnodelorme@gmail.com
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
%
% 1. Redistributions of source code must retain the above copyright notice,
% this list of conditions and the following disclaimer.
%
% 2. Redistributions in binary form must reproduce the above copyright notice,
% this list of conditions and the following disclaimer in the documentation
% and/or other materials provided with the distribution.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
% THE POSSIBILITY OF SUCH DAMAGE.

function plot3dmeshalign(filename1, filename2, transform, color1, color2, region)

if isempty(filename1)
    return;
end
if nargin < 3
    transform = [];
end
if nargin < 4
    color1  = [1 1 1]*0.5;
end
if nargin < 5
    color2  = [1 0 0];
end
if nargin < 6
    region = [];
end

fig = figure;
ax = axes('unit', 'normalized', 'position', [ 0.05 0.05 0.9 0.9]);
[pos1,tri1] = loadmeshdata(filename1);
[pos2,tri2] = loadmeshdata(filename2, transform);
if isempty(pos2)
    [~,~,ext] = fileparts(filename2);
    if ~strcmpi(ext, '.dip')
        pos2 = loadvolumedata(filename2, transform, region);
        title('Vertices falling outside the brain mesh will be automatically removed');
    else
        pos2 = importdata(filename2);
        pos2 = pos2(1:end/3,1:3);
    end
else
    title('Vertices falling outside the brain mesh will automatically be moved inside the mesh');
end
axis off; 
try, ax.Toolbar.Visible = 'off'; catch, end

ratio = abs( max(reshape(pos1, numel(pos1), 1))/max(reshape(pos2, numel(pos2), 1)) );
if ratio > 100 || ratio < 0.01
    disp('Warning: widely different scale, one of the mesh might not be visible');
end

pairwiseDist = ones(size(pos1,1),4);
colors = pairwiseDist(:,4)*color1;

axes('unit', 'normalized', 'position', [ 0.05 0.05 0.9 0.9]);
patch('Faces',tri1,'Vertices',pos1, 'FaceVertexCdata',colors,'facecolor','interp','edgecolor','none', 'facealpha', 0.5);
hold on
if ~isempty(tri2)
    pairwiseDist = ones(size(pos2,1),4);
    colors = pairwiseDist(:,4)*color2;
    patch('Faces',tri2,'Vertices',pos2, 'FaceVertexCdata',colors,'facecolor','interp','edgecolor','none', 'facealpha', 0.5);
else
    h = plot3(pos2(:,1),pos2(:,2),pos2(:,3), '.', 'color', color2);
end

axisBrain = gca;
axis equal;
axis off;
hold on;

% change lighting
set(fig, 'renderer', 'opengl');
lighting(axisBrain, 'phong');
hlights = findobj(axisBrain,'type','light');
delete(hlights)
hlights = [];
camlight(0,0);
camlight(90,0);
camlight(180,0);
camlight(270,0);
camproj orthographic
axis vis3d
camzoom(1);
hlegend = legend({'Head model' 'ROI source model' });
set(hlegend, 'position', [0.7473 0.7935 0.2304 0.0774]);
                    
% ----------------------------------------
% function to load mesh and transform mesh
% ----------------------------------------
function [pos,tri] = loadmeshdata(filename, transform)

pos = [];
tri = [];
if ischar(filename)
    try
        f = load('-mat', filename);
    catch
        return;
    end
else
    f = filename;
end

if isfield(f, 'cortex')
    f = f.cortex;
end
if isfield(f, 'SurfaceFile') % Brainstrom leadfield
    p = fileparts(fileparts(fileparts(filename)));
    try
        f = load('-mat', fullfile(p, 'anat', f.SurfaceFile));
    catch
        error('Cannot find Brainstorm mesh file')
    end
end
if isfield(f, 'pos')
    pos = f.pos;
    tri = f.tri;
elseif isfield(f, 'Vertices') && isfield(f, 'Faces') 
    pos = f.Vertices;
    tri = f.Faces;
elseif isfield(f, 'Vertices') && ~isfield(f, 'Faces') 
    pos = f.Vertices;
elseif isfield(f, 'vertices') && isfield(f, 'faces') 
    pos = f.vertices;
    tri = f.faces;
elseif isfield(f, 'vol')
    pos = f.vol.bnd(3).pnt;
    tri = f.vol.bnd(3).tri;
else
    return
end

if size(pos,1) == 3
    pos = pos';
end
if nargin > 1 && ~isempty(transform)
    pos = traditionaldipfit(transform)*[pos ones(size(pos,1),1)]';
    pos(4,:) = [];
    pos = pos';
end

% ---------------------
% function to load mesh
% ---------------------
function [pos] = loadvolumedata(filename, transform, regions)

if ischar(filename)
    atlas = ft_read_atlas(filename);
else
    atlas = filename;
end

if isfield(atlas, 'tissue')
    if isempty(regions)
        mri = sum(atlas.tissue(:,:,:,:),4) > 0;
    else
        for iRegion = 1:length(regions)
            if iRegion == 1
                mri = sum(atlas.tissue(:,:,:,:),4) == regions(iRegion);
            else
                mri = mri | (sum(atlas.tissue(:,:,:,:),4) == regions(iRegion));
            end
        end
    end
elseif isfield(atlas, 'brick0')
    if isempty(regions)
        mri = sum(atlas.brick0(:,:,:,:),4) > 0;
    else
        for iRegion = 1:length(regions)
            if isnumeric(regions)
                indRegion = regions(iRegion);
            else
                indRegion = strmatch(regions{iRegion}, atlas.brick0label);
                if isempty(indRegion), error('Region not found'); end
            end
            if iRegion == 1
                mri = sum(atlas.brick0(:,:,:,:),4) == indRegion;
            else
                mri = mri | (sum(atlas.brick0(:,:,:,:),4) == indRegion);
            end
        end
    end
%     mri = sum(atlas.brick0(:,:,:,:),4) == region;
%     mri = atlas.brick1(:,:,:,1);
    transform2 = atlas.transform;
else
    error('Unknown MRI file/structure format');
end
[r,c,v] = ind2sub(size(mri),find(mri));
pos = [r c v];
pos = atlas.transform*[pos ones(size(pos,1),1)]';
pos(4,:) = [];
pos = pos';

if nargin > 1 && ~isempty(transform)
    pos = traditionaldipfit(transform)*[pos ones(size(pos,1),1)]';
    pos(4,:) = [];
    pos = pos';
end
