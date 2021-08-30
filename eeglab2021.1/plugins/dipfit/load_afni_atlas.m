function [newVol, xyz, labels, labelsstr ] = load_afni_atlas(sourcemodel, headmodel, sourcemodel2mni, downsample)

    afni = ft_read_atlas(sourcemodel);
    vol = afni.brick1(:,:,:,1);
    labelsstr = afni.brick1label;
    
    sub = downsample; % Subsample by 4
    newVolCell = cell(ceil(size(vol)/sub));
    for x = 1:size(vol,1)
        for y = 1:size(vol,2)
            for z = 1:size(vol,3)
                newVolCell{ceil(x/sub), ceil(y/sub), ceil(z/sub)} = [ newVolCell{ceil(x/sub), ceil(y/sub), ceil(z/sub)} vol(x,y,z) ];
            end
        end
    end
    
    % get dominating label for voxel
    newVol = zeros(size(newVolCell));
    for iVol = 1:prod(size(newVolCell))
        uniq = unique(newVolCell{iVol});
        if length(uniq) > 1
            count = histc(newVolCell{iVol}, uniq);
            [~,indMax] = max(count);
            newVol(iVol) = uniq(indMax);
        else
            newVol(iVol) = uniq;
        end
    end
    
    % recompute coordinates
    inds = find(newVol);
    [r,c,v] = ind2sub(size(newVol),find(newVol));
    r = (r-1)*downsample + (downsample-1)/2+1;
    c = (c-1)*downsample + (downsample-1)/2+1;
    v = (v-1)*downsample + (downsample-1)/2+1;
    labels = newVol(inds);
    
    % transform coordinates
    xyz = [r c v ones(length(r),1)];
    xyz = afni.transform*xyz';
    if nargin > 2 && ~isempty(sourcemodel2mni)
        xyz = traditionaldipfit(sourcemodel2mni)*xyz;
    end
    xyz(4,:) = [];
    xyz = xyz';
    
    headmodel = load('-mat', headmodel);
    try 
        inside = ft_inside_headmodel(xyz, headmodel);
    catch
        % ft_inside_headmodel not compatible with headmodel, using custom code
        p = pwd;
        cd('/data/matlab/eeglab/plugins/fieldtrip/forward/private')
        inside = bounding_mesh(xyz, headmodel.vol.bnd(end).pnt, headmodel.vol.bnd(end).tri);
        cd(p);
    end
    inside = inside > 0;

    if 0
        figure; plot3dmeshalign(headmodel);
        plot3(xyz(inside,1),xyz(inside,2),xyz(inside,3), 'b.');
        hold on; plot3(xyz(~inside,1),xyz(~inside,2),xyz(~inside,3), 'r.');
    end

    xyz(~inside,:) = [];
    labels(~inside) = [];
    