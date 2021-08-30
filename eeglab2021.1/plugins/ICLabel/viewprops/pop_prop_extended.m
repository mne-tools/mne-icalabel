function [fh, EEG, com] = pop_prop_extended(EEG, typecomp, chanorcomp, winhandle, spec_opt, erp_opt, scroll_event, classifier_name, varargin)
% POP_PROP_EXTENDED See various common properties of an EEG channel or component
%   Creates a figure containing a scalp topography or channel location,
%   erpimage of the data, power spectral density creaty by spectopo(), a
%   scrolling plot of activity with events and epoch markings overlaid, and
%   dipole locations and residual variance if showing component properties
%   and dipole information is already present in the EEG structure. If
%   called with only the first two arguments, a GUI opens to select the
%   rest. If only one argument is given, typecomp will be set to channels 
%   (1) and the GUI will open.
%
%   Inputs
%       EEG:  EEGLAB EEG structure
%       typecomp: 0 for component, 1 for channel
%       chanorcomp:  channel or component index to plot
%       winhandle:  pass NaN, used for compatability with component rejection
%       spec_opt:  cell array of options which are passed to spectopo()
%       erp_opt:  cell array of options which are passed to erpimage()
%       scroll_event:  0 to hide events in scroll plot, 1 to show them
%       classifier_name:  string indicating which component classifier to
%           use (must match a field name in EEG.etc.ic_classification)
%       varargin:  do not use this.
%
%   Outputs:
%       fh: handle for figure used
%       EEG: EEG structure
%
%   Notes: for the dipole plot, you need EEG.dipfit precalculated
%
%   See also: spectopo(), erpimage(), scrollplot(), topoplot()
%
%   TODO Notes: remove all axes(ax* called in erpimage (2793 2836 3515 [3475])
%   also fix axcopy in the same way
%   make erpimage, spectopo, and dipplot accept axhandle inputs.
%
%   starting from pop_prop:
%   updated by Ramon Martinez-Cancino and Luca Pion-Tonachini (2015)
%   updated again by Luca Pion-Tonachini (2017)


% setup
if nargin < 1
	help pop_prop_extended;
	return;   
end;
if nargin < 5
	spec_opt = {};
end;
if nargin < 6
	erp_opt = {};
end;
if nargin < 7
	scroll_event = 1;
end;
if nargin < 8
    classifier_name = '';
    if ~typecomp && isfield(EEG.etc, 'ic_classification')
        classifiers = fieldnames(EEG.etc.ic_classification);
        if ~isempty(classifiers)
            if any(strcmpi(classifiers, 'ICLabel'));
                classifier_name = 'ICLabel';
            else
                classifier_name = classifiers{1};
            end
        end
    end
end;
if nargin == 1
	typecomp = 1;    % defaults
    chanorcomp = 1;
end;
if nargin == 10
    % from callback
    EEG = chanorcomp;
    typecomp = winhandle;
    chanorcomp = spec_opt;
    winhandle = erp_opt;
    spec_opt = scroll_event;
    erp_opt = classifier_name;
    scroll_event = varargin{1};
    classifier_name = varargin{2};
    varargin = {};
end
if typecomp == 0 && isempty(EEG.icaweights)
   error('No ICA weights recorded for this dataset -- first run ICA on it');
end;   
if nargin == 2
	promptstr    = { fastif(typecomp,'Channel index(ices) to plot:','Component index(ices) to plot:') ...
                     'Spectral options (see spectopo() help):','Erpimage options (see erpimage() help):' ...
                     [' Draw events over scrolling ' fastif(typecomp,'channel','component') ' activity']};
	inistr       = { '1' '''freqrange'', [2 80]' '', 1};
    stylestr     = {'edit', 'edit', 'edit', 'checkbox'};
    % labels when available
    if ~typecomp && isfield(EEG.etc, 'ic_classification')
        classifiers = fieldnames(EEG.etc.ic_classification);
        if ~isempty(classifiers)
            iclabel_ind = find(strcmpi(classifiers, 'ICLabel'));
            promptstr = [promptstr {classifiers}];
            inistr = [inistr {fastif(isempty(iclabel_ind), 1, iclabel_ind)}];
            stylestr = [stylestr {'popupmenu'}];
        end
    end
        
	result       = inputdlg3( 'prompt', promptstr,'style', stylestr, 'default',  inistr, ...
        'title', [fastif(typecomp,'Channel','Component') ' properties - pop_prop_extended()']);
	if size( result, 1 ) == 0
        return; end
   
	chanorcomp   = eval( [ '[' result{1} ']' ] );
    spec_opt     = eval( [ '{' result{2} '}' ] );
    erp_opt     = eval( [ '{' result{3} '}' ] );
    scroll_event   = result{4};
    if ~typecomp && isfield(EEG.etc, 'ic_classification') && ~isempty(classifiers)
        classifiers = fieldnames(EEG.etc.ic_classification);
        classifier_name = classifiers{result{5}};
    end
end;

    
% plotting several component properties
% -------------------------------------
if length(chanorcomp) > 1
    for index = chanorcomp
        pop_prop_extended(EEG, typecomp, index, nan, spec_opt, erp_opt, scroll_event, classifier_name, varargin{:});  % call recursively for each chanorcomp
    end;
	com = sprintf('pop_prop_extended( %s, %d, [%s], NaN, %s, %s, %d, ''%s''', inputname(1), ...
                  typecomp, int2str(chanorcomp), vararg2str({spec_opt}), vararg2str({erp_opt}), scroll_event, classifier_name);
    if ~isempty(varargin)
        com = [com sprintf(', %s', vararg2str(varargin))];
    end
    com = [com ');'];
    return;
end;

if chanorcomp < 1 || chanorcomp > EEG.nbchan % should test for > number of components ??? -sm
   error('Component index out of range');
end;   

% initiialize figure
try 
    icadefs;
catch
    BACKCOLOR = [0.9300 0.9600 1.0000];
end
if typecomp
    basename = ['Channel ' EEG.chanlocs(chanorcomp).labels ];
else
    basename = ['IC' int2str(chanorcomp) ];
end
fh = figure('name', [basename ' - pop_prop_extended()'],...
    'color', BACKCOLOR,...
    'numbertitle', 'off',...
    'PaperPositionMode','auto',...
    'Visible', 'off', ...
    'ToolBar', 'none',...
    'MenuBar','none');
pos = get(fh,'position');
set(fh,'Position', [pos(1)-1200+pos(3) pos(2)-700+pos(4) 1200 700]);

% initialize ica data
if ~typecomp
    if ~isempty(EEG.icaact)
        icaacttmp = EEG.icaact(chanorcomp, :, :);
    else
        icaacttmp  = eeg_getdatact(EEG, 'component', chanorcomp);
    end
end

% check for labels. if they exist, shorten scroll and plot them
if ~typecomp && isfield(EEG.etc, 'ic_classification') && ~isempty(classifier_name)
    classifiers = fieldnames(EEG.etc.ic_classification);
    classifier_name = classifiers{strcmpi(classifiers, classifier_name)};
    if size(EEG.etc.ic_classification.(classifier_name).classifications, 1) ...
            ~= size(EEG.icawinv, 2)
        warning(['The number of ICs do not match the number of IC classifications. This will result in incorrectly plotted labels. Please rerun ' classifier_name])
    end
    nclass = length(EEG.etc.ic_classification.(classifier_name).classes);
    labelax = axes('Parent', fh, 'Position', [0.32 0.6389 0.035 0.28]);
    yoffset = 0.5;
    xoffset = 0.01;
    barh(EEG.etc.ic_classification.(classifier_name).classifications(chanorcomp, end:-1:1), 'y')
    axis(labelax, [-xoffset, 1, 1 - yoffset, nclass + yoffset])
    set(labelax, 'YTickLabel', EEG.etc.ic_classification.(classifier_name).classes(end:-1:1), ...
        'XGrid', 'on', 'XTick', 0:0.5:1)
    xlabel 'Probability'
    title(classifier_name)

    for it = 1:nclass
       text(0.5, it, sprintf('%.1f%%', EEG.etc.ic_classification.(classifier_name).classifications(chanorcomp, end - it + 1) * 100), ...
           'fontsize', 11, 'HorizontalAlignment', 'center', ...
           'Parent', labelax)
    end

    scroll_position = [0.4 0.7389 0.5929 0.18];
else
    scroll_position = [0.3712 0.7389 0.5641 0.18];
end
    
% plot time series
% datax = axes('Parent', fh, 'position',,'units','normalized');
datax = axes('Parent', fh, 'Position',scroll_position,'units','normalized');
scrollax = uicontrol('Parent', fh, 'Style', 'Slider', ...
    'Units', 'Normalized', 'Position', [scroll_position(1) 0.6389 scroll_position(3) 0.025]);
if ~scroll_event
    EEG.event = []; end
if typecomp
    scrollplot(EEG.times, single(EEG.data(chanorcomp, :, :)), 5, EEG.event, fh, datax, scrollax);
    tstitle_h = title('Channel Time Series', 'fontsize', 14, 'FontWeight', 'Normal');
else
    scrollplot(EEG.times, single(icaacttmp), 5, EEG.event, fh, datax, scrollax);
    tstitle_h = title(['Scrolling IC' int2str(chanorcomp) ' Activity'], 'fontsize', 14, 'FontWeight', 'Normal');
end
set(tstitle_h,'FontSize',14, 'Position', get(tstitle_h, 'Position'), 'units', 'normalized');
set(datax,'FontSize',12);
xlabel(datax,'Time (ms)','fontsize', 14);
ylabel(datax,'uV');

% plot scalp map
axes('Parent', fh, 'position',[0.0143 0.6331 0.3121 0.3267],'units','normalized');
if typecomp
    topoplot( chanorcomp, EEG.chanlocs, 'chaninfo', EEG.chaninfo, ...
             'electrodes','off', 'style', 'blank', 'emarkersize1chan', 12); axis square;
    title(['Channel ' EEG.chanlocs(chanorcomp).labels], 'fontsize', 14, 'FontWeight', 'Normal');
else
    topoplot(EEG.icawinv(:,chanorcomp), EEG.chanlocs, ...
        'chaninfo', EEG.chaninfo, 'electrodes','on'); axis square;
    title(['IC' num2str(chanorcomp)], 'fontsize', 14, 'FontWeight', 'Normal');
end

% plot pvaf
if ~typecomp
    maxsamp = 1e5;
    n_samp = min(maxsamp, EEG.pnts*EEG.trials);
    try
        samp_ind = randperm(EEG.pnts*EEG.trials, n_samp);
    catch
        samp_ind = randperm(EEG.pnts*EEG.trials);
        samp_ind = samp_ind(1:n_samp);
    end
    if ~isempty(EEG.icachansind)
        icachansind = EEG.icachansind;
    else
        icachansind = 1:EEG.nbchan;
    end
    datavar = mean(var(EEG.data(icachansind, samp_ind), [], 2));
    projvar = mean(var(EEG.data(icachansind, samp_ind) - ...
        EEG.icawinv(:, chanorcomp) * icaacttmp(1, samp_ind), [], 2));
    pvafval = 100 *(1 - projvar/ datavar);
    pvaf = num2str(pvafval, '%3.1f');

    text(0.5, -0.12, {['{% scalp data var. accounted for}: ' pvaf '%']}, ...
        'fontsize', 13,'Units','Normalized', 'HorizontalAlignment', 'center');
end

% % plot labels
% if ~typecomp && isfield(EEG.etc, 'ic_classification')
%     classifier = classifiers{1}; % TODO: gui option for this
%     [slabels, sind] = sort(EEG.etc.ic_classification.(classifier).classifications(chanorcomp, :), 'ascend');
%     text(0.5, -0.2, sprintf('%s: %s %.1f%%, %s %.1f%%', classifier, ...
%         EEG.etc.ic_classification.(classifier).classes{sind(end)}, 100 * slabels(end), ...
%         EEG.etc.ic_classification.(classifier).classes{sind(end - 1)}, 100 * slabels(end - 1)), ...
%         'fontsize', 13,'Units','Normalized', 'HorizontalAlignment', 'center');
% end


% plot erpimage
herp = axes('Parent', fh, 'position',[0.0643 0.1102 0.2421 0.3850],'units','normalized');
eeglab_options;
if EEG.trials > 1 % epoched data
    axis(herp, 'off')
    EEG.times = linspace(EEG.xmin, EEG.xmax, EEG.pnts);
    if EEG.trials < 6
        ei_smooth = 1;
    else
        ei_smooth = 1;
    end

    if typecomp == 1 % plot channel
         offset = nan_mean(EEG.data(chanorcomp,:));
         erp=nan_mean(squeeze(EEG.data(chanorcomp,:,:))')-offset;
         erp_limits=get_era_limits(erp);
         [t1,t2,t3,t4,axhndls] = erpimage( EEG.data(chanorcomp,:)-offset, ones(1,EEG.trials)*10000, EEG.times*1000, ...
                       '', ei_smooth, 1, 'caxis', 2/3, 'cbar','erp','erp_vltg_ticks',erp_limits, erp_opt{:});   
    else % plot component
         offset     = nan_mean(icaacttmp(:));
         era        = nan_mean(squeeze(icaacttmp)')-offset;
         era_limits = get_era_limits(era);
         [t1,t2,t3,t4,axhndls] = erpimage( icaacttmp-offset, ones(1,EEG.trials)*10000, EEG.times*1000, ...
                       '', ei_smooth, 1, 'caxis', 2/3, 'cbar','erp','erp_vltg_ticks',era_limits, erp_opt{:});   
    end;
    title(['Epoched IC' int2str(chanorcomp) ' Activity'], 'fontsize', 14, 'FontWeight', 'Normal');
    lab = text(1.27, .95,'RMS uV per scalp channel');
    
else % continuoous data
    ERPIMAGELINES = 200; % show 200-line erpimage
    while size(EEG.data,2) < ERPIMAGELINES*EEG.srate
        ERPIMAGELINES = 0.9 * ERPIMAGELINES;
    end
    ERPIMAGELINES = round(ERPIMAGELINES);
    if ERPIMAGELINES > 2   % give up if data too small
        if ERPIMAGELINES < 6
            ei_smooth = 1;
        else
            ei_smooth = 3;
        end
            
        
        erpimageframes = floor(size(EEG.data,2)/ERPIMAGELINES);
        erpimageframestot = erpimageframes*ERPIMAGELINES;
        eegtimes = linspace(0, erpimageframes-1, length(erpimageframes));
        if typecomp == 1 % plot channel
            offset = nan_mean(EEG.data(chanorcomp,:));
            % Note: we don't need to worry about ERP limits, since ERPs
            % aren't visualized for continuous data
            [t1,t2,t3,t4,axhndls] = erpimage( reshape(EEG.data(chanorcomp,1:erpimageframestot),erpimageframes,ERPIMAGELINES)-offset, ones(1,ERPIMAGELINES)*10000, eegtimes , ...
                '', ei_smooth, 1, 'caxis', 2/3, 'cbar', erp_opt{:});
        else % plot component
            offset = nan_mean(icaacttmp(:));
            [t1,t2,t3,t4,axhndls] = erpimage(reshape(icaacttmp(:,1:erpimageframestot),erpimageframes,ERPIMAGELINES)-offset,ones(1,ERPIMAGELINES)*10000, eegtimes , ...
                '', ei_smooth, 1, 'caxis', 2/3, 'cbar', erp_opt{:});
        end
        
        try 
            ylabel(axhndls{1}, 'Data');
        catch
            ylabel(axhndls(1), 'Data');
        end
        title('Continuous Data', 'fontsize', 14, 'FontWeight', 'Normal');
        lab = text(1.27, .85,'RMS uV per scalp channel');
    else
        axis off;
        text(0.1, 0.3, [ 'No erpimage plotted' 10 'for small continuous data']);
    end
end

if exist('axhndls', 'var')
    try
        % 2014+
        axhndls{1}.FontSize = 12;
        axhndls{1}.YLabel.FontSize = 14;
        set(axhndls{2},'position', get(axhndls{2},'position') - [0.01 0 0.02 0]);
        try
            axhndls{3}.FontSize = 12;
            axhndls{3}.XLabel.FontSize = 14; %#ok<NASGU>
        catch
            axhndls{1}.XLabel.FontSize = 14; %#ok<NASGU>
        end
    catch
        % 2013-
        set(axhndls(1), 'FontSize', 12)
        set(get(axhndls(1), 'Ylabel'), 'FontSize', 14)
        set(axhndls(2),'position', get(axhndls(2),'position') - [0.01 0 0.02 0], ...
            'Fontsize', 12)
        if ~isnan(axhndls(3))
            set(axhndls(3), 'FontSize', 12)
            set(get(axhndls(3), 'Xlabel'), 'FontSize', 14)
        else
            set(get(axhndls(1), 'Xlabel'), 'FontSize', 14)
        end
    end
    set(lab, 'rotation', -90, 'FontSize', 12)
end

% plot spectrum
try
    hfreq = axes('Parent', fh, 'position', [0.5765 0.1109 0.3587 0.4336], 'units', 'normalized');
    if typecomp
        spectopo( EEG.data(chanorcomp,:), EEG.pnts, EEG.srate, spec_opt{:} );
        title(hfreq,'Channel Activity Power Spectrum','units','normalized', 'fontsize', 14, 'FontWeight', 'Normal');
    else
        spectopo( icaacttmp(1, :), EEG.pnts, EEG.srate, 'mapnorm', EEG.icawinv(:,chanorcomp), spec_opt{:} );
        title(hfreq,['IC' int2str(chanorcomp) ' Activity Power Spectrum'],'units','normalized', 'fontsize', 14, 'FontWeight', 'Normal');
    end
	set(get(hfreq, 'ylabel'), 'string', 'Power 10*log_{10}(uV^2/Hz)', 'fontsize', 14); 
	set(get(hfreq, 'xlabel'), 'string', 'Frequency (Hz)', 'fontsize', 14, 'fontweight', 'normal'); 
	set(hfreq, 'fontsize', 14, 'fontweight', 'normal');
    xlims = xlim;
    hfreqline = findobj(hfreq, 'type', 'line');
    xdata = get(hfreqline, 'xdata');
    ydata = get(hfreqline, 'ydata');
    ind = xdata >= xlims(1) & xdata <= xlims(2);
    axis on;
    axis([xlims min(ydata(ind)) max(ydata(ind))])
    box on;
    grid on;
catch e
    cla(hfreq);
    disp(e)
    text(0.1, 0.3, [ 'Error: no spectrum plotted' 10 ' make sure you have the ' 10 'signal processing toolbox']);
end

% Defining path for system
eeglabpath = which('eeglab.m');
pathtmp = fileparts(eeglabpath);
dipfits = dir(fullfile(pathtmp, 'plugins', 'dipfit*'));
[~, dipfit_order] = sort(cellfun(@(c) str2double(c(7:end)), {dipfits.name}), 'descend');
for it_dipfit_version = dipfit_order
    dipfit_folder = fullfile(pathtmp, 'plugins', dipfits(it_dipfit_version).name);
    meshdatapath = fullfile(dipfit_folder, 'standard_BEM', 'standard_vol.mat');
    mripath = fullfile(dipfit_folder, 'standard_BEM', 'standard_mri.mat');

    if ~typecomp && exist(meshdatapath,'file') == 2 && exist(mripath,'file') == 2
        % dipplot
        if isfield(EEG, 'dipfit') && ~isempty(EEG.dipfit)
            try
                rv = num2str(EEG.dipfit.model(chanorcomp).rv*100, '%.1f');
            catch
                rv = 'N/A';
            end
            dip_background = axes('Parent', fh, 'position', [0.41 0.1 0.1 0.1557*3+0.0109], ...
                'units', 'normalized', 'XLim', [0 1], 'Ylim', [0 1]);
            patch([0 0 1 1], [0 1 1 0], 'k', 'parent', dip_background)
            axis(dip_background, 'off')
            colors = {'g', 'm', 'y'};

            % axial
            ax(1) = axes('Parent', fh, 'position', [0.41 0.1109 0.1 0.1557], 'units', 'normalized');
            axis equal off
            dipplot(EEG.dipfit.model(chanorcomp), ...
                'meshdata', meshdatapath, ...
                'mri', mripath, ...
                'normlen', 'on', 'coordformat', 'MNI', 'axistight', 'on', 'gui', 'off', 'view', [0 0 1], 'pointout', 'on');
            temp = axes('Parent', fh, 'position', [0.41 0.1109 0.1 0.1557], 'units', 'normalized');
            copyobj(allchild(ax(1)),temp);
            delete(ax(1))
            ax(1) = temp;
            axis equal off
            temp = get(ax(1),'children');
            ind = find(strcmp('line', get(temp, 'type')));
            for it = 1:length(ind)
                if mod(it, 2)
                    set(temp(ind(it)), 'markersize', 15, 'color', colors{ceil(it / 2)})
                else
                    set(temp(ind(it)), 'linewidth', 2, 'color', colors{ceil(it / 2)})
                end
            end

            % coronal
            ax(2) = axes('Parent', fh, 'position', [0.41 0.2666 0.1 0.1557], 'units', 'normalized');
            axis equal off
            copyobj(allchild(ax(1)),ax(2));
            view([0 -1 0])
            axis equal off
            temp = get(ax(2),'children');
            ind = find(strcmp('line', get(temp, 'type')));
            for it = 1:length(ind)
                if mod(it, 2)
                    set(temp(ind(it)), 'markersize', 15, 'color', colors{ceil(it / 2)})
                else
                    set(temp(ind(it)), 'linewidth', 2, 'color', colors{ceil(it / 2)})
                end
            end

            % sagital
            ax(3) = axes('Parent', fh, 'position', [0.41 0.4223 0.1 0.1557], 'units', 'normalized');
            axis equal off
            copyobj(allchild(ax(1)),ax(3));
            view([1 0 0])
            axis equal off
            temp = get(ax(3),'children');
            ind = find(strcmp('line', get(temp, 'type')));
            for it = 1:length(ind)
                if mod(it, 2)
                    set(temp(ind(it)), 'markersize', 15, 'color', colors{ceil(it / 2)})
                else
                    set(temp(ind(it)), 'linewidth', 2, 'color', colors{ceil(it / 2)})
                end
            end

            % dipole text
            dip_title = title(dip_background, 'Dipole Position', 'FontWeight', 'Normal');
            set(dip_title,'FontSize',14);
            set(fh, 'CurrentAxes', ax(1))
            if size(EEG.dipfit.model(chanorcomp).momxyz, 1) == 2
                dmr = norm(EEG.dipfit.model(chanorcomp).momxyz(1,:)) ...
                    / norm(EEG.dipfit.model(chanorcomp).momxyz(2,:));
                if dmr<1
                    dmr = 1/dmr; end
                text(-50,-173,{['RV: ' rv '%']; ['DMR:' num2str(dmr,'%.1f')]})
            else
                text(-50,-163,['RV: ' rv '%'])
            end
            
            % exit loop over dipfit versions
            break
        end
    end
end

% final figure adjustments
rotate3d(fh, 'off');
set(fh, 'color', BACKCOLOR, 'visible', 'on')


% display buttons
% ---------------
if ~exist('winhandle', 'var')
    winhandle = nan; end
if isobject(winhandle) || ~isnan(winhandle)
	COLREJ = '[1 0.6 0.6]';
	COLACC = '[0.75 1 0.75]';
    bottom = 0.005;
    height = 0.04;
	% CANCEL button
	% -------------
	h  = uicontrol(gcf, 'Style', 'pushbutton', 'backgroundcolor', GUIBUTTONCOLOR, 'string', 'Cancel', 'Units','Normalized','Position',[0.2 bottom 0.1 height], 'callback', 'close(gcf);');

	% VALUE button
	% -------------
	hval  = uicontrol(gcf, 'Style', 'pushbutton', 'backgroundcolor', GUIBUTTONCOLOR, 'string', 'Values', 'Units','Normalized', 'Position', [0.325 bottom 0.1 height]);

	% REJECT button
	% -------------
    if ~isempty(EEG.reject.gcompreject)
    	status = EEG.reject.gcompreject(chanorcomp);
    else
        status = 0;
    end;
	hr = uicontrol(gcf, 'Style', 'pushbutton', 'backgroundcolor', eval(fastif(status,COLREJ,COLACC)), ...
				'string', fastif(status, 'REJECT', 'ACCEPT'), 'Units','Normalized', 'Position', [0.45 bottom 0.1 height], 'userdata', status, 'tag', 'rejstatus');
	command = [ 'set(gcbo, ''userdata'', ~get(gcbo, ''userdata''));' ...
				'if get(gcbo, ''userdata''),' ...
				'     set( gcbo, ''backgroundcolor'',' COLREJ ', ''string'', ''REJECT'');' ...
				'else ' ...
				'     set( gcbo, ''backgroundcolor'',' COLACC ', ''string'', ''ACCEPT'');' ...
				'end;' ];					
	set( hr, 'callback', command); 

	% HELP button
	% -------------
	h  = uicontrol(gcf, 'Style', 'pushbutton', 'backgroundcolor', GUIBUTTONCOLOR, 'string', 'HELP', 'Units','Normalized', 'Position', [0.575 bottom 0.1 height], 'callback', 'pophelp(''pop_prop'');');

	% OK button
	% ---------
 	command = [ 'global EEG;' ...
 				'tmpstatus = get( findobj(''parent'', gcbf, ''tag'', ''rejstatus''), ''userdata'');' ...
 				'EEG.reject.gcompreject(' num2str(chanorcomp) ') = tmpstatus;' ]; 
	if winhandle ~= 0
	 	command = [ command ...
	 				sprintf('if tmpstatus set(%3.15f, ''backgroundcolor'', %s); else set(%3.15f, ''backgroundcolor'', %s); end;', ...
					winhandle, COLREJ, winhandle, COLACC)];
	end;				
	command = [ command 'close(gcf); clear tmpstatus' ];
	h  = uicontrol(gcf, 'Style', 'pushbutton', 'string', 'OK', 'backgroundcolor', GUIBUTTONCOLOR, 'Units','Normalized', 'Position',[0.7 bottom 0.1 height], 'callback', command);

	% draw the figure for statistical values
	% --------------------------------------
	index = num2str( chanorcomp );
	command = [ ...
		'figure(''MenuBar'', ''none'', ''name'', ''Statistics of the component'', ''numbertitle'', ''off'');' ...
		'' ...
		'pos = get(gcf,''Position'');' ...
		'set(gcf,''Position'', [pos(1) pos(2) 340 340]);' ...
		'pos = get(gca,''position'');' ...
		'q = [pos(1) pos(2) 0 0];' ...
		's = [pos(3) pos(4) pos(3) pos(4)]./100;' ...
		'axis off;' ...
		''  ...
		'txt1 = sprintf(''(\n' ...
						'Entropy of component activity\t\t%2.2f\n' ...
					    '> Rejection threshold \t\t%2.2f\n\n' ...
					    ' AND                 \t\t\t----\n\n' ...
					    'Kurtosis of component activity\t\t%2.2f\n' ...
					    '> Rejection threshold \t\t%2.2f\n\n' ...
					    ') OR                  \t\t\t----\n\n' ...
					    'Kurtosis distibution \t\t\t%2.2f\n' ...
					    '> Rejection threhold\t\t\t%2.2f\n\n' ...
					    '\n' ...
					    'Current thesholds sujest to %s the component\n\n' ...
					    '(after manually accepting/rejecting the component, you may recalibrate thresholds for future automatic rejection on other datasets)'',' ...
						'EEG.stats.compenta(' index '), EEG.reject.threshentropy, EEG.stats.compkurta(' index '), ' ...
						'EEG.reject.threshkurtact, EEG.stats.compkurtdist(' index '), EEG.reject.threshkurtdist, fastif(EEG.reject.gcompreject(' index '), ''REJECT'', ''ACCEPT''));' ...
		'' ...				
		'uicontrol(gcf, ''Units'',''Normalized'', ''Position'',[-11 4 117 100].*s+q, ''Style'', ''frame'' );' ...
		'uicontrol(gcf, ''Units'',''Normalized'', ''Position'',[-5 5 100 95].*s+q, ''String'', txt1, ''Style'',''text'', ''HorizontalAlignment'', ''left'' );' ...
		'h = uicontrol(gcf, ''Style'', ''pushbutton'', ''string'', ''Close'', ''Units'',''Normalized'', ''Position'', [35 -10 25 10].*s+q, ''callback'', ''close(gcf);'');' ...
		'clear txt1 q s h pos;' ];
	set( hval, 'callback', command); 
	if isempty( EEG.stats.compenta )
		set(hval, 'enable', 'off');
	end;
	
	com = sprintf('pop_prop_extended( %s, %d, [%s], 0, %s, %s, %d, ''%s''', inputname(1), ...
                  typecomp, int2str(chanorcomp), vararg2str({spec_opt}), vararg2str({erp_opt}), scroll_event, classifier_name);
    if ~isempty(varargin)
        com = [com sprintf(', %s', vararg2str(varargin))];
    end
    com = [com ');'];
else
	com = sprintf('pop_prop_extended( %s, %d, [%s], NaN, %s, %s, %d, ''%s''', inputname(1), ...
                  typecomp, int2str(chanorcomp), vararg2str({spec_opt}), vararg2str({erp_opt}), scroll_event, classifier_name);
    if ~isempty(varargin)
        com = [com sprintf(', %s', vararg2str(varargin))];
    end
    com = [com ');'];
end;

drawnow;


function era_limits=get_era_limits(era)
%function era_limits=get_era_limits(era)
%
% Returns the minimum and maximum value of an event-related
% activation/potential waveform (after rounding according to the order of
% magnitude of the ERA/ERP)
%
% Inputs:
% era - [vector] Event related activation or potential
%
% Output:
% era_limits - [min max] minimum and maximum value of an event-related
% activation/potential waveform (after rounding according to the order of
% magnitude of the ERA/ERP)

mn=min(era);
mx=max(era);
mn=orderofmag(mn)*round(mn/orderofmag(mn));
mx=orderofmag(mx)*round(mx/orderofmag(mx));
era_limits=[mn mx];


function ord=orderofmag(val)
%function ord=orderofmag(val)
%
% Returns the order of magnitude of the value of 'val' in multiples of 10
% (e.g., 10^-1, 10^0, 10^1, 10^2, etc ...)
% used for computing erpimage trial axis tick labels as an alternative for
% plotting sorting variable

val=abs(val);
if val>=1
    ord=1;
    val=floor(val/10);
    while val>=1,
        ord=ord*10;
        val=floor(val/10);
    end
    return;
else
    ord=1/10;
    val=val*10;
    while val<1,
        ord=ord/10;
        val=val*10;
    end
    return;
end

% inputdlg3() - A comprehensive gui automatic builder. This function takes
%               text, type of GUI and default value and builds
%               automatically a simple graphic interface.
%
% Usage:
%   >> [outparam outstruct] = inputdlg3( 'key1', 'val1', 'key2', 'val2', ... );
% 
% Inputs:
%   'prompt'     - cell array of text
%   'style'      - cell array of style for each GUI. Default is edit.
%   'default'    - cell array of default values. Default is empty.
%   'tags'       - cell array of tag text. Default is no tags.
%   'tooltip'    - cell array of tooltip texts. Default is no tooltip.
%
% Output:
%   outparam   - list of outputs. The function scans all lines and
%                add up an output for each interactive uicontrol, i.e
%                edit box, radio button, checkbox and listbox.
%   userdat    - 'userdata' value of the figure.
%   strhalt    - the function returns when the 'userdata' field of the
%                button with the tag 'ok' is modified. This returns the
%                new value of this field.
%   outstruct  - returns outputs as a structure (only tagged ui controls
%                are considered). The field name of the structure is
%                the tag of the ui and contain the ui value or string.
%
% Note: the function also adds three buttons at the bottom of each 
%       interactive windows: 'CANCEL', 'HELP' (if callback command
%       is provided) and 'OK'.
%
% Example:
%   res = inputdlg3('prompt', { 'What is your name' 'What is your age' } );
%   res = inputdlg3('prompt', { 'Chose a value below' 'Value1|value2|value3' ...
%                   'uncheck the box' }, ...
%                   'style',  { 'text' 'popupmenu' 'checkbox' }, ...
%                   'default',{ 0 2 1 });
%
% Author: Arnaud Delorme, Tim Mullen, Christian Kothe, SCCN, INC, UCSD
%
% See also: supergui(), eeglab()

% Copyright (C) Arnaud Delorme, SCCN, INC, UCSD, 2010, arno@ucsd.edu
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

function [result, userdat, strhalt, resstruct] = inputdlg3( varargin);

if nargin < 2
   help inputdlg3;
   return;
end;	

% check input values
% ------------------
[opt addopts] = finputcheck(varargin, { 'prompt'  'cell'  []   {};
                                        'style'   'cell'  []   {};
                                        'default' 'cell'  []   {};
                                        'tag'     'cell'  []   {};
                                        'tooltip','cell'  []   {}}, 'inputdlg3', 'ignore');
if isempty(opt.prompt),  error('The ''prompt'' parameter must be non empty'); end;
if isempty(opt.style),   opt.style = cell(1,length(opt.prompt)); opt.style(:) = {'edit'}; end;
if isempty(opt.default), opt.default = cell(1,length(opt.prompt)); opt.default(:) = {0}; end;
if isempty(opt.tag),     opt.tag = cell(1,length(opt.prompt)); opt.tag(:) = {''}; end;

% creating GUI list input
% -----------------------
uilist = {};
uigeometry = {};
outputind  = ones(1,length(opt.prompt));
for index = 1:length(opt.prompt)
    if strcmpi(opt.style{index}, 'edit')
        uilist{end+1} = { 'style' 'text' 'string' opt.prompt{index} };
        uilist{end+1} = { 'style' 'edit' 'string' opt.default{index} 'tag' opt.tag{index} 'tooltip' opt.tag{index}};
        uigeometry{index} = [2 1];
    else
        uilist{end+1} = { 'style' opt.style{index} 'string' opt.prompt{index} 'value' opt.default{index} 'tag' opt.tag{index} 'tooltip' opt.tag{index}};
        uigeometry{index} = [1];
    end;
    if strcmpi(opt.style{index}, 'text')
        outputind(index) = 0;
    end;
end;

[tmpresult, userdat, strhalt, resstruct] = inputgui('uilist', uilist,'geometry', uigeometry, addopts{:});
try
    result = cell(1,length(opt.prompt));
    result(find(outputind)) = tmpresult;
catch
    result = [];
end