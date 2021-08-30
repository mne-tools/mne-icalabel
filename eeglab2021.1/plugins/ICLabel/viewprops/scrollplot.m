function [ haxis ] = scrollplot(times, data, t_window, events, hparent, haxis, hscroll)
%SCROLL Plot EEG channel or component activity with events and scrollbar.
%   Creates a 1-dimensional scrolling plot that can incorporate EEGLAB
%   events.
%
%   Inputs:
%       times: vector of times used to label the horizontal axis (usually EEG.times)
%       data: data to plot (usually EEG.data(i, :), etc)
%       t_window: how many seconds to show initially
%       events: events to plot (usually EEG.event to show events or [] for none)
%       hparent (optional): handle for parent figure or other container
%       haxis (optional): handle for existing axis to be used for plot
%       hscroll (optional): handle for existing uicontroll slider
%
%   Outputs:
%       haxis: handle for axis used
%
% Created by Luca Pion-Tonachini 

% configure axis
if ~exist('hparent', 'var') || isempty(hparent)
    haxis = figure; end
if ~exist('haxis', 'var') || isempty(haxis)
    haxis = axes('parent', hparent); end

% reorinent time
if size(times, 1) == 1
    times = times'; end

% examine and organize data
times = single(times);
data = single(squeeze(data));
if size(data, 1) == 1
    data = data';
    epoched = false;
elseif size(data, 2) == 1
    epoched = false;
elseif numel(size(data)) == 2
    if size(data, 2) == size(times, 2)
        data = data'; end
    e_points = length(times);
    epoched = true;
else
    error('Data can be at most 2 dimensional (time x trials).')
end

% examine events
if ~exist('events', 'var') || isempty(events)
    has_events = false;
else
    has_events = true;
    event_color_dict = unique(cellfun(@num2str, {events.type}','UniformOutput', 0));
    for it = 1:length(event_color_dict)
        if isscalar(event_color_dict{it})
            event_color_dict{it} = num2str(event_color_dict{it}); end
    end
    event_color_dict = [event_color_dict, mat2cell(lines(length(event_color_dict)), ones(length(event_color_dict), 1), 3)];
end

% number of points to draw
srate = (length(times) - 1) * 1000 / (times(end) - times(1));
n_points = round(t_window * srate);
n_points = min(numel(data), n_points);

% initial hight (3std)
v_scale = double(std(data(:)) * 5 + eps);

% initial plot
hline = plot(haxis, 1:n_points, data(1:n_points));

% configure scrollbar
if ~exist('hscroll', 'var') || isempty(hscroll)
    apos = get(haxis, 'Position');
    hscroll = uicontrol('Parent', hparent, 'Style', 'Slider', ...
        'Min', 1, 'Max', numel(data) - n_points + 1, 'Value', 1, ...
        'SliderStep', double([round(n_points/10), n_points] / numel(data)), ...
        'Units', 'Normalized', ...
        'Position', [apos(1), apos(2) - 0.1, apos(3), 0.05], ...
        'Callback', @update_plot);
else
    set(hscroll, 'Callback', @update_plot, 'Value', 1)
    update_scroll(hscroll, n_points, numel(data))
%     if n_points <= numel(data)
%         set(hscroll, 'Min', 1, 'Max', 2, ...
%             'Value', 1, 'SliderStep', double([round(n_points/10), n_points]) / numel(data), ...
%             'Callback', @update_plot, 'Visible', 'off');
%     else
%         set(hscroll, 'Min', 1, 'Max', numel(data) - n_points + 1, ...
%             'Value', 1, 'SliderStep', double([round(n_points/10), n_points]) / numel(data), ...
%             'Callback', @update_plot, 'Visible', 'on');
%     end
end

% save data
all_data.hdressings = [];
all_data.haxis = haxis;
all_data.hline = hline;
all_data.data = data;
all_data.times = times;
all_data.n_points = n_points;
all_data.v_scale = v_scale;
if has_events
    all_data.events = events;
    all_data.event_color_dict = event_color_dict;
end
if epoched
    all_data.e_points = e_points; end
set(hscroll, 'UserData', all_data);

% finish initial plot
update_plot(hscroll, [])

% set figure callbacks
set(hparent, 'KeyPressFcn', {@on_key, hscroll})

end


function update_plot(hObj, event)
% setup
all_data = get(hObj, 'UserData');
hdressings = all_data.hdressings;
haxis = all_data.haxis;
hline = all_data.hline;
data = all_data.data;
times = all_data.times;
n_points = all_data.n_points;
v_scale = all_data.v_scale;
if isfield(all_data, 'events')
    events = all_data.events;
    event_color_dict = all_data.event_color_dict;
    has_events = true;
else
    has_events = false;
end
if isfield(all_data, 'e_points')
    e_points = all_data.e_points;
    epoched = true;
else
    epoched = false;
end

% update activity
index = round(get(hObj, 'Value'));
xlims = [index, index + n_points - 1];
set(haxis, 'XTickMode', 'auto')
set(hline, 'XData', xlims(1):xlims(2), 'YData', data(xlims(1):xlims(2)))
axis(haxis, [xlims, -v_scale, v_scale])
set_xtick()

% mark epochs
n_line = 1;
if epoched % TODO: save line and text for later deletion
    for lat  = ceil(xlims(1) / e_points) * e_points:e_points:floor(xlims(2) / e_points) * e_points
        mark_event(lat, ['epoch ' num2str(lat / e_points, '%d')], 'r', '-')
    end
end

% place events
if has_events
    event_latencies = [events.latency];
    for evnt = events(event_latencies >= xlims(1) & event_latencies <= xlims(2))
        evnt.type = num2str(evnt.type);
        mark_event(evnt.latency, evnt.type, ...
            event_color_dict{strcmp(event_color_dict, evnt.type), 2}, '--')
    end
end

% hide other lines
hide_extras()

% resave hdressings
all_data.hdressings = hdressings;
set(hObj, 'UserData', all_data);

    function set_xtick()
        xtick = get(haxis, 'XTick');
        % check if a zero-index is present, and remove if so
        if ~xtick(1)
            xtick(1) = [];
            xtick0 = true;
        else
            xtick0 = false;
        end
        if epoched
            % this assumes latencies returned will be integer valued
            [s1, s2] = ind2sub(size(data), xtick);
            xticklabel = times(s1);
            insert_zero_ticks()
            set(haxis, 'XTick', xtick, 'XTickLabel', xticklabel)
        else
            % this assumes latencies returned will be integer valued
            xticklabel = times(xtick);
            if xtick0
                set(haxis, 'XTick', xtick, 'XTickLabel', xticklabel)
            else
                set(haxis, 'XTickLabel', xticklabel)
            end
        end
        
        
        function insert_zero_ticks()
            if times(1) > 0 || times(end) < 0
                % no zero latency exists
                return
            else
                % find non-integer zero latency
                ind = 2;
                while ~isempty(ind) && ind < length(xtick)
                    % find first positive value
                    ind = find(xticklabel(ind:end) > 0, 1) + ind - 1;
                    if xticklabel(ind-1) < 0
                        new_latency = (xtick(ind-1) * xticklabel(ind) ...
                            - xtick(ind) * xticklabel(ind-1)) ...
                            / (xticklabel(ind) - xticklabel(ind-1));
                        % maybe move these to the end and do it all at once
                        xtick = [xtick(1:ind-1) new_latency xtick(ind:end)]; 
                        xticklabel = [xticklabel(1:ind-1);0;xticklabel(ind:end)];
                        ind = ind + 1;
                    end
                    ind = find(xticklabel(ind+1:end) <= xticklabel(ind), 1) + ind + 1;
                end
            end
        end
            
        
    end


    function mark_event(x, label, color, style)
        x = double(x);
        if size(hdressings, 2) < n_line
            % make new markers
            hdressings(1, n_line) = line([x x], [-v_scale, v_scale], ...
                'Color', color, 'Linestyle', style, 'Parent', haxis);
            hdressings(2, n_line) = text(x, v_scale * 1.02, label, 'Color', color, ...
                'HorizontalAlignment', 'left', 'Rotation', 45, 'Parent', haxis, ...
                'Interpreter', 'none');
        else
            % reuse old ones
            set(hdressings(1, n_line), 'XData', [x x], 'YData', [-v_scale, v_scale], ...
                'Color', color, 'LineStyle', style, 'Parent', haxis, 'Visible', 'on');
            set(hdressings(2, n_line), 'Position', [x, v_scale * 1.02, 0], ...
                'String', label, 'Color', color, 'HorizontalAlignment', 'left', ... 
                'Rotation', 45, 'Parent', haxis, 'Visible', 'on');
        end
        n_line = n_line + 1;
    end


    function hide_extras()
        % hide unused markers
        for it = n_line:size(hdressings, 2)
            set(hdressings(1, it), 'Visible', 'off');
            set(hdressings(2, it), 'Visible', 'off');
        end
        % delete lost markers
        for h = get(haxis, 'Children')'
            try
                x = get(h, 'position');
            catch
                continue
            end
            if (x(1) < xlims(1) || x(1) > xlims(2)) && strcmp(get(h, 'visible'), 'on')
                delete(h);
            end
        end
    end


end


% borrowed from vis_stream (BCILAB)
function on_key(hObj, event, hscroll)
% retrieve info
all_data = get(hscroll, 'UserData');
switch lower(event.Key)
    case 'uparrow'
        % decrease datascale
        all_data.v_scale = all_data.v_scale*0.9;
    case 'downarrow'
        % increase datascale
        all_data.v_scale = all_data.v_scale*1.1;
    case 'rightarrow'
        % increase timerange
        all_data.n_points = ceil(min(numel(all_data.data), all_data.n_points*1.1));
    case 'leftarrow'
        % decrease timerange
        all_data.n_points = ceil(all_data.n_points*0.9);
end

% save info
set(hscroll, 'UserData', all_data);

% update plot
update_plot(hscroll, [])

% update scrollbar
update_scroll(hscroll, all_data.n_points, numel(all_data.data))
end


% update scroll bar
function update_scroll(hscroll, n_points, n_data)
if n_points >= n_data
    set(hscroll, 'Min', 1, 'Max', 2, 'Visible', 'off', ...
        'Value', 1, 'SliderStep', double([round(n_points/10), n_points]) / n_data);
else
    set(hscroll, 'Min', 1, 'Max', n_data - n_points + 1, 'Visible', 'on', ...
        'SliderStep', double([round(n_points/10), n_points]) / n_data);
end
end
