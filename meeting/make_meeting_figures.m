function make_meeting_figures(part)
%MAKE_MEETING_FIGURES  Figures for the meeting with the supervisor (23 Oct 2026).
%   make_meeting_figures('mirror')   fast (about 1-2 min): 1 trailer, two start poses
%   make_meeting_figures('phase')    slow (about 15 min):  2 trailers, phase of the oscillating terms
%   make_meeting_figures()           both
%   Output: PNG and PDF files in <thesis>/meeting/ ; data of the slow part in phase_runs.mat

    if nargin < 1, part = 'all'; end
    base = '/Users/marke/Documents/MATLAB/turtlebot/bachelor thesis';
    outd = '/Users/marke/Documents/thesis/meeting';
    if ~exist(outd, 'dir'), mkdir(outd); end
    c0 = pwd;  back = onCleanup(@() cd(c0));

    if any(strcmp(part, {'all', 'mirror'})), fig_mirror(base, outd); end
    if any(strcmp(part, {'all', 'phase'})),  fig_phase(base, outd);  end
end

% ------------------------------------------------------------------------
function fig_mirror(base, outd)
% One trailer: design start (-0.2,-0.2) against the mirrored start (+0.2,+0.2).
    cd(fullfile(base, '1 trailer'));
    P = params();  d0 = P.d0;  l1 = P.l1;  alpha = d0*(d0 + l1)/2;
    gam = 3.2028e-3;  ep = 5;  k1 = 3;  k2 = 2;  delta = 0.02;     % design of chapter 7
    h = @(z) 1/sqrt(norm(z)^2 + delta^2);
    starts = {[-0.2; -0.2; pi/4; pi/4], [0.2; 0.2; pi/4; pi/4]};
    names  = {'start (-0.2, -0.2) m  (design)', 'start (+0.2, +0.2) m  (mirrored)'};
    col    = {[0 0.447 0.741], [0.835 0.369 0]};
    T = 400;  R = cell(1, 2);
    for i = 1:2
        [te, xe, ze] = trailer1('simulate_case_physical', starts{i}, d0, l1, alpha, ...
                                gam, ep, 2*pi/ep, k1, k2, 0, T, 0.02, h);
        R{i} = struct('t', te, 'x', xe, 'nz', vecnorm(ze, 2, 2));
        R{i}.nz = R{i}.nz / R{i}.nz(1);
        fprintf('%s: final trailer angle %.2f deg, final ||z||/||z0|| = %.3e, max |hitch| = %.3f rad\n', ...
                names{i}, rad2deg(xe(end, 4)), R{i}.nz(end), max(abs(xe(:,3) - xe(:,4))));
    end
    sing = acos(-l1/d0);                                  % singularity of the hitch map [rad]
    % samples at the ends of the sampling intervals (t = j*eps): what the controller sees
    for i = 1:2, R{i}.s = abs(R{i}.t/ep - round(R{i}.t/ep)) < 1e-6; end
    thin = 0.35;                                          % transparency of the full signal

    f = figure('Color', 'w', 'Position', [100 100 1100 360]);
    tiledlayout(1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
    nexttile; hold on; box on;
    for i = 1:2
        pl = plot(R{i}.t, rad2deg(R{i}.x(:,4)), 'Color', [col{i} thin], 'LineWidth', 0.5);
        plot(R{i}.t(R{i}.s), rad2deg(R{i}.x(R{i}.s,4)), '-', 'Color', col{i}, 'LineWidth', 2);
    end
    yline(0, ':', 'Color', [0.4 0.4 0.4]);
    xlabel('time [s]'); ylabel('trailer angle \theta_1 [deg]'); title('(a) trailer angle');
    text(0.04, 0.10, sprintf('mirrored start: %.0f deg at the end of every interval', ...
         rad2deg(R{2}.x(end,4))), 'Units', 'normalized', 'Color', col{2}, 'FontSize', 9);
    nexttile; hold on; box on;
    for i = 1:2
        plot(R{i}.t, max(R{i}.nz, 1e-16), 'Color', [col{i} thin], 'LineWidth', 0.5);
        plot(R{i}.t(R{i}.s), max(R{i}.nz(R{i}.s), 1e-16), '-', 'Color', col{i}, 'LineWidth', 2);
    end
    set(gca, 'YScale', 'log'); ylim([1e-8 2]);
    xlabel('time [s]'); ylabel('||z|| / ||z_0||'); title('(b) distance to the goal in z');
    nexttile; hold on; box on;
    for i = 1:2
        hh = abs(R{i}.x(:,3) - R{i}.x(:,4));
        plot(R{i}.t, hh, 'Color', [col{i} thin], 'LineWidth', 0.5);
        plot(R{i}.t(R{i}.s), hh(R{i}.s), '-', 'Color', col{i}, 'LineWidth', 2);
    end
    yline(1.40, '--', 'URDF joint limit 1.40 rad', 'Color', [0.3 0.3 0.3], ...
          'LabelHorizontalAlignment', 'right', 'LabelVerticalAlignment', 'bottom');
    yline(sing, '--', sprintf('singularity %.3f rad', sing), 'Color', [0.7 0 0], ...
          'LabelHorizontalAlignment', 'right', 'LabelVerticalAlignment', 'bottom');
    xlabel('time [s]'); ylabel('hitch angle |x_3 - x_4| [rad]'); title('(c) hitch angle');
    % legend: only the two thick lines
    h1 = plot(nan, nan, '-', 'Color', col{1}, 'LineWidth', 2);
    h2 = plot(nan, nan, '-', 'Color', col{2}, 'LineWidth', 2);
    lg = legend([h1 h2], [names, {}], 'Orientation', 'horizontal', 'Box', 'off');
    lg.Layout.Tile = 'south';
    annotation(f, 'textbox', [0.0 0.0 0.2 0.05], 'String', 'thick: end of each interval; thin: full signal', ...
               'EdgeColor', 'none', 'FontSize', 8, 'Color', [0.4 0.4 0.4]);
    save_both(f, fullfile(outd, 'mirror_start_1tr'));
end

% ------------------------------------------------------------------------
function fig_phase(base, outd)
% Two trailers, eps = 3 s, start x0.25: phase shift of the oscillating groups.
    cd(fullfile(base, '2 trailers'));
    cases = {[0 0 0], [0 pi/2 0], [0 0 pi/2]};
    lab   = {'phase [0 0 0]  (original)', 'phase [0 \pi/2 0]', 'phase [0 0 \pi/2]'};
    col   = {[0.835 0.369 0], [0 0.447 0.741], [0 0.620 0.451]};
    L = cell(1, numel(cases));
    for i = 1:numel(cases)
        fprintf('\n##### phase %s\n', mat2str(cases{i}, 4));
        R = thesis_sim('long', struct('t_end', 6000, 'scale', 0.25, 'plots', false, 'phase', cases{i}));
        L{i} = R.long;
    end
    save(fullfile(outd, 'phase_runs.mat'), 'L', 'cases');
    f = figure('Color', 'w', 'Position', [100 100 1000 380]);
    tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    nexttile; hold on; box on;
    for i = 1:numel(cases)
        plot(L{i}.t, L{i}.exact, 'Color', col{i}, 'LineWidth', 1.6);
    end
    plot(L{1}.t, L{1}.approx, 'k--', 'LineWidth', 1.4);
    xlabel('time [s]'); ylabel('||z|| / ||z_0||'); title('(a) exact model');
    ylim([0 3]);
    legend([lab, {'approximation (design model)'}], 'Location', 'northwest', 'Box', 'off');
    nexttile; hold on; box on;
    for i = 1:numel(cases)
        plot(L{i}.t(2:end), L{i}.u1_int, '.', 'Color', col{i}, 'MarkerSize', 4);
    end
    yline(1, 'k-', 'limit');
    xlabel('time [s]'); ylabel('peak |u_1| in the interval / limit');
    title('(b) forward speed per interval (exact model)');
    save_both(f, fullfile(outd, 'phase_test_2tr'));
end

% ------------------------------------------------------------------------
function save_both(f, name)
    try
        exportgraphics(f, [name '.png'], 'Resolution', 200);
        exportgraphics(f, [name '.pdf'], 'ContentType', 'vector');
    catch
        print(f, [name '.png'], '-dpng', '-r200');
    end
    fprintf('saved %s.png\n', name);
end
