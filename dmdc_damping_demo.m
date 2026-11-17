function out = dmdc_damping_demo()
%DMDC_DAMPING_DEMO  Steering ALS out of a two-factor degeneracy with DMDc-controlled damping.
% Degenerate tensor of swamp_diagnostic_demo (rank 3, border rank 2, 1 % noise), fitted with
% R = 2 from one random start.  Methods: plain ALS, fixed damping (three levels), decaying
% damping, rate-triggered damping, DMDc proportional damping.  Prints a table, writes
% results/fig_dmdc.png and returns the per-method outputs.
maxNumCompThreads(1);
I = [30 30 30];
rng(3); nv = @(v) v / norm(v);
a = nv(randn(I(1), 1)); a2 = nv(randn(I(1), 1)); b = nv(randn(I(2), 1)); b2 = nv(randn(I(2), 1));
c = nv(randn(I(3), 1)); c2 = nv(randn(I(3), 1));
X = reshape(a * kron(c2, b)' + a * kron(c, b2)' + a2 * kron(c, b)', I);
X = X + 0.01 * norm(X(:)) / sqrt(numel(X)) * randn(I);
R = 2; rng(11); A0 = arrayfun(@(n) randn(I(n), R), 1:3, 'uni', 0);
half = 0.5 * sum(X(:).^2);
M = {struct('control','none'), struct('control','fixed','mu0',1e-3), struct('control','fixed','mu0',1e-2), ...
     struct('control','fixed','mu0',1e-1), struct('control','trigger','mu_high',1e-2), ...
     struct('control','ratefb','mu0',1e-3,'lam_target',0.9), struct('control','dmdc','mu0',1e-3,'lam_target',0.9), ...
     struct('control','dmdc','mu0',1e-3,'lam_target',0.95)};
names = {'plain ALS', 'fixed \mu = 10^{-3}', 'fixed \mu = 10^{-2}', 'fixed \mu = 10^{-1}', 'rate-triggered \mu = 10^{-2}', ...
         'rate feedback, target 0.9', 'DMDc proportional, target 0.9', 'DMDc proportional, target 0.95'};
cols = [42 120 214; 235 104 52; 27 175 122; 237 161 0; 232 123 164; 0 131 0; 98 80 214; 227 73 72] / 255;
out = cell(1, numel(M));
fprintf('%-34s %6s %9s %10s %9s %8s %9s %7s %8s\n', 'method', 'iters', 'stop', 'f/(.5|X|^2)', 'max w', 'congr', 'final mu', 't_on', 'phase');
for i = 1:numel(M)
    o = M{i}; o.max_iters = 1500; o.init = A0;
    [~, out{i}] = cp_als_dmdc(X, R, o); r = out{i};
    fprintf('%-34s %6d %9s %10.4e %9.3f %8.3f %9.2e %7s %8s\n', strrep(strrep(names{i}, '\mu', 'mu'), '\cdot', '*'), r.niters, r.stop, r.fval / half, r.wmax_hist(end), r.congr_hist(end), r.mu_hist(end), mat2str(r.t_on), r.phase);
end
fmin = min(cellfun(@(r) min(r.f_hist), out));
fig = figure('Color', 'w', 'Position', [60 60 1000 640], 'Visible', 'off');
tl = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
ax1 = nexttile; hold(ax1, 'on');
ax2 = nexttile; hold(ax2, 'on');
ax3 = nexttile; hold(ax3, 'on');
ax4 = nexttile; hold(ax4, 'on');
for i = 1:numel(M)
    r = out{i}; k = (1:r.niters)';
    plot(ax1, k, r.f_hist / half, '-', 'Color', cols(i, :), 'LineWidth', 1.8, 'DisplayName', names{i});
    plot(ax2, k, max(r.mu_hist, 1e-6), '-', 'Color', cols(i, :), 'LineWidth', 1.8);
    plot(ax3, k, r.wmax_hist, '-', 'Color', cols(i, :), 'LineWidth', 1.8);
    plot(ax4, k, r.congr_hist, '-', 'Color', cols(i, :), 'LineWidth', 1.8);
end
set([ax1 ax2 ax3 ax4], 'XScale', 'log', 'Box', 'off', 'YGrid', 'on', 'GridAlpha', 0.15);
set(ax1, 'YScale', 'log'); set(ax2, 'YScale', 'log'); set(ax3, 'YScale', 'log');
ylabel(ax1, 'relative objective f / (1/2 ||X||^2)'); ylabel(ax2, 'damping \mu_k'); ylabel(ax3, 'largest weight'); ylabel(ax4, 'signed congruence of the pair');
xlabel(ax3, 'iteration'); xlabel(ax4, 'iteration');
yline(ax4, -1, ':', 'Color', [0.4 0.4 0.4], 'HandleVisibility', 'off');
lg = legend(ax1, 'Location', 'eastoutside', 'Box', 'off'); lg.Layout.Tile = 'east';
title(tl, 'Degenerate tensor (border rank 2) fitted with R = 2: damping strategies', 'FontSize', 11);
if ~exist('results', 'dir'), mkdir('results'); end
exportgraphics(fig, fullfile('results', 'fig_dmdc.png'), 'Resolution', 150); close(fig);
fprintf('figure written to results/fig_dmdc.png\n');
for i = find(cellfun(@(r) ~isempty(r.ident), out))
    fprintf('\n%s: identification attempts (k, ok, lam1, b1, c1, alpha):\n', strrep(names{i}, '\mu', 'mu'));
    for j = 1:numel(out{i}.ident)
        d = out{i}.ident(j);
        fprintf('  k=%4d  ok=%d  lam1=%.4f  b1=%+.3e  c1=%+.3e  alpha=%+.3e  %s\n', d.k, d.ok, d.lam1, d.b1, d.c1, d.alpha, d.reason);
    end
    fprintf('  mu trajectory after switch-on: %s\n', mat2str(out{i}.mu_hist(out{i}.t_on:min(end, out{i}.t_on+30))', 3));
end
end
