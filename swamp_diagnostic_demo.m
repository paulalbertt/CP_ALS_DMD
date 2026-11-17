function swamp_diagnostic_demo()
%SWAMP_DIAGNOSTIC_DEMO  DMD eigenvalues of the ALS map as an online convergence diagnostic.
%
% Plain ALS (no acceleration) with opts.monitor = true on three regimes:
%   A  easy:       congruence 0.5, 1 % noise           fast linear convergence
%   B  swamp:      congruence 0.95, 1 % noise          slow but convergent, non-degenerate
%   C  degenerate: X = a o b o c2 + a o b2 o c + a2 o b o c (rank 3, border rank 2) fitted
%                  with R = 2, 1 % noise.  ALS runs into a two-factor degeneracy: two
%                  components become collinear with opposite signs, their weights diverge
%                  together and the fit never converges.
% The diagnostic read every iteration from the last 6 iterates: dominant DMD eigenvalue
% (the rate), forecast of the iterations until the relative-change stopping rule fires,
% which components carry the slow mode and their signed congruence.
% Writes results/fig_diagnostic.png and prints a short table.
maxNumCompThreads(1);
I = [30 30 30]; tol = 1e-8;
reg = struct('name', {}, 'X', {}, 'R', {}, 'max_iters', {});
[X, ~] = make_cp_tensor(I, 4, 0.5, 0.01, 1);
reg(end+1) = struct('name', 'A: easy (congruence 0.5)', 'X', X, 'R', 4, 'max_iters', 5000);
[X, ~] = make_cp_tensor(I, 4, 0.95, 0.01, 2);
reg(end+1) = struct('name', 'B: swamp (congruence 0.95)', 'X', X, 'R', 4, 'max_iters', 5000);
rng(3); nv = @(v) v / norm(v);
a = nv(randn(I(1), 1)); a2 = nv(randn(I(1), 1)); b = nv(randn(I(2), 1)); b2 = nv(randn(I(2), 1));
c = nv(randn(I(3), 1)); c2 = nv(randn(I(3), 1));
Xd = reshape(a * kron(c2, b)' + a * kron(c, b2)' + a2 * kron(c, b)', I);
Xd = Xd + 0.01 * norm(Xd(:)) / sqrt(numel(Xd)) * randn(I);
reg(end+1) = struct('name', 'C: degenerate (border rank 2, R = 2)', 'X', Xd, 'R', 2, 'max_iters', 2000);

cols = [42 120 214; 235 104 52; 27 175 122] / 255;
fig = figure('Color', 'w', 'Position', [60 60 1100 900], 'Visible', 'off');
tl = tiledlayout(4, numel(reg), 'TileSpacing', 'compact', 'Padding', 'compact');
fprintf('%-38s %6s %8s %8s %8s %10s %10s %8s\n', 'regime', 'iters', 'stop', 'rho@end', 'rate@end', 'fc_it@100', 'act_rem@100', 'congr@end');
for g = 1:numel(reg)
    o = struct('accel', 'none', 'monitor', true, 'monitor_forecast', false, 'max_iters', reg(g).max_iters, 'tol', tol, 'seed', 11);
    [~, out] = cp_als_dmd(reg(g).X, reg(g).R, o);
    d = out.diag; K = out.niters; k = (1:K)';
    capped = strcmp(out.stop, 'max_iters');
    k100 = min(100, K);
    fprintf('%-38s %6d %8s %8.4f %8.4f %10.0f %10d %8.3f\n', reg(g).name, K, out.stop, d.rho(K), d.rate_emp(K), d.iters_to_tol(k100), K - k100, d.pair_congr(K));
    % row 1: excess objective
    ax = nexttile(g); hold(ax, 'on');
    ex = max(out.f_hist - out.f_hist(end), eps);
    plot(ax, k, ex, '-', 'Color', cols(1, :), 'LineWidth', 2, 'DisplayName', 'f_k - f_{last}');
    set(ax, 'XScale', 'log', 'YScale', 'log', 'Box', 'off', 'YGrid', 'on', 'GridAlpha', 0.15);
    title(ax, reg(g).name, 'FontWeight', 'normal');
    if g == 1, ylabel(ax, 'f_k - f_{last}'); legend(ax, 'Location', 'southwest', 'Box', 'off'); end
    if capped, text(ax, 0.03, 0.08, 'stopped at cap', 'Units', 'normalized', 'FontSize', 9, 'Color', [0.3 0.3 0.3]); end
    % row 2: rates
    ax = nexttile(numel(reg) + g); hold(ax, 'on');
    plot(ax, k, min(d.rho, 1.05), '-', 'Color', cols(1, :), 'LineWidth', 2, 'DisplayName', 'DMD: max |\lambda|');
    plot(ax, k, d.rate_emp, '-', 'Color', cols(2, :), 'LineWidth', 1.2, 'DisplayName', 'empirical ||r_k|| / ||r_{k-1}||');
    yline(ax, 1, ':', 'Color', [0.4 0.4 0.4], 'HandleVisibility', 'off');
    set(ax, 'XScale', 'log', 'Box', 'off', 'YGrid', 'on', 'GridAlpha', 0.15, 'YLim', [0 1.1]);
    if g == 1, ylabel(ax, 'convergence rate'); legend(ax, 'Location', 'southwest', 'Box', 'off'); end
    % row 3: forecast vs actual remaining iterations
    ax = nexttile(2*numel(reg) + g); hold(ax, 'on');
    plot(ax, k, max(d.iters_to_tol, 1), '-', 'Color', cols(1, :), 'LineWidth', 2, 'DisplayName', 'DMD forecast: iterations to stop');
    if capped, ls = '--'; nm = 'actual remaining (run capped)'; else, ls = '-'; nm = 'actual remaining iterations'; end
    plot(ax, k, max(K - k, 1), ls, 'Color', cols(2, :), 'LineWidth', 1.5, 'DisplayName', nm);
    set(ax, 'XScale', 'log', 'YScale', 'log', 'Box', 'off', 'YGrid', 'on', 'GridAlpha', 0.15);
    if g == 1, ylabel(ax, 'iterations'); legend(ax, 'Location', 'southwest', 'Box', 'off'); end
    % row 4: slow-mode composition
    ax = nexttile(3*numel(reg) + g); hold(ax, 'on');
    top2 = sum(maxk(d.part, 2, 2), 2);
    plot(ax, k, d.pair_congr, '-', 'Color', cols(1, :), 'LineWidth', 2, 'DisplayName', 'signed congruence of the two slow-mode components');
    plot(ax, k, top2, '-', 'Color', cols(3, :), 'LineWidth', 1.5, 'DisplayName', 'share of the slow mode on those two components');
    yline(ax, -1, ':', 'Color', [0.4 0.4 0.4], 'HandleVisibility', 'off');
    set(ax, 'XScale', 'log', 'Box', 'off', 'YGrid', 'on', 'GridAlpha', 0.15, 'YLim', [-1.05 1.05]);
    xlabel(ax, 'iteration');
    if g == 1, ylabel(ax, 'two-factor degeneracy indicator'); legend(ax, 'Location', 'southwest', 'Box', 'off'); end
end
title(tl, 'DMD diagnostic of plain ALS (window 6), 30^3 tensors', 'FontSize', 11);
if ~exist('results', 'dir'), mkdir('results'); end
exportgraphics(fig, fullfile('results', 'fig_diagnostic.png'), 'Resolution', 150); close(fig);
fprintf('figure written to results/fig_diagnostic.png\n');
end
