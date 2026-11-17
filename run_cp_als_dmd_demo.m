function run_cp_als_dmd_demo()
%RUN_CP_ALS_DMD_DEMO  One hard synthetic CP problem, all accelerators from the same start.
%
% Generates a 50 x 50 x 50 rank-5 tensor with strongly collinear factors (congruence
% 0.9, a "swamp" for ALS) and 1 % noise, then runs plain ALS, the Bro line search, the
% exact line search of Rajih et al., Anderson acceleration and the DMD fixed-point
% extrapolation from one common random initialisation.  Prints a comparison table, the
% DMD eigenvalues of the first extrapolation attempts (the slow mode is the one with
% |lambda| close to 1) and saves a convergence plot to results/demo_convergence.png.
maxNumCompThreads(1);   % multithreaded BLAS is counter-productive for these small matrices
I = [50 50 50]; R = 5; congr = 0.9; noise_rel = 0.01; seed = 1;
[X, Atrue] = make_cp_tensor(I, R, congr, noise_rel, seed);
rng(seed + 100); A0 = arrayfun(@(n) randn(I(n), R), 1:numel(I), 'uni', 0);

methods = {'none', 'bro', 'els', 'aa', 'dmd', 'dmdbt'};
labels  = {'ALS', 'ALS + Bro line search', 'ALS + ELS (Rajih)', 'ALS + Anderson', 'ALS + DMD (ELS)', 'ALS + DMD (backtracking)'};
cols    = [42 120 214; 235 104 52; 27 175 122; 237 161 0; 232 123 164; 0 131 0] / 255;
out = cell(1, numel(methods)); fms = zeros(1, numel(methods));
fprintf('%-26s %7s %8s %8s %12s %7s %9s\n', 'method', 'iters', 'MTTKRP', 'time[s]', 'rel. fit err', 'FMS', 'accepted');
for i = 1:numel(methods)
    opts = struct('accel', strrep(methods{i}, 'dmdbt', 'dmd'), 'max_iters', 5000, 'tol', 1e-8, 'init', {A0}, 'store_eigs', true);
    if strcmp(methods{i}, 'dmdbt'), opts.linesearch = 'backtrack'; end
    [A, out{i}] = cp_als_dmd(X, R, opts);
    fms(i) = fms_score(A, Atrue);
    fprintf('%-26s %7d %8d %8.3f %12.3e %7.4f %5d/%-4d\n', labels{i}, out{i}.niters, out{i}.n_mttkrp, out{i}.time, ...
        sqrt(2*out{i}.fval)/norm(X(:)), fms(i), out{i}.n_accept, out{i}.n_try);
end
% DMD eigenvalues: the convergence modes of ALS near the solution
e = out{5}.eigs; ks = find(~cellfun(@isempty, e));
fprintf('\nDMD eigenvalues (estimated eigenvalues of the ALS Jacobian) at the first attempts:\n');
for k = ks(1:min(4, numel(ks)))'
    lam = e{k}; [~, o] = sort(abs(lam), 'descend');
    fprintf('  iteration %3d: %s\n', k, mat2str(lam(o).', 4));
end
% convergence plot
fb = min(cellfun(@(o) o.fval, out));
fig = figure('Color', 'w', 'Position', [100 100 760 330]);
tl = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
for panel = 1:2
    ax = nexttile; hold(ax, 'on');
    for i = 1:numel(methods)
        o = out{i}; ex = max(o.f_hist - fb, eps);
        if panel == 1, x = 1:o.niters; else, x = o.n_mttkrp * (1:o.niters) / o.niters; end
        plot(ax, x, ex, '-', 'Color', cols(i, :), 'LineWidth', 2, 'DisplayName', labels{i});
    end
    set(ax, 'YScale', 'log', 'XScale', 'log', 'Box', 'off', 'YGrid', 'on', 'GridAlpha', 0.15);
    if panel == 1, xlabel(ax, 'iteration (ALS sweep)'); ylabel(ax, 'f_k - f_{best}'); else, xlabel(ax, 'cost (MTTKRP equivalents)'); end
end
lg = legend(ax, 'Location', 'eastoutside', 'Box', 'off'); lg.Layout.Tile = 'east';
title(tl, sprintf('%s, R=%d, congruence %.2f, noise %g%%', mat2str(I), R, congr, 100*noise_rel), 'FontSize', 11);
if ~exist('results', 'dir'), mkdir('results'); end
exportgraphics(fig, fullfile('results', 'demo_convergence.png'), 'Resolution', 150);
fprintf('\nplot saved to results/demo_convergence.png\n');
end
