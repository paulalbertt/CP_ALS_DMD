# CP_ALS_DMD

ALS for CP/PARAFAC with Dynamic Mode Decomposition (DMD) fixed-point extrapolation, compared
against the Bro/Harshman line search, the exact line search of Rajih et al. and Anderson
acceleration. See `REPORT.md` for the assessment, the list of fixes and the simulation results.

Quick start (MATLAB, no toolboxes needed):

```matlab
addpath(pwd); maxNumCompThreads(1);
run_cp_als_dmd_demo                      % one hard instance, all methods, plot in results/
[A, out] = cp_als_dmd(X, R, struct('accel', 'dmd'));   % DMD-accelerated ALS on your tensor
```

The simulation study: `benchmark_cp_als_accel` then `summarize_benchmark('results/benchmark_results.mat')`.
The DMD convergence diagnostic (rate, slow modes, forecast of remaining iterations, degeneracy indicator):
`[A, out] = cp_als_dmd(X, R, struct('monitor', true)); out.diag`, demo `swamp_diagnostic_demo`, accuracy study `forecast_evaluation`.
DMD-with-control damping that stops a two-factor degeneracy: `cp_als_dmdc(X, R, struct('control', 'dmdc'))`, demos `dmdc_damping_demo` and `dmdc_continuation_demo` (continuation along the regularisation path, and the run-watch-hold recipe).
Real EEM data study (nonnegativity, missing values): `real_data_test('aminoacids'|'sugar'|'dorrit'|'dreem_tutorial')`, then `summarize_real_data`. The datasets are read from the N-way toolbox and drEEM folders named in
`local_paths.m`: copy `local_paths_example.m` to `local_paths.m` and set the two paths (that file is git-ignored).
The nonnegative N-way toolbox variant: put the N-way toolbox and `addTensorNoise` on the path
(after this folder) and run `pf_dmd_compare`, or call `parafac(X, R, Options, const, Old, [], [], 'dmd')`.

`parafac.m` is a modified copy of the N-way toolbox function (Copyright 1995-2006 Rasmus Bro & Claus Andersson,
GNU GPL version 2 or later; the notice inside the file applies to it). The unmodified file is `old/parafac_v0.m`.
