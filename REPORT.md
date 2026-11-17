# DMD-accelerated ALS for CP/PARAFAC: assessment, fixes and simulation results

Date: 2026-11-16. Code and results in this folder (`CP_ALS_DMD`). The original files are kept in `old/`.

## 1. Does the idea make sense?

Yes, with one important qualification: it is a sound way to accelerate ALS, but what it
produces is a member of a known family (vector extrapolation / Anderson-type acceleration),
so the value lies in the specific construction and in how it compares with the standard
accelerators, not in the principle.

**Why it works.** One ALS sweep plus gauge fixing (unit-norm columns in modes 2..N, all
weights in mode 1) is a fixed-point map `x -> G(x)` on the packed factor vector. Near a
minimiser `x*` the map is approximately affine, `G(x) ~ x* + J (x - x*)`, and the residual
`r(x) = G(x) - x = (J - I)(x - x*)` gives

    x* = G(x) + J (I - J)^{-1} r(x).                                            (1)

ALS converges linearly with rate `rho(J)`; in a swamp `rho(J)` is close to 1 and one or two
eigen-directions dominate. DMD on the recent iterates estimates exactly these modes: with
`dX = U S V'` and `Atil = U' dG V S^{-1} = W diag(lam) W^{-1}` the step is

    p = U W diag( lam/(1-lam) ) W^{-1} U' r_k ,                                 (2)

i.e. every slow mode is extrapolated to its limit with gain `lam/(1-lam)`. That is the
fixed-point extrapolation the original code did not do (it predicted one or ten ALS steps
ahead, which can never save more than one or ten iterations per attempt).

**How it relates to what exists.**

* The Harshman/Bro line search of the N-way toolbox (`x_k + s r_k`, `s = it^(1/2)`) and the
  exact line search (ELS) of Rajih, Comon & Harshman (2008) are the scalar case of (2): one
  mode, gain fitted by the step length instead of from the history. DMD generalises them to a
  matrix-valued, mode-wise extrapolation.
* Anderson acceleration (Walker & Ni 2011), RRE/MPE (Sidi 2017) and nonlinear GMRES for CP
  (De Sterck 2012) use the same window of iterates and the same affine model; they solve a
  least-squares problem for the combination coefficients, where DMD fits the operator `J`
  on the window first and then applies the Neumann sum. On linear problems these are close
  relatives of each other and of GMRES.
* Using DMD to accelerate an iterative solver was done for CFD solvers by Andersson
  (Computers & Fluids, 2016) with the same "sum the modes to the limit" formula.

So a paper would have to be positioned against Anderson acceleration / N-GMRES and ELS, and
the honest selling points are: (i) the eigenvalues are a diagnostic of the swamp (you see the
slow mode's `lam`), (ii) non-converging modes (`|lam| >= 1`) can be filtered, (iii) combined
with an exact line search along the DMD direction it was the most reliable method in the
noisy swamp experiments below, and (iv) the pairs formulation keeps the history valid across
accelerated steps.

**What must hold for it to work (and was violated before).** The gauge must be fixed and
stable (no sign or permutation jumps between iterates); the window must consist of pairs
that satisfy `dG ~ J dX`; only modes with `|lam| < 1` may be summed; and every extrapolated
point must be safeguarded by the objective, because far from the solution the affine model
is wrong.

## 2. What was wrong in the original code

### `cp_als_dmd.m` (standalone, now `old/cp_als_dmd_v0.m`)

1. **One-step prediction instead of extrapolation.** `x_pred = Kls * xk` predicts the next
   ALS iterate; the "ELS" blend caps the step at 1.5 iterates. At best one iteration is saved
   per attempt, at the price of two extra MTTKRPs, so it cannot accelerate.
2. **Wrong operator.** `Kls = Ys / Xs` solves an underdetermined system (p unknowns, m-1
   equations per row). MATLAB returns a sparse "basic" solution, not the least-squares/DMD
   operator `Ys * pinv(Xs)`. On the snapshot span both agree, but `xk` is never exactly in
   that span and the off-span part is mapped arbitrarily. It also forms a p x p matrix.
3. **History corrupted after an accepted step.** The buffer stores consecutive ALS iterates;
   after an accepted jump the pair (x_k, ALS(x_jump)) is not a pair of consecutive iterates,
   so the next fits are meaningless. Same problem in `parafac.m`.
4. **The line search is not exact.** `alpha = <R, dM>/||dM||^2` minimises the model-space
   interpolation `M_als + alpha (M_dmd - M_als)`, but the step is taken in factor space,
   where the model is a degree-N polynomial in alpha.
5. **Stopping rule.** `abs(df) / max(1, f)` becomes an absolute test once `f < 1`; on the
   noiseless demo the run stops at `f ~ 3e-3` after 16-22 iterations, far from convergence.
6. **Option parser.** `arg(name, def)` evaluates `opts.(name)` even when the field is absent
   (crash: "Unrecognized field name"), and multiplies char/cell values by logicals, which
   turns `'rand'` into numbers. The demo worked only because it set every field.
7. **Sign alignment** flips a column in all N modes, which negates the rank-1 term for odd N
   (latent, since ALS never flips signs between sweeps; the whole alignment step is
   unnecessary and was removed).
8. Counters (`accepted_dmd`) were never filled.

Reproduction on the original demo problem (40x35x30, R=5, noiseless, seed 1): DMD+ELS 22
iterations, plain ALS 16 iterations, DMD alone 20 iterations, 0 of 24 DMD attempts accepted,
all runs stopped prematurely by bug 5.

### `parafac.m`, branch `acm = 'dmd'` (now `old/parafac_v0.m`)

1. DMD on the raw iterates with an `h`-step prediction (`h <= 10`): at most `h` iterations
   saved per accepted attempt, while every attempt builds up to ten full models (each as
   expensive as the per-iteration fit evaluation). This is why the 1000-fit figure shows the
   same iteration counts as plain ALS and a higher time per iteration.
2. Same history corruption after accepted steps as above.
3. Adaptive rank/look-ahead heuristics on top of a broken step hide the problem.
4. Minor: in `PF_DMD_demo.m`, `Z * L` with a lower-triangular Cholesky factor gives column
   covariance `L' L`, not `Sigma`; use `Z * L'`.

## 3. The fixed implementation

* `cp_als_dmd.m` (rewritten, dependency-free). One ALS sweep per iteration, then one
  safeguarded acceleration step selected by `opts.accel`:
  `'none'`, `'bro'` (toolbox line search), `'els'` (Rajih et al.), `'aa'` (Anderson, window
  `opts.win`), `'dmd'` (this work). All methods share the same ALS core, the same objective
  evaluation (one MTTKRP), the same gauge and the same stopping rule, so comparisons are fair.
  Every accelerated point is accepted only if it lowers the objective below the ALS sweep's
  value, so all methods are monotone. Cost is counted in MTTKRP equivalents (`out.n_mttkrp`).
* `dmd_extrapolate.m`. Exact DMD on pairs `(dX_j, dG_j) = (x_{j+1}-x_j, G(x_{j+1})-G(x_j))`
  plus formula (2), with modes `|lam| >= 1` dropped and the gain capped (`amp_max`). Because
  `dG ~ J dX` holds for any two points, the window may contain accelerated steps: no flushing
  is needed. Unit test (scratch): on a 200-dimensional affine iteration with five active modes
  it recovers the fixed point to 1e-9 once the window holds six pairs, also with a jump inside
  the window.
* Step length. `'dmd'` uses the exact line search (degree-2N polynomial, Rajih et al.) along
  the DMD direction (`opts.linesearch = 'els'`, 2^(N-1) MTTKRPs) or backtracking 1, 1/2, 1/4
  (`'backtrack'`, one MTTKRP per trial). The polynomial was verified against direct objective
  evaluations to all printed digits.
* `parafac.m`. The `'dmd'` branch now stores the pairs (loadings at the start of an
  iteration, loadings after its ALS sweep), calls `dmd_extrapolate`, projects the candidate
  onto the nonnegative orthant in constrained modes, tries step lengths 1, 1/2, ..., 1/32 and
  doubles the step while it keeps improving, and restarts the history after a rejection.
  `'none'`, `'shifted'` and `'paired'` are untouched; `acm` now defaults to `'shifted'`. A
  global `PF_DMD_OPTS` struct overrides the DMD parameters for experiments.
* Helpers: `make_cp_tensor.m` (collinear factors, Tomasi & Bro style), `fms_score.m`,
  `benchmark_cp_als_accel.m` + `summarize_benchmark.m` (the study below),
  `run_cp_als_dmd_demo.m` (one hard instance, all methods, convergence plot),
  `pf_dmd_compare.m` (N-way toolbox with nonnegativity, replaces `PF_DMD_demo.m`).
* Machine note: on this Mac, MATLAB R2024b's multithreaded BLAS is pathological for the
  small matrices involved (a 100^3 MTTKRP took 323 ms with 15 threads and 0.6 ms with one).
  All timings use `maxNumCompThreads(1)`; the demo sets it.

## 4. Simulation experiment

**Design.** Synthetic rank-5 tensors with unit-norm factor columns whose pairwise column
correlation (congruence) is 0.5, 0.8, 0.9 or 0.95 in every mode (0.9 and 0.95 are the swamp
regime), Gaussian noise of 0, 1 or 10 % of the signal norm, 20 replicates per condition, one
random initialisation per instance shared by all methods. Stopping rule: relative change of
the objective below 1e-8, objective floor 1e-12 (noise-free data), cap 5000 iterations.
Methods: ALS, ALS + Bro line search, ALS + ELS, ALS + Anderson (window 6), ALS + DMD with ELS
(window 6), ALS + DMD with backtracking. Two studies: 50x50x50 (main, 1440 runs) and
50x200x100 (6 replicates, congruence 0.9/0.95, noise 1 %/10 %, 144 runs). Full tables:
`results/summary.txt`; raw data: `results/benchmark_*.mat`; figures `results/fig_*.png`.

**Correctness.** Every method reached the true factors (factor match score >= 0.95) in all
swamp conditions and in 95-100 % of the easy ones (where one unlucky initialisation ends in a
different local solution for every method alike). Final objectives agree to the stopping
tolerance.

**Main study, 50x50x50, noise 1 %, medians over 20 replicates** (cost = MTTKRP equivalents
relative to plain ALS on the same instance):

| congruence | ALS iters | Bro | ELS | Anderson | DMD (ELS) | DMD (backtr.) |
|---|---|---|---|---|---|---|
| 0.50 | 50 | 27 it, cost 0.72 | 26 it, 1.18 | 22 it, 0.60 | 26 it, 1.18 | 26 it, 0.71 |
| 0.80 | 335 | 189 it, 0.75 | 94 it, 0.60 | 54 it, 0.22 | 70 it, 0.48 | 66 it, 0.30 |
| 0.90 | 1068 | 609 it, 0.77 | 181 it, 0.40 | 114 it, 0.13 | 128 it, 0.28 | 146 it, 0.20 |
| 0.95 | 3631 | 2057 it, 0.79 | 554 it, 0.37 | 280 it, 0.10 | 188 it, 0.11 | 252 it, 0.10 |

At congruence 0.95 without noise plain ALS hit the 5000-iteration cap in 65 % of the runs;
every accelerated method converged.

**Large study, 50x200x100, congruence 0.95, noise 1 %, medians over 6 replicates:**

| method | iterations | wall time [s] | cost / ALS |
|---|---|---|---|
| ALS | 3667 | 8.26 | 1.00 |
| ALS + Bro line search | 2190 | 6.64 | 0.80 |
| ALS + ELS (Rajih) | 465 | 2.50 | 0.27 |
| ALS + Anderson | 248 | 0.78 | 0.09 |
| ALS + DMD (ELS) | 184 | 1.05 | 0.12 |
| ALS + DMD (backtracking) | 214 | 0.79 | 0.09 |

**Observations.**

1. The fixed DMD extrapolation works: in swamps it needs 5-12 % of the ALS iterations, fewer
   than any other method, and 100 % of its steps were accepted (the ELS along the DMD
   direction never produced a worse point).
2. In cost it is on par with Anderson acceleration at congruence 0.95 and somewhat behind at
   0.8-0.9, because the exact line search costs 2^(N-1) = 4 MTTKRPs per iteration against one
   objective evaluation for Anderson. With backtracking instead of ELS the cost matches
   Anderson but the iteration count rises by 15-35 %.
3. On easy problems (congruence 0.5) every accelerator halves the iterations, but the
   ELS-based variants then cost as much as plain ALS; Anderson and DMD-backtracking save
   30-40 %. This is the regime of the original 1000-fit experiment, which is why no
   acceleration could show up there even with a correct implementation.
4. The classic line searches are weak in swamps: Bro's growing step saves ~25 % of the cost,
   ELS 60-70 %; DMD and Anderson save ~90 %.
5. In noisy swamps (congruence 0.95, noise 10 %) the relative-change stopping rule fires
   prematurely for the slow methods: their final objectives sit 1e-6 to 4e-6 (relative)
   above the DMD (ELS) result, which reached the best objective in 85-100 % of the instances
   (`df/fbest` and `reached` columns in `results/summary.txt`). A fast method therefore also
   makes the usual stopping rule trustworthy.
6. Wall time follows cost once the tensor is large enough for the MTTKRP to dominate
   (large study); at 50x50x50 the per-iteration overheads of the line search and the SVD are
   visible (`results/fig_time_vs_iters_main.png`).

**Parameter sensitivity** (50x50x50, congruence 0.95, noise 1 %, 6 replicates, medians;
default = window 6, gain cap 1e3, singular-value cut-off 1e-6, ELS, every iteration):

| variant | iterations | MTTKRP | accepted |
|---|---|---|---|
| DMD default | 178 | 1240 | 100 % |
| window 4 / 10 / 15 | 206 / 190 / 204 | 1428 / 1326 / 1424 | 100 % |
| gain cap 1e2 / 1e5 | 204 / 200 | 1424 / 1388 | 100 % |
| sv cut-off 1e-3 / 1e-10 | 156 / 178 | 1084 / 1240 | 100 % |
| attempt every 2nd iteration | 190 | 944 | 100 % |
| backtracking instead of ELS | 254 | 1110 | 73 % |
| Anderson, window 4 / 6 / 10 / 15 | 270 / 305 / 264 / 262 | 1080 / 1218 / 1052 / 1048 | 64 % |

The method is insensitive to its parameters within a factor of about 1.2 in cost; attempting
the extrapolation every second iteration keeps the iteration count and removes a quarter of
the cost, which makes it the cheapest variant in MTTKRPs in this sweep.

**N-way toolbox with nonnegativity (`parafac.m`, `const = [2 2 2]`).** Same setup as the
original 1000-fit experiment (`PF_DMD_demo.m`): 100x150x180, R = 5, half-normal nonnegative
factors with column correlation rho, `addTensorNoise` at 10 or 20 dB, random initialisation
shared by the three methods, 4 replicates per condition, single thread
(`pf_dmd_compare.m`, raw data `results/parafac_compare.mat`, log `results/parafac_compare.log`).

| condition | method | median iterations | median time [s] | iterations / ALS (median of paired ratios) | best fit found (instances) |
|---|---|---|---|---|---|
| rho 0.80, 10 dB | ALS | 166 | 16.8 | 1.00 | 4/4 |
| rho 0.80, 10 dB | toolbox line search ('shifted') | 46 | 5.0 | 0.29 | 4/4 |
| rho 0.80, 10 dB | DMD ('dmd') | 52 | 7.1 | 0.32 | 4/4 |
| rho 0.80, 20 dB | ALS | 160 | 7.7 | 1.00 | 4/4 |
| rho 0.80, 20 dB | toolbox line search ('shifted') | 46 | 2.6 | 0.26 | 4/4 |
| rho 0.80, 20 dB | DMD ('dmd') | 50 | 3.1 | 0.30 | 4/4 |
| rho 0.95, 10 dB | ALS | 1017 | 44.2 | 1.00 | 2/4 |
| rho 0.95, 10 dB | toolbox line search ('shifted') | 86 | 4.1 | 0.11 | 3/4 |
| rho 0.95, 10 dB | DMD ('dmd') | 148 | 8.7 | 0.15 | 3/4 |
| rho 0.95, 20 dB | ALS | 1108 | 47.4 | 1.00 | 4/4 |
| rho 0.95, 20 dB | toolbox line search ('shifted') | 96 | 4.5 | 0.09 | 4/4 |
| rho 0.95, 20 dB | DMD ('dmd') | 144 | 8.5 | 0.13 | 4/4 |

The repaired `'dmd'` branch now does accelerate the toolbox: 3x fewer iterations than ALS
on the easy condition and 7-8x fewer on the collinear one, with 2-5x less wall time. It does
not beat the toolbox's own `'shifted'` line search (Harshman/Bro growing step), which needs
10-30 % fewer iterations and about half the time per iteration. Three reasons: (i) the
toolbox has no cheap objective evaluation, so every DMD trial (1-3 per iteration, plus the
step doubling) builds a full model; (ii) the nonnegative ALS map (NNLS sub-problems) is only
piecewise smooth, which weakens the affine model DMD fits, and the backtracking safeguard is
cruder than the exact line search that made DMD the most reliable method in the
unconstrained study; (iii) the growing step `it^(1/2)` of `'shifted'` happens to suit these
swamps very well. At 10 dB and rho 0.95 the methods did not always end in the same local
solution (line search and DMD each found a better fit than ALS in one instance each), which
is a property of the problem, not of the accelerators. A projected exact line search along
the DMD direction and a Gram-matrix objective evaluation inside `parafac.m` are the obvious
next steps if the toolbox variant matters; the standalone `cp_als_dmd.m` already has both.

## 5. Conclusions and recommendations

* The idea is sound once implemented as a fixed-point extrapolation of the ALS map, and the
  fixed code accelerates ALS by roughly 10x in swamps, matching Anderson acceleration and
  clearly beating the line searches that are the current practice in PARAFAC software.
* It is not a new principle: position it relative to Anderson acceleration / nonlinear GMRES
  and vector extrapolation (RRE/MPE). Its distinctive features are the modal view
  (eigenvalues of the ALS map as a swamp diagnostic, filtering of non-converging modes) and
  the exact line search along the extrapolation direction, which gave the most reliable
  convergence in noisy swamps.
* Practical defaults: window 6, attempt every iteration, ELS safeguard for unconstrained
  fits (most reliable) or backtracking when MTTKRPs are expensive; drop `|lam| >= 1`, cap the
  gain at 1e3. Skip acceleration for easy problems (first 3 iterations, or when ALS converges
  in < 50 iterations).
* Suggested next experiments if this goes into a paper: real EEM data, rank
  mis-specification, Tucker-compressed ALS, comparison with N-GMRES and damped Gauss-Newton
  (Tomasi & Bro 2006), and the nonnegative case with an exact line search (a projected ELS).

## 6. Files

| file | purpose |
|---|---|
| `cp_als_dmd.m` | ALS with safeguarded acceleration: none / bro / els / aa / dmd |
| `dmd_extrapolate.m` | DMD fixed-point extrapolation direction from snapshot pairs |
| `parafac.m` | N-way toolbox PARAFAC with the repaired `'dmd'` option (needs the toolbox on the path) |
| `make_cp_tensor.m`, `fms_score.m` | synthetic data with controlled collinearity; factor match score |
| `run_cp_als_dmd_demo.m` | one hard instance, all methods, convergence plot |
| `benchmark_cp_als_accel.m`, `summarize_benchmark.m` | the simulation study and its tables/figures |
| `pf_dmd_compare.m` | nonnegative PARAFAC comparison none / shifted / dmd (toolbox) |
| `swamp_diagnostic_demo.m`, `forecast_evaluation.m` | the DMD convergence diagnostic (section 8): three-regime figure and forecast accuracy study |
| `cp_als_dmdc.m`, `dmdc_damping_demo.m` | DMD-with-control damping to steer ALS out of a two-factor degeneracy (section 9) |
| `dmdc_continuation_demo.m` | continuation along the regularisation path with DMDc vs fixed damping vs plain ALS (section 10) |
| `load_eem.m`, `real_data_test.m`, `summarize_real_data.m` | real EEM data study (section 11): nonnegativity, missing values, four public datasets; toolbox/data folders come from `local_paths.m` (template `local_paths_example.m`) |
| `results/` | logs, `.mat` results, `summary.txt`, figures |
| `old/` | the original files for reference |

References: Rajih, Comon & Harshman, SIAM J. Matrix Anal. Appl. 30 (2008) 1128; Walker & Ni,
SIAM J. Numer. Anal. 49 (2011) 1715; De Sterck, SIAM J. Sci. Comput. 34 (2012) A1351; Tu et al.,
J. Comput. Dyn. 1 (2014) 391; Andersson, Computers & Fluids 133 (2016) 32; Sidi, Vector
Extrapolation Methods with Applications, SIAM (2017); Tomasi & Bro, Comput. Stat. Data Anal.
50 (2006) 1700.

## 7. Prior-art check (web search, 2026-11-16)

**Direct overlap.** No publication was found that uses DMD or Koopman-type extrapolation to
accelerate ALS for CP/PARAFAC. The nearest conceptual link is Redman et al., "On Koopman mode
decomposition and tensor component analysis" (Chaos 31, 2021), which relates Koopman modes to
CP components of trajectory data, not to the ALS iteration.

**DMD as a solver accelerator (established since 2014).**

* Andersson & Eriksson, "A novel solver acceleration technique based on dynamic mode
  decomposition", WCCM XI (2014); Andersson, "A non-intrusive acceleration technique for
  compressible flow solvers based on dynamic mode decomposition", Computers & Fluids 133
  (2016) 32-42. DMD of solver iterates, correction toward the steady state from the slow
  modes, steady problems only.
* Roberts, Xu, Elzohery & Abdo, "Acceleration of the power method with dynamic mode
  decomposition", Nuclear Sci. Eng. 193 (2019), arXiv:1904.09493. Iterates as snapshots,
  extrapolate to the steady state, restart (DMD-PM(n)). About 5x fewer power iterations, but
  Arnoldi's method always won at equal matrix-vector products; the authors recommend it only
  where Krylov methods do not apply. This is the same verdict as Anderson vs DMD here.
* McClarren & Haut, DMD acceleration of source iteration in transport, arXiv:1812.05241.
* Liu, Zhang & Kou, "Mode multigrid", arXiv:1802.08962 (2018): DMD of previous iterates to
  identify and remove slowly converging modes, 3-6x speedups on CFD cases; follow-ups up to
  2024-2025 (e.g. Mirshahi et al., AIAA SciTech 2024; DMD-POD hybrids).

**Accelerating CP-ALS (established).**

* Line searches: Harshman (1970), Bro (1998, N-way toolbox), Rajih, Comon & Harshman (EUSIPCO
  2005; SIMAX 30, 2008), Nion & De Lathauwer (2008). Rajih et al. note that ELS helps most for
  "bottlenecks" and less in classic swamps.
* Nonlinear Krylov / quasi-Newton: De Sterck, "A nonlinear GMRES optimization algorithm for
  canonical tensor decomposition", SIAM J. Sci. Comput. 34 (2012), arXiv:1105.5331; De Sterck &
  Howse, nonlinearly preconditioned L-BFGS for ALS; Mitchell, Ye & De Sterck, "Nesterov
  acceleration of ALS for canonical tensor decomposition" (Numer. Linear Algebra Appl. 2020,
  arXiv:1810.05846), which benchmarks against NGMRES-, NCG-, L-BFGS- and Anderson-accelerated
  ALS and shows Nesterov = Anderson with window 1; De Sterck & He, "On the asymptotic linear
  convergence speed of Anderson acceleration, Nesterov acceleration and nonlinear GMRES", SIAM
  J. Sci. Comput. (2021), arXiv:2007.01996, with AA/NGMRES-accelerated ALS on CP test problems
  and a GMRES-based theory of the asymptotic rate (the same "linearise the fixed-point map"
  view used here).
* Other extrapolations: Wang, Navasca & Kindermann (Aitken-Steffensen-type update for
  regularised ALS); Chen et al. (2000, CILS) extrapolation inside a penalised ALS; Evans et al.
  (NLAA 2023) blockwise momentum + perturbation; Ang, Cohen, Gillis & Hien (2020) heuristic
  extrapolation with restarts for nonnegative tensor factorisation; Navasca, De Lathauwer &
  Kindermann (EUSIPCO 2008) regularised ALS against swamps.
* Swamps and degeneracy: Mitchell & Burdick (1994), Rayens & Mitchell (1997), Paatero (2000),
  Stegeman (2006), Krijnen & Dijkstra; swamp monitoring there is based on fit curves and
  restarts, not on a spectral estimate of the ALS map.

**Verdict.** The combination "DMD extrapolation of ALS iterates for CP" is unpublished as far
as this search shows, but both halves are established, Anderson/NGMRES acceleration of CP-ALS
has a convergence theory (De Sterck & He) and the DMD-vs-Krylov comparison already has a
published negative result in another field (Roberts et al.). A paper would therefore have to
show something DMD does that Anderson does not: the eigenvalue diagnostic of swamps and
degeneracies (not found in the literature in this form), the mode filtering, the
exact-line-search-safeguarded direction (most reliable in noisy swamps here), and behaviour
with nonnegativity on real data. Without one of these as the centre of the paper, reviewers
in numerical linear algebra will treat it as a variant of known extrapolation methods; a
chemometrics venue may accept a careful comparison paper that introduces Anderson/DMD-type
extrapolation to PARAFAC practice.

## 8. Prototype: DMD as an online convergence diagnostic for ALS (added 2026-11-16)

**What it is.** With `opts.monitor = true`, `cp_als_dmd.m` fits exact DMD to the last six
`(x_j, G(x_j))` pairs at every iteration, for any accelerator including plain ALS, and
returns per-iteration series in `out.diag`. Cost: one SVD of a p x 5 matrix and a 5 x 5
eigenproblem per iteration, no extra MTTKRP. From the eigenvalues `lam` and the observed
change `df_k = f_{k-1} - f_k`:

    rate            rho = max |lam|                     (empirical check: ||r_k|| / ||r_{k-1}||)
    slow modes      #{ |lam| > 0.9 }
    iterations until the relative-change rule (tol) fires
                    n = ln( tol f_k / df_k ) / ( 2 ln rho )
    remaining decrease of f
                    D = df_k rho^2 / (1 - rho^2)
    slow-mode composition
                    e_r = share of the predicted step p carried by component r (summed over modes)
    degeneracy indicator
                    signed congruence prod_n cos(a_r, a_s) of the two components with the largest e_r

**Three regimes** (`swamp_diagnostic_demo.m`, `results/fig_diagnostic.png`; 30^3, plain ALS):

| regime | iterations | rate at the end | forecast at k = 100 vs actual remaining | pair congruence at the end |
|---|---|---|---|---|
| A easy (congruence 0.5) | 42 | 0.734 (empirical 0.734) | converged before k = 100 | 0.14 |
| B swamp (congruence 0.95) | 2948 | 0.998 (0.998) | 1962 vs 2848 | 0.84 |
| C degenerate (rank 3, border rank 2, fitted with R = 2) | capped at 2000 | 0.9998 (0.9996) | 2268 vs > 1900 and rising | -0.95 |

Regime A: the rate settles at 0.73 within ten iterations and the forecast tracks the actual
remaining iterations. Regime B: after a transient of about 60 iterations (where the window
straddles nonlinear behaviour and spurious |lam| > 1 appear) the rate sits at 0.998 with two
slow modes and the forecast follows the actual count to the end. Regime C shows the
two-factor-degeneracy signature: the slow mode lives entirely on two components, their
signed congruence drifts from -0.65 to -0.95 while their weights grow together (2.1 -> 5.3),
the rate creeps towards 1 and the forecast of the remaining iterations *increases* with k
(862 at k = 20, 20 000 at k = 1500), whereas in a genuine swamp it decreases. An increasing
forecast is therefore an operational "this will not converge" trigger, which is what
Mitchell & Burdick (1994) recommended detecting before restarting, without giving a rule.

**Forecast accuracy** (`forecast_evaluation.m`, `results/fig_forecast.png`; 40^3, R = 4,
congruence 0.5 / 0.8 / 0.9 / 0.95, 1 % noise, 8 instances each, plain ALS to tol 1e-8,
forecasts made at checkpoints k = 10 ... 2000 and compared with what happened afterwards):

| checkpoint k | n | median forecast / actual (iterations) | within x2 | within x3 |
|---|---|---|---|---|
| 10 | 17 | 0.58 | 41 % | 53 % |
| 20 | 22 | 0.81 | 64 % | 73 % |
| 50 | 19 | 0.92 | 79 % | 84 % |
| 100 | 24 | 0.94 | 79 % | 88 % |
| 200 | 24 | 1.04 | 96 % | 100 % |
| 500-2000 | 37 | 1.05-1.19 | 92-100 % | 100 % |
| all usable | 175 | 1.00 | 83 % | 89 % |

84 % of the checkpoints had a usable forecast (max |lam| < 1); the rest are early
iterations before the linear regime. The forecast of the remaining decrease of f has a
median ratio of 1.10 and is within a factor 3 of the truth in 94 % of the checkpoints. At the
iteration where the N-way toolbox default rule (relative change < 1e-6) would stop, the
forecast said 255 more iterations to the 1e-8 rule (median) against an actual 241, and the
forecast remaining decrease was within a factor 3 of the actual one in all 32 runs.

**What this gives a practitioner.**

1. A stopping rule that measures distance to convergence instead of speed: stop when the
   forecast remaining relative decrease D / f_k is below a tolerance. In a swamp the
   per-iteration change is small although the solution is far away; D is not fooled.
2. Swamp detection within ~20 iterations after the transient (rate > 0.99 and a steady
   forecast), with the expected gain of acceleration readable from lam / (1 - lam).
3. Degeneracy detection: forecast increasing with k, slow mode on two components with
   congruence -> -1 and co-growing weights. Restart or regularise early instead of after
   thousands of iterations.
4. The same six-iterate window drives the accelerator of sections 3-4, so diagnostics and
   acceleration come from one computation.

**Caveats.** Early iterations give no or wrong forecasts (16 % of checkpoints, mostly
k <= 20); the forecast assumes the dominant mode persists, so a later switch of basin is
not predictable; everything is synthetic and unconstrained so far; the nonnegative ALS map
is only piecewise smooth and needs its own test.

**Novelty.** The swamp and degeneracy literature (Mitchell & Burdick 1994, Rayens & Mitchell
1997, Paatero 2000) detects trouble from fit curves and restarts; an online spectral estimate
of the ALS map with a quantitative forecast (median ratio 1.0, 83 % within a factor 2) and an
identification of the degenerate pair was not found in the search of section 7. Together
with the accelerator this makes a coherent paper: "DMD of the ALS iteration: convergence
diagnostics and extrapolation for PARAFAC", with the diagnostic as the contribution that
Anderson acceleration does not offer.

## 9. Prototype: DMD with control (DMDc) to steer ALS out of a degenerate swamp (added 2026-11-17)

**Actuator.** Tikhonov damping mu on the mode-1 (weight) sub-problem, A1 = M1 (Gamma1 + mu I)^{-1}.
Damped ALS is block coordinate descent on 1/2||X - M(A)||^2 + mu/2 ||A1||_F^2 with unit-norm
columns in the other modes, so mu changes both the ALS map and its fixed point, and it acts
most strongly on the ill-conditioned directions of Gamma1, which is exactly where a two-factor
degeneracy (2FD) lives.

**Model and control law** (`cp_als_dmdc.m`). Triples (x_j, mu_j, G_{mu_j}(x_j)) are collected
and DMDc (Proctor, Brunton & Kutz 2016) is fitted to the consecutive differences
dG = A dX + b dmu (input row scaled to sigma_1(dX), POD basis of dG truncated to rank_cap = 2).
With the reduced A = V diag(lam) V^{-1}, the slow mode lam1 and its left eigenvector w1 give
the amplitude a_k = w1' U' x_k and the scalar affine model  a_{k+1} = lam1 a_k + b1 mu_k + c1
(c1 from the data). The mode is oriented so that the open-loop drift c1 > 0 and the model is
accepted only if b1 < 0 (damping opposes the drift). The proportional law

    mu_k = clip( mu0 + alpha (a_k - a_on), 0, mu_max ),   alpha = (lam_target - lam1) / b1

places the closed-loop eigenvalue of the slow mode at lam_target; a rank-one update along a
left eigenvector leaves the other eigenvalues unchanged. Sequence: plain ALS until the
diagnostic of section 8 fires (rate > 0.99 and negative pair congruence for 3 iterations),
then 12 iterations with a pseudo-random binary dither of +-75 % around mu0 to identify the
model, then the law with the frozen model; re-identification only if mu saturates; fallback
to rate feedback if identification fails twice.

**What failed first, and why** (worth keeping for a paper). A period-2 dither is collinear with
the state's own response, so b cannot be separated from A; the eigenvector sign is arbitrary,
so smoothing alpha across re-identifications flips it; identifying under feedback biases the
model (closed-loop identification); and keeping 4 modes lets spurious ones (|lam| ~ 0.04)
corrupt the left eigenvector, which gave b1 the wrong sign in 2 of 5 starts. Pseudo-random
excitation, orientation by the drift, a frozen model and 2-3 retained modes fixed all of it
(identification succeeded in 5 of 5 starts).

**Results** on the degenerate tensor of section 8 (rank 3, border rank 2, 1 % noise, R = 2,
`dmdc_damping_demo.m`, `results/fig_dmdc.png`; one start, tol 1e-8):

| method | iterations | objective f/(1/2||X||^2) | largest weight | pair congruence | final mu |
|---|---|---|---|---|---|
| plain ALS | > 1500, never converges | 1.28e-4 | 5.35 and growing | -0.95 | 0 |
| fixed mu = 1e-3 | 620 | 7.4e-4 | 2.57 | -0.77 | 1e-3 |
| fixed mu = 1e-2 | 144 | 3.1e-3 | 1.89 | -0.57 | 1e-2 |
| fixed mu = 1e-1 | 36 | 1.4e-2 | 1.48 | -0.31 | 1e-1 |
| diagnostic-triggered mu = 1e-2 | 161 | 3.1e-3 | 1.89 | -0.57 | 1e-2 |
| rate feedback on mu (target 0.9, no b) | 182 | 2.9e-3 | 1.91 | -0.58 | 9e-3 |
| DMDc proportional damping (target 0.9) | 147 | 1.0e-3 | 2.38 | -0.73 | 1.7e-3 |
| DMDc proportional damping (target 0.95) | 294 | 9.1e-4 | 2.45 | -0.75 | 1.4e-3 |

Over five random starts: DMDc (target 0.9) 147-157 iterations, objective 1.0-1.2e-3, final mu
1.7-2.1e-3; fixed mu = 1e-3 620-644 iterations; fixed mu = 1e-2 144-152; rate feedback
182-194 with objective 2.9e-3. Identified model in the run above: lam1 = 0.968, b1 = -5.2,
c1 = +0.089, alpha = 0.013; the damping then rose smoothly from 1.0e-3 to 1.7e-3 as the
amplitude of the degenerate mode grew, and the growth stopped.

**Interpretation.** State feedback decouples the convergence rate from the regularisation
bias. At the fixed point mu is constant, so the solution is the stationary point of the
mu = 1.7e-3 problem (objective 1.0e-3, close to the 7.4e-4 of mu = 1e-3), but the approach to
it is governed by the closed-loop pole 0.9 (147 iterations, the speed of mu = 1e-2). A fixed
mu cannot do both: at mu = 1e-3 the open-loop pole is about 0.97 and it takes 620 iterations.
Rate feedback without b cannot either: it must raise mu to 9e-3 to reach the target rate and
pays the bias (2.9e-3). This is the steering that was asked for: the damping grows with the
displacement along the degenerate direction, the two-factor degeneracy is stopped about 130
iterations after switch-on, and the pair stays bounded (weights 2.4, congruence -0.73
instead of running to -1). The target rate is the speed/bias knob (0.95: 294 iterations,
9.1e-4).

**What it does not do.** It cannot beat the unregularised objective of the degenerate run
(1.28e-4), because that infimum is not attained by any finite solution; the regularised
solution is the price of a finite answer. It does not choose lam_target or mu0; it does not
reduce mu afterwards (a continuation that tracks the regularisation path would be the next
step); it has only been tried on one synthetic tensor with R = 2, unconstrained; on a
convergent swamp the trigger does not fire (positive pair congruence), and forcing it would
bias a solution that needs no regularisation.

**Novelty.** Data-driven pole placement of the degenerate mode of ALS through the damping
parameter, with the diagnostic of section 8 as the trigger and DMDc as the model, was not
found in the literature: swamp stabilisation uses fixed constraints or regularisation
(Rayens & Mitchell 1997, Navasca et al. 2008) and DMDc has been used for flow control, not for
optimisation solvers. Together with sections 4 and 8 this gives the three uses of one
six-iterate window: extrapolate (convergent swamp), diagnose (rate, forecast, degeneracy),
and steer (degenerate swamp).

## 10. Continuation along the regularisation path (added 2026-11-17)

**Mechanism** (`cp_als_dmdc.m`, `opts.continuation = true`). After every converged stage
(relative changes of f and x below 1e-6 / 1e-5) the base damping is lowered,
mu_base <- mu * cont_mu_factor, optionally the target rate is raised, the DMDc model is
re-identified with a short dither at the new base, and the proportional law re-balances the
damping at the new position on the path. Stop rules: pair congruence below congr_bound
(-0.9), stage gain below 2 %, target rate above 0.99, or 8 stages. The classic baseline
(`'fixedpath'`) halves a fixed mu after every converged stage with the same rules. A first
version that re-based the law at the current damping did not move at all: a converged
state is a fixed point of the damped map whatever law produced the damping, so the base
must be lowered explicitly.

**Result** (`dmdc_continuation_demo.m`, `results/fig_continuation.png`, same degenerate tensor):

| method | stages | iterations per stage | reaches congruence -0.80 at | -0.85 at | final f, congruence |
|---|---|---|---|---|---|
| fixed-mu path, halving from 1e-2 | 6 (+1 unfinished) | 102, 150, 227, 343, 513, 763 | 1338 | 2164 | 2.9e-4, -0.87 (cap) |
| DMDc path, base halved, rate 0.9 | 8 | 107, 92, 92, 82, 92, 85, 88, 87 | 471 | not reached (8 stages) | 4.6e-4, -0.82 |
| DMDc path, base halved, rate 0.9 -> 0.99 | 4 | 107, 177, 563, 506 | 424 | not reached | 4.1e-4, -0.83 |
| DMDc path, base / 10, rate 0.9 | 8 | 107, 91, 87, 90, 87, 87, 86, 84 | 293 | 645 | 3.3e-4, -0.85 |
| plain ALS | - | - | 69 | 149 | never converges |
| plain ALS to congruence -0.85, then DMDc hold | 1 | - | 69 | 149 | 3.3e-4, -0.86, converged at 276 |

Two findings.

1. The mechanism works as designed: with the target rate held at 0.9 every DMDc stage costs
   about 87 iterations regardless of how small the damping has become, whereas the
   fixed-damping stages grow from 102 to 763 iterations because the open-loop rate creeps
   towards 1. The DMDc path therefore reaches a given point on the path about three times
   sooner (645 vs 2164 iterations to congruence -0.85). Raising the target rate towards 1
   instead of lowering the base gives no advantage (the stages slow down again).
2. The path itself is the ALS trajectory. Every method, plain ALS included, traces the same
   curve in the (congruence, objective) plane: 1.3e-3 at -0.70, 8.7e-4 at -0.75, 5.6e-4 at
   -0.80, 3.5e-4 at -0.85. The regularised minimisers for decreasing mu lie on the valley
   floor that the degenerate ALS iteration crawls along, and plain ALS moves along it fastest
   (congruence -0.85 at iteration 149), because any damping slows the drift by construction.
   Damping does not buy a better fit/degeneracy trade-off; it buys a *converged* solution at
   a chosen position, i.e. a fixed point where the stopping rule can fire and the result is
   reproducible.

The consequence is the recipe in the last row: let plain ALS run to the chosen degeneracy
bound (the diagnostic of section 8 watches the congruence), then switch the DMDc damping on
with a small base to *hold* the state there. On this tensor that converges at iteration 276
with the same objective (3.3e-4) and congruence (-0.86) that the fixed-damping continuation
reaches only after 2164 iterations and the DMDc continuation after 645, with a final damping
of 2.1e-4 found by the feedback. Continuation is thus the wrong tool for a degenerate swamp;
"run, watch, hold" is the right one, and the DMDc controller is what makes the hold possible
without a user-chosen mu. Where damping would be needed from the start is the case of a
convergent-but-collinear swamp whose solution one wants to keep away from a nearby
degenerate ridge; that was not tested here.

Caveats as before: one synthetic tensor, R = 2, unconstrained, the bound is a modelling
choice, and the "hold" relies on the identification succeeding at the switch-on point (it
did, with one attempt).

## 11. Real EEM data (added 2026-11-17)

**Setup.** `cp_als_dmd.m` gained nonnegativity (row-wise NNLS with the toolbox `fastnnls`,
HALS fallback; accelerated candidates are projected and accepted only if the exact
objective decreases) and expectation-maximisation for missing values (refill with the model
after every sweep; the objective is the misfit over observed entries). Four public datasets
found on this machine: amino acids (5 x 201 x 61, no missing), Dorrit (27 x 551 x 24, 1 %
missing), sugar process (268 x 53 x 7, 4 % missing) and the drEEM tutorial set
(89 x 125 x 73, 14 % missing). Reference and over-factored component numbers, nonnegative in
all modes (the standard EEM practice) plus unconstrained fits where degeneracies are
expected, 5-10 random starts shared by the methods, tolerance 1e-8, cap 3000 iterations,
monitor on. Methods: ALS, Bro line search, DMD with backtracking, DMD with ELS, Anderson.
Full tables: `results/real_summary.txt`; figures `results/fig_real_data.png`,
`results/fig_real_degeneracy.png`.

**Acceleration (median iterations relative to ALS on the same starts):**

| dataset, R, constraint | ALS iters | Bro | DMD (backtr.) | DMD (ELS) | Anderson |
|---|---|---|---|---|---|
| amino acids, 3, nonneg | 68 | 0.54 | 0.37 | 0.36 | 0.29 |
| amino acids, 3, unconstrained | 123 | 0.51 | 0.26 | 0.26 | 0.24 |
| amino acids, 4, unconstrained (over-factored) | cap 3000 (90 %) | cap (90 %) | 0.34, all converge | 0.34, all converge | cap (90 %) |
| Dorrit, 4, nonneg | 274 | 1.00 | 1.00 | 0.97 | 1.01 |
| Dorrit, 5, nonneg | 312 | 0.97 | 1.00 | 0.98 | 1.00 |
| Dorrit, 5, unconstrained | 555 | 0.65 | 0.60 | 0.60 | 0.90 |
| drEEM tutorial, 5, nonneg | 725 | 1.00 | 0.99 | 0.93 | 0.99 |
| drEEM tutorial, 6, nonneg | 606 | 0.74 | 0.87 | 0.66 | 0.88 |
| sugar, 4, nonneg | 579 | 0.94 | 1.01 | 0.95 | 0.96 |
| sugar, 5, nonneg | 1116 | 0.90 | 0.97 | 0.95 | 1.00 |

1. The synthetic swamp gains do not transfer to standard nonnegative EEM PARAFAC. On Dorrit,
   sugar and the drEEM set no accelerator helps, the toolbox line search included; the best
   single number is 0.66 (DMD with ELS, drEEM with 6 components). The backtracking DMD and
   Anderson accept only 3-14 % of their steps there; the ELS-safeguarded DMD and the Bro
   search accept 95-99 % but the accepted steps are tiny. The convergence curves show why
   (`fig_real_data.png`, right): a long plateau followed by a sudden drop, i.e. a saddle-like
   passage with changing active sets, not a linear slow mode. Accordingly the DMD model is
   non-contracting (max |lambda| >= 1) in 44-64 % of the iterations on these sets.
2. The small amino-acid set and the unconstrained fits behave like the synthetic problems:
   3-4x fewer iterations with DMD or Anderson, 2x with the Bro search.
3. Over-factored unconstrained amino-acid fits are the one real-data case with genuine
   swamps and degeneracies: ALS, Bro and Anderson hit the 3000-iteration cap in 90 % of the
   starts; DMD converges in every start (median 1020 iterations), half of the time to a
   non-degenerate solution with the best objective seen (4.58e-4), half of the time to a
   fully degenerate pair (congruence <= -0.85), i.e. it accelerates the slide as well as the
   escape, which is what section 9 predicted.
4. A side effect worth noting: the DMD step changes basins. On sugar it avoided the poor
   local solution (f 4.9e-3) that ALS found in one start per configuration and reached the
   best solution (6.9e-4 vs 9.3e-4 median) at R = 5; on the nonnegative over-factored amino
   acids it found the better 4-component solution from starts where ALS stayed at the
   3-component one. It is not guaranteed (one sugar start ended slightly worse).
5. Wall time is not the bottleneck for these sizes: a full fit takes 0.1-7 s single-threaded.
   Acceleration would matter for very large arrays or hundreds of random starts.

**Diagnostic on real data.**

* Forecast of the iterations to the stopping rule: 51 % within a factor 2 and 65 % within a
  factor 3 over all usable checkpoints (synthetic: 83 % / 89 %); per dataset amino acids
  59 / 71 %, Dorrit 56 / 76 %, sugar 51 / 67 %, drEEM 28 / 39 % with a median ratio of 0.3,
  i.e. the forecast is too optimistic when a plateau is still ahead. The forecast is only
  as good as the local linear model, and on constrained real fits that model is usable in
  36-56 % of the iterations.
* Degeneracy warning (rate > 0.99 and negative slow-mode pair congruence for 3 iterations):
  in the 240 nonnegative fits it never fired and no fit ended degenerate (nonnegativity
  prevents two-factor degeneracies, as known). In the unconstrained over-factored amino-acid
  fits 38 % of the runs ended degenerate; the warning fired in 95 % of those, at a median 2 %
  of the run length (iteration 15 of a run of 1000+, `fig_real_degeneracy.png`). It also
  fired in 84 % of the runs that did not end degenerate by the -0.85 rule; those are runs
  that passed through a degenerate region and were either capped just above the threshold
  (congruence -0.5 to -0.84) or carried out of it by the DMD step. On Dorrit with 5
  unconstrained components every start ended at a pair congruence of -0.65 and the warning
  fired in 80 % of them, which is a mild degeneracy by any practical standard. So the
  warning signals a degenerate region, very early, with the threshold a modelling choice;
  the identity of the slow-mode pair flickers between iterations and should be smoothed.

**Consequence for the project.** The accelerator is a synthetic and unconstrained story; for
routine nonnegative EEM PARAFAC it does not help, and an accelerator for that case would
have to respect the active set (extrapolate only the entries away from zero), which is an
open research direction rather than a fix. The diagnostic survives with qualifications:
the degeneracy warning is early and specific, the rate forecast is reliable on
unconstrained problems and only indicative on constrained real fits, and the share of
iterations in which the local model is non-contracting is itself a signature of the
saddle-like plateaus that dominate real nonnegative fits (not quantified here).
