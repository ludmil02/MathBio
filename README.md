# MathBio

Next generation matrix of a meta-population COVID model with vaccination, and
the optimization of its spectral radius over the vaccination rates.

The model has `m = 4` locations (Northeast, Midwest, South, West) and `n = 4`
age groups (0–19, 20–39, 40–59, 60+).  Each of the sixteen groups carries a
susceptible and a vaccinated-or-boosted channel, so the next generation matrix
`NGM` is `2mn = 32` by `32`, built from `m^2` blocks of size `2n` by `2n`,

    NGM[i, p] = Omega(chi)[i] . ccMat[i, p] . Psi_I[p] . Q .

Only `Omega(chi)` depends on the vaccination rates `chi1, chi2, chi3`; the rest
is the constant factor `Ac`.

## Contents

| directory | |
| --- | --- |
| `Mathematica_Wolfram/` | the notebooks, `settings_zf_jg.wl` and `ref_h.wls` |
| `code/python/` | the python, its `*_results.txt` and `intermediate/` |
| `notes/` | the main document and `bib_mb.bib`; `notes/figures/` the figures as png, with the `fig_*.dat` data of the figures in `notes/figures/dat/`; `notes/suplemental/` the supplement and the superseded three-dose note |
| `old_files/` | superseded material |

### Notebooks

In `Mathematica_Wolfram/`.

| file | |
| --- | --- |
| `settings_zf_jg.wl` | the settings shared by the two notebooks below: parameters, data, `mchi1 = mchi2 = mchi3 = 0`, and the construction of `NGM`.  Taken from the input cells of `eigenvalues_H_ZF_JG.nb`; it gives the same `NGM` to the last bit.  Each notebook loads it in its first input cell with `Get[FileNameJoin[{NotebookDirectory[], "settings_zf_jg.wl"}]]`, so a parameter changed here changes both. |
| `eigenvalues_H_ZF_JG.nb` | loads `settings_zf_jg.wl`, checks `NGM` (dimensions, eigenvalues, Perron vectors) and ends with John's method: the dominant eigenvalue and the age profile `l0` of the dominant left eigenvector, plotted per location.  At `chi = 0` it gives `rho = 4.423068109190629`.  One `Export["NGM.csv", NGM]` writes into the working directory when evaluated.  It is the reference of `smoke_zf_jg.py` and `reduced_test.py`.  The section with the stored allocation `OptimalChi1` (weekly `b1`, `rho = 2.9247935891`) and the diagnostic cells after it were removed on 2026-10-17; `OptimalChi1` is kept in `reduced_test.py` as `OPT_H`. |
| `min_rho_single_dose_H_ZF_JG.nb` | loads `settings_zf_jg.wl`, computes `l0` as John's method does, and minimizes `rho(NGM)` for a single dose with the weekly budget `b1` (row sums of `C2`) and `Sstar`, the susceptibles at `t = 347`: interior point method on the product of simplices, then quasi-Newton on the groups that receive doses, with the gradient (3.20) from one Perron pair of `NGM` and one of its transpose.  Gives `rho = 2.905938287779`, the 60+ group at zero in every location, and the optimality conditions; John's allocation, `l0` scaled to spend the same `b1`, gives `2.931231776312`.  Plot `f12` of the minimizing `chi1`. |
| `16_Group_Model_Results_t=347_ZF.nb` | the model solution at `t = 347`: `mS`, `mV`, `mP`, the `psi` and `s` factors, and `C2`, the primary-series doses by location and age group.  It no longer writes anything to disk. |

### Python

In `code/python/`.  Run the scripts from `code/python/`: the imports are between files of that
directory, and each `*_results.txt` is written to the working directory.  The
csv files under `old_files/`, the files of `Mathematica_Wolfram/` and the
`fig_*.dat` files in `notes/figures/dat/` and the figures in `notes/figures/` are located from the script's own path.

The python reproduces `eigenvalues_H_ZF_JG.nb` and carries the optimization
that `min_rho_single_dose_H_ZF_JG.nb` repeats in Wolfram.

| file | |
| --- | --- |
| `zf_jg_params.py` | the notebook's parameters, exported through a Wolfram kernel so every float round-trips.  `default_params()`. |
| `zf_jg.py` | the model: `dfe_fractions`, `compute_yz`, `q_matrix`, `cc_block`, `ac_matrix`, `omega_matrix`, `ngm_blocks`, `ngm_factored`, `ngm_reversed`, `spectrum`, `dominant` and `john_method`.  Sparse throughout; eigenvalues come from Arnoldi (`scipy.sparse.linalg.eigs`). |
| `zf_jg_doses.py` | the constraint right hand sides: `b1` from `C2`, and `b2 = b3 = 0`, no booster budget having been specified. |
| `smoke_zf_jg.py` | checks `zf_jg.py` against the notebook, per `chi`. |
| `gp_single_dose.py` | the minimization for `b2 = b3 = 0` with the weekly budget `b1`, the same problem as `min_rho_single_dose_H_ZF_JG.nb`, three ways: the interior penalty method of Theorem 3.2 on the product of simplices, the geometric program (S4.2) of the supplement, and eight random starts; then the optimality conditions (3.21) and John's allocation for comparison.  Reproduces `rho = 2.9059382878` (John's `2.9312317763`) and every number of Section S2 of the supplement.  The Arnoldi start vector is fixed, so a run reproduces to the last digit. |
| `plot_chi1_single_dose.py` | solves the single dose problem as `gp_single_dose.py` does and writes `notes/figures/dat/fig_chi1_single_dose.dat`, the minimizing `chi1`, and `notes/figures/dat/fig_chi1_john.dat`, the `chi1` of John's method for the same `b1` (units `1e-3` per day, one column per location).  Figure 3.1 of the write-up shows the two side by side. |
| `gs_three_dose.py` | three doses, the weekly test instance (`sigma2 = sigma3 = P`, `b2 = b3 = b1/4`): the joint local method (trust region interior point) and block Gauss-Seidel over the dose types, from the barycenter and the 48 starts near a vertex of one factor, with a KKT check at every endpoint.  Writes `gs_three_dose_results.txt`; runs in about 6 minutes.  Table T3.1 of the earlier version of the three-dose note, `old_files/math_bio_three_doses.tex`.  The Yogi scripts import from it the instance, the objective function, its gradient (4.4) of the main document, the 49 starts and the KKT check; `gauss_seidel` takes an optional `hist` list that records `rho` per sweep. |
| `test_barycenter_three_dose.py` | 36 three-dose instances (weekly or daily `b1`, six ratios `(b2, b3)/b1`, `sigma2`, `sigma3` = `P` or `P` times log-normal noise): the joint method and Gauss-Seidel from the barycenter against the best of 69 joint starts.  Writes `test_barycenter_three_dose_results.txt`; took 88 minutes, two instances taking 22 and 39 minutes.  Section T4 of the earlier version; in Section 4.4 of the main document the best of the 69 interior point runs is the reference `rho_best`, read by `yogi_instances_three_dose.py`. |
| `yogi_three_dose.py` | projected Yogi, Algorithm 1 of Section 4.3 of the main document: the weighted projection onto the simplices (Lemma 4.1(b)), the stationarity measure `r` (4.8), and the run on the weekly instance from the 49 starts.  All end at `rho = 1.661826042009` with `r < 1e-10`.  Writes `yogi_three_dose_results.txt`; with `--tmax 200000` it took 939 s.  Table 4.1.  `yogi` takes an optional `hist` list that records `(t, rho, r)` per iteration. |
| `yogi_instances_three_dose.py` | projected Yogi from the barycenter on the 36 instances of `test_barycenter_three_dose.py`, against the best of its 69 interior point runs: successful in 31, below that value in two of them.  Writes `yogi_instances_three_dose_results.txt`; took 71 s.  Table 4.2. |
| `plot_yogi_three_dose.py` | the data of the figures of Section 4.4, written to `notes/figures/dat/`: `fig_yogi_hist_{bary,k3,k2}.dat`, the relative excess and `r` per iteration from three starts (Figure 4.1), `fig_gs_hist_{bary,k3}.dat`, Gauss-Seidel per sweep (in no figure of the main document), and `fig_alloc_{s2,p}_{joint,yogi}.dat`, `chi_k` at the endpoints of the interior point method and of Yogi in the two instances in which Yogi gives the smaller value (Figures 4.2 and 4.3, units `1e-3` per day).  About 6 minutes. |
| `make_figures.py` | the four figures of the write-up with matplotlib, from the `.dat` files in `notes/figures/dat/`: `fig_chi1`, `fig_yogi_conv`, `fig_alloc_s2` and `fig_alloc_p`, each as png (300 dpi) in `notes/figures/`, 6.5 in wide, the text width.  Run it after either plot script. |
| `check_block2.py` | samples `g_p` of (2.4) in the second dose: `log g` is not convex and `g` not monotone in `u2`.  Section 4 of the main document. |
| `check_gs_endpoints.py` | two Gauss-Seidel endpoints of `gs_three_dose.py` with larger `rho`: local minimizers to the resolution of a random-direction test, and a violation of convexity on the segment to the joint minimizer.  Section T3 of the earlier version. |
| `reduced_test.py` | checks that the reduced problem `min rho(A1 diag(g))`, of order `N`, is the original one, of order `2N`, with `eigenvalues_H_ZF_JG.nb` as reference.  It minimizes both, for one dose with the budgets `b1/7`, `b1` (the write-up's) and `3 b1` and for three doses on test data (`sigma2 = sigma3 = P`, `b2 = b3 = b1/4`) from four common starts; then, for 23 values of `chi` including all the minimizers, it compares `diag(d) P A1 P^T` with the notebook's `NGM` entrywise, the eigenvalues of `A1 diag(g)` with those of the notebook's `NGM`, and `rho`. |
| `../../Mathematica_Wolfram/ref_h.wls` | the Wolfram script `reduced_test.py` calls: it runs `settings_zf_jg.wl` with that `chi` and then the input cells of `eigenvalues_H_ZF_JG.nb` for each `chi` of `intermediate/cases.json`, with `Export` and `SetDirectory` disabled, and stores `NGM`, `Y`, `Z`, `rho` and sixteen eigenvalues in `intermediate/ref_h.json`.  The notebook is not modified. |

`NGM` is built three ways, which is the point of the checks:

* `ngm_blocks(chi1, chi2, chi3)` assembles it block by block the way the
  notebook does, `mH[i, p] = Omega[i] . ccMat[i, p] . Psi_I[p] . Q`;
* `ngm_factored(chi1, chi2, chi3)` forms the single product `D(chi) . Ac`.
  `D(chi)` is diagonal, so this is the same matrix: the left product only
  scales row `k` of `Ac` by `d_k`;
* `ngm_reversed(chi1, chi2, chi3)` forms `Ac . D(chi)`.  That is a different
  matrix, and it must carry the same eigenvalues, since `AB` and `BA` share a
  characteristic polynomial.  This is the `F V^-1` versus `V^-1 F` argument of
  the write-up, and it says `rho` may be minimized in whichever order is
  convenient.

`build_ngm` is an alias of `ngm_factored`.

Run the model and the test with

    cd code/python
    python3 -u zf_jg.py
    python3 -u smoke_zf_jg.py [--seed S] [--random K] [--tol T]
    python3 -u reduced_test.py

`zf_jg.py` prints `rho`, `l0`, the agreement between the block and the
factored assembly, and the agreement between the spectra of `D(chi) . Ac` and
`Ac . D(chi)`.  `smoke_zf_jg.py` needs
`wolframscript` on the path; for each `chi` it re-evaluates the notebook in a
Wolfram kernel and reports six numbers: the block assembly against the
notebook's `Y`, `Z` and `NGM`; `D(chi) . Ac` against the block assembly,
entrywise; `rho` and `l0` against the notebook; `Ac . D(chi)` against
`D(chi) . Ac` through Arnoldi, the thirty eigenvalues of largest modulus
scaled by `rho`; and, for information only, how far apart those last two
matrices are entrywise, which is of order one and shows the spectral
agreement is not an identity.  Its cases are `chi = 0`, five structured
`chi`, and `K` random draws whose scale is swept over four decades.
Agreement is at the level of rounding: entrywise below `6e-16` relative,
spectra below `2.1e-15`, and `rho` and `l0` below `4.1e-14`.

`reduced_test.py` also needs `wolframscript`.  The two minimizations agree to
`6.2e-10` in `rho` or better in every case, and the entries, eigenvalues and
`rho` agree with the notebook to `4.4e-15` relative or better.  For three doses
one of the four starts ends at a different local minimum, the same one in both
formulations.

### Write-up

In `notes/`, with `bib_mb.bib` and, in `notes/figures/`, the png figures it includes; build in `notes/`.

`min_spectral_radius_vaccination.tex` — minimizing the spectral radius of a
diagonally scaled positive matrix.  Section 1 defines the next generation matrix
`D(u) Ac` and the constraints as the notebooks have them; section 2 gives the equivalent
problem `min rho(A1 diag(g))` with `A1` of order `N = mn` against the `2N` of
`Ac`, and lists three properties (i)-(iii) with short proofs: the feasible set
is a Cartesian product of simplices, the two problems are equivalent, and
`rho(A1 diag(g))` is strictly increasing in `g`; section 3
treats the single dose, `u2 = u3 = 0`: the setup, the feasible set `K` in `t`
and its image `K-hat`, the product of solid simplices in the variables
`u-hat_1` of property (i), Theorem 3.1 (the objective function is strictly convex, the minimizer
unique, the KKT conditions sufficient), Theorem 3.2 (the interior penalty method
is well defined for every weight `mu`, its value is within `mu m (n+1)` of the
minimum, and its minimizers converge to the minimizer), and the algorithm, whose
gradient needs one Perron vector of `A1 diag(g)` and one of its transpose;
section 4 treats the three doses, to which the theorems do not apply: the
problem is not convex (the sampling of `check_block2.py`, and KKT points with
different values), the objective function `Theta` is smooth with gradient
(4.4), the projection onto the simplices in a diagonal metric (Lemma 4.1), its
fixed points are exactly the KKT points (Lemma 4.2), projected Yogi
(Algorithm 1), and if its iterates converge their limit satisfies the KKT
conditions (Lemma 4.3); Section 4.4 the weekly instance from 49 starts and
the 36 instances, Tables 4.1 and 4.2 and Figures 4.1--4.3.  Values of `rho` in
figures and tables have four decimals.
The hypothesis on `A1` throughout is `A1 >= 0` irreducible; for the data of the
model `A1 > 0`.  Appendix A proves the equivalence of the two matrices,
appendix B collects the facts used.  The main document refers to nothing
outside itself.  Bibliography in `bib_mb.bib`, which also keeps the unused
`2015Kingma_Ba`.

In `notes/suplemental/`, built there, the two documents below; each takes
`../bib_mb.bib` and the main document's labels from
`../min_spectral_radius_vaccination.aux`.

`min_spectral_radius_vaccination_supplement.tex` — what is not needed for the
theorems and the algorithm, its numbers prefixed S: S1 the bounds
`(1 - eps1) rho(A1) <= rho <= rho(A1)`, S2 the numerical test for the single
dose (every number from `gp_single_dose.py`) and the stored allocation
`OptimalChi1`, S3 the three-dose test instance, S4 the geometric programming
form.  It refers to the main document's labels through `xr-hyper` with the
prefix `M-`, so build the main document first.

`math_bio_three_doses.tex` — superseded: its projected Yogi version was merged
into Section 4 of the main document on 2026-11-08, with the notation of the
main document (`Theta`, weights `theta`, first moment `kappa`).  It still
builds on its own, with `M-` cross-references to the main document whose
numbers are no longer current.  The earlier version, with block Gauss-Seidel,
the joint method and the convexity of the dose-1 and dose-3 blocks, is in
`old_files/`.


### Data

The python carries its own data:
`zf_jg_params.py` holds the notebook parameters, exported through a Wolfram
kernel, and `zf_jg_doses.py` holds `C2`.  The csv exports of the older
notebooks are in `old_files/`.  Two of them are read: `gp_single_dose.py` and
`reduced_test.py` take `S` from `old_files/solution_nb_Data/s1Matrix.csv`, and
`reduced_test.py` takes `P` from `p1Matrix.csv` in the same directory.

### `code/python/intermediate/`

Not tracked (in `.gitignore`).  Files produced by a run and not kept under any other name: `cases.json` and
`ref_h.json` of `reduced_test.py`, and `reduced_test.log`, the output of its
last run; `yogi_three_dose.log`, `yogi_instances_three_dose.log` and
`plot_yogi_three_dose.log`, the output of the last runs of the Yogi scripts.

### `old_files/`

Superseded material, kept for reference and, the two csv files above apart, not read by anything:
the eleven earlier `eigenvalues_H*` and `F_ZF*` notebooks,
`eigenvalues_ZF_JG.nb` (the same data and `NGM` as `eigenvalues_H_ZF_JG.nb`,
moved here on 2026-10-17; `zf_jg_params.py` was exported from it),
`math_bio_positive_matrices.tex` and its pdf,
`math_bio_three_doses.tex` and its pdf in the version with block Gauss-Seidel
and the joint method (copied here on 2026-11-08, before the note was shortened
to projected Yogi), `math_bio_positive_matrices_clean.tex` and its pdf as they
were before the three-dose material was merged into Section 4 (copied here on
2026-11-08), the older python drivers
(`functions_bio.py`, `x_/y_/z_math_bio.py`, `power_method.py` and friends),
the `.m` files (`aa.m`, `dfs0.m`, `ngm_eig.m`), the csv exports
(`nb_Data/`, `new_nb_Data/`, `solution_nb_Data/`, `solution_nb_Data.tar.gz`)
and the `.dat` files.
`notebooks_summary.md` there surveys the twelve original notebooks.

## Conventions

Inside a `2n` by `2n` block the rows and columns are the pairs (susceptible,
vaccinated-or-boosted) of one age group, interleaved:
`(Y_i0, Z_i0, Y_i1, Z_i1, ...)`.  The global index of `(i, j, a)`, with
`a = 0` for `Y` and `a = 1` for `Z`, is `i * 2n + 2 * j + a`.

The entries of `eta` are ordered as the infected compartments of
`v_ij = (E^s, E^p, I^a, I^ps, I^s, I^h, I^pm, I^m)`, that is
`eta = (eta_a, eta_ps, eta_s, eta_h, eta_pm, eta_m) = (0.5, 1.25, 1, 0.1, 1, 0.75)`,
preceded by two zeros for the two exposed columns.

In the notebooks, matrix names ending in `r` correspond to random values of
`chi`, and names ending in `0` to `chi = 0`.
