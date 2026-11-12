# MathBio

Next generation matrix (NGM) of the age- and location-stratified SEIR model of
Feng, Day, Glasser (J. Theor. Biol. 633 (2026) 112531), and the minimization of
its spectral radius over the vaccination rates `chi1, chi2, chi3`.

The NGM has order `2N`, `N = mn` groups (`m` locations, `n` age groups):

    NGM = diag(d(chi)) Ac,    rho(NGM) = rho(A1 diag(g)),

with `Ac` and `A1 = diag(d_c) C diag(psi_I q_s)` independent of `chi`, and
`g = y + r z`, `r = q_m / q_s`.  The two column weights `q_s` (naive infected)
and `q_m` (partially immune infected) are those of the paper.  The derivation,
the reduction to order `N` and the formulas for the minimization are in
`notes/stratified_seir_ngm.tex`.

Two data sets are used:

* `4 x 4`: four census regions, ages 0-19, 20-39, 40-59, 60+ (`zf_jg_params.py`);
* `51 x 16`: the 50 states and D.C., ages 0-4, ..., 70-74, 75+, at `t = 347`,
  the day before vaccination began (`code/python/intermediate/seir7_t347/`).

## Contents

| directory | |
| --- | --- |
| `notes/` | `stratified_seir_ngm.tex`, `bib_mb.bib`, and the figures in `notes/figures/` |
| `code/python/` | the python; `intermediate/seir7_t347/` the 51 x 16 data; other files in `intermediate/` are run output, not tracked |
| `Mathematica_Wolfram/` | the notebooks and Wolfram scripts |
| `archive/` | `single_weight_model.zip` and `old_files.zip`, see below |

### Write-up

`notes/stratified_seir_ngm.tex`, built in `notes/` with `pdflatex`, `bibtex`,
`pdflatex` twice.  Section 1 the setting and the problem; Section 2 the NGM,
the weights `q_s`, `q_m` (Lemma 2.1) and `NGM = diag(d) Ac` (Proposition 2.2);
Section 3 the reduction `rho(NGM) = rho(A1 diag(g))` (Theorem 3.1); Section 4
the formulas for the minimization (one dose, the gradient, alternating over the
doses); Section 5 the numerical verification, Table 1 and Figures 1-3.

### Python

In `code/python/`; run the scripts from there.  Eigenvalues by Arnoldi
(`scipy.sparse.linalg.eigs`); `F V^-1` is applied matrix free, with `V`
factored by sparse LU.

| file | |
| --- | --- |
| `zf_jg_params.py` | the 4 x 4 parameters, exported through a Wolfram kernel.  `default_params()`. |
| `zf_jg_doses.py` | the doses `b1` from `C2`, and `b2 = b3 = 0`. |
| `zf_jg.py` | the 4 x 4 model: `q_vectors` (`q_s`, `q_m`), `ac_matrix`, `ngm_factored` (`D(chi) Ac`), `ngm_blocks`, `ngm_reversed`, and the reduced form `a1_matrix`, `g_vector`, `ngm_reduced` (`A1 diag(g)`).  `weights="notebook"` gives the single weight of `eigenvalues_H_ZF_JG.nb` for both columns.  `python3 -u zf_jg.py` prints `rho = 4.440677536267` at `chi = 0` and checks the reduced form. |
| `ngm_seir7.py` | the 51 x 16 model: the data of `intermediate/seir7_t347/`, the contact matrix `mc`, `Qs`, `Qm`, the NGM as an operator, `full_fv` (`rho(F V^-1)` of the 8-state system, for either data set), and `left_and_gradient` (the left eigenvector and `d rho / d chi1` at `chi = 0`).  Run, it prints `rho(NGM) = 1.307390501274`, `rho(F V^-1)`, the left eigenvector by age and state, and the gradient for the HHS region headquarter states. |
| `verify_stratified_seir_ngm.py` | reproduces every number of Section 5 and Table 1 of the write-up and ends with PASS or FAIL. |
| `figures_stratified_seir_ngm.py` | Figures 1-3 of the write-up, written to `notes/figures/`. |

### Notebooks and Wolfram scripts

In `Mathematica_Wolfram/`.

| file | |
| --- | --- |
| `export_seir7.wls` | runs the setup cells of John's notebook `Age & Location Stratified SEIR 7 Actual Vax 3 with RE_orig.nb` (not tracked, 10 MB) and exports the 51 x 16 data at a given `t` to CSV: `wolframscript -file export_seir7.wls NOTEBOOK OUTDIR 347`.  Its log is `intermediate/seir7_t347/export_seir7.log`. |
| `eigenvalues_H_ZF_JG.nb`, `settings_zf_jg.wl` | the 4 x 4 NGM as first built, with the single weight; `zf_jg_params.py` holds its data. |
| `min_rho_single_dose_H_ZF_JG.nb`, `ref_h.wls` | the single dose minimization and the reference script of the single weight model. |
| `16_Group_Model_Results_t=347_ZF.nb` | the 4 x 4 model solution at `t = 347`, with `C2`. |
| `*_orig.nb` | the notebooks as received. |
| `run_notebook.wls` | evaluates the input cells of a notebook without a front end: `wolframscript -file run_notebook.wls NOTEBOOK.nb > LOG`. |

Not tracked: John's SEIR 7 notebook and `F_and_V_ZF_18Jan_orig.nb` (the
4 x 4 `F` and `V`), and `JTB26.pdf`, the paper.

### `archive/`

* `single_weight_model.zip`, unzips to `single_weight_model/`: the earlier
  notes `min_spectral_radius_vaccination.tex`, their figures and supplemental
  documents, and the scripts and results of the single dose, joint (trust
  region), Gauss-Seidel, alternating and projected Yogi methods.  That model
  uses one weight for both column types of the NGM; its `README.md` explains
  the difference.
* `old_files.zip`, unzips to `old_files/`: the earlier notebooks, notes,
  python drivers and csv exports.

## Reproducing the write-up

From `code/python/`:

    python3 -u verify_stratified_seir_ngm.py | tee intermediate/verify_stratified_seir_ngm.log
    python3 -u figures_stratified_seir_ngm.py | tee intermediate/figures_stratified_seir_ngm.log

Both take under a second.  Then build `notes/stratified_seir_ngm.tex`.  The
51 x 16 data are tracked; regenerating them needs `wolframscript` and John's
notebook (`export_seir7.wls` above).

## Conventions

Groups are ordered `p = i n + j` (location `i`, age group `j`, from 0).  In the
NGM the rows and columns of group `p` are `2p` (susceptible rows `Y`, columns
`q_s`) and `2p + 1` (vaccinated or partially immune rows `Z`, columns `q_m`).

The infected states of a group are `(E_s, E_p, I_a, I_ps, I_s, I_h, I_pm, I_m)`,
and `eta = (eta_a, eta_ps, eta_s, eta_h, eta_pm, eta_m) = (0.5, 1.25, 1, 0.1, 1, 0.75)`
follows the infectious ones.
