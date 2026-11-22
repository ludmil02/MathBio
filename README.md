# MathBio

The age- and location-stratified SEIR model of Feng, Day, Glasser (J. Theor. Biol. 633 (2026)
112531): its next generation matrix H, the minimization of the spectral radius of H
over the vaccination rates `chi1, chi2, chi3` (first, second, third dose), and the weekly
simulation of the model with the optimal rates.  All data are published or public, except two
inputs, which are mock data.

    H = diag(d(chi)) Hc,    rho(H) = rho(H1 diag(g)),

with `Hc` and `H1 = diag(d_c) C diag(psi_I q_s)` independent of `chi`, `g = y + r z`,
`r = q_m / q_s`.

## Documents (`notes/`)

| file | |
| --- | --- |
| `stratified_seir_ngm.tex` | the main document: the next generation matrix H, the reduction to order `n = n_l n_a` (Section 2), the gradient (Section 3), the one dose problem: optimality conditions, uniqueness and the fixed point iteration (Section 4), the alternating method for three doses (Section 5), the simulation with the weekly optimization and the bound on the rates (Section 6), and in the appendices the optimality conditions for three doses (A), the products with the contact matrix (B) and the parameter values (C) |
| `suplemental/stratified_seir_ngm_supplement.tex` | the supplement (more results for the alternating method, a Newton method) |
| `DATA.tex` | the data, each with its source, and the list of files (Section 1: F1-1 to F1-26); the model and the computation (Section 2: the program and its files F2-1 to F2-5; Section 2.3: the eigenvector strategy) |
| `simulation_weekly_optimal.tex` | the weekly optimal simulation (figures: rho(H) of both strategies, then the compartments, the infectious and the deaths of the optimal run), the public and the mock data, and its comparison with the eigenvector strategy (a table; the infectious, hospitalized and deaths of both strategies; the doses by age) |
| `bib_mb.bib` | the bibliography |

Build each with `pdflatex`, `bibtex`, `pdflatex` twice.

## Data

* `data/Downloaded_from_source/`: the downloads, as downloaded (DATA.tex, Section 1.1.1):
  `papers/` (named `YYYYAuthor1...AuthorN.pdf`), `census/`, `prem2017/`, `oxcgrt/`, `cdc/`.
  The data files (kept gzipped, `.csv.gz`, read by pandas without unpacking), the two zip
  archives and the papers are tracked.
* `code/python/data/`: the inputs of the model, built from the downloads by the scripts below
  (DATA.tex, Section 1.1.2, F1-16 to F1-26).

## Python (`code/python/`)

Run the scripts from `code/python/`.  Eigenvalues by Arnoldi (`scipy.sparse.linalg.eigs`).

### The 51 x 16 model: data and simulation

| file | |
| --- | --- |
| `census_N_51x16.py` | populations `N` (Census Bureau, Vintage 2024, July 1, 2020) |
| `prem_U_16x16.py` | contact matrix `U` (Prem et al. 2017, US) |
| `geo_51.py` | distances `d` (2010 centers of population) and activity factors `L` (county data) |
| `build_C_51x16.py` | the 816 x 816 contact matrix `C` from `U`, `L`, `d`, saved to `data/model/C_816x816.csv.gz` |
| `glasser2023_inputs.py` | Table 2 of Glasser et al. 2023 and `beta0` digitized from its Fig. 4 |
| `oxcgrt_s_51.py` | stringency factors `s_i(t) = 1 - SI_i(t)/100` (OxCGRT) |
| `cdc_variants.py` | the days of the variants (CDC variant proportions) |
| `sero_initial_51x16.py` | the initial state at `t = 251` (MOCK, from the CDC serosurvey) |
| `doses_51x16x16.py` | reads a dose file, real or mock (`read_sheet`, used by every run of `seir7_ode.py`); run as a script, it writes the weekly doses (MOCK), `data/mock/doses_51x16x16_mock_<W>wk.xlsx` for `W = 126`, `55` (the default) and `15` (a short test run), the first `W` weeks of the same values: sheet 1 `by_state` (51 rows, all ages; used by `simulate optimal`), sheet 2 `by_state_age` (816 rows; used only by `simulate actual`); header row 1 `Week 1` ... `Week 55`, each in the middle one of its three columns, over `Dose 2` (week `w` starts at `t = t_vacc + 7 (w - 1)`, `t_vacc` of the parameter set), row 2 `State` (and `Age`), then `Dose 1`, `Dose 2`, `Dose 3+` under each week; the doses by age are rounded to integers (stored as floats), and sheet 1 is their sum over the ages, exactly |
| `seir7_ode.py` | the model and its solution: `simulate optimal` (the weekly optimization of Section 6 of the main document, started each week from the uniform allocation, from the rates of the eigenvector strategy, and from the rates of the eigenvector run in the same week (`..._eigenvector__<tag>_chi.npz`, so `simulate eigenvector` runs first), so that its rho is at most that of the eigenvector strategy at the same state), `simulate eigenvector` (the same doses by the eigenvector strategy of Feng, Day, Glasser 2026, `DATA.tex`, Section 2.3), `solve`, `verify` |
| `sdirk.py` | an implicit Runge-Kutta method with step control (SDIRK of order 4, embedded order 3, Hairer-Wanner), matrix-free: Newton-GMRES with the Jacobian-vector product of `seir7_ode.py`; used by `seir7_ode.py ... method=SDIRK4` |
| `alternating.py` | the alternating method of Section 5, with the bound on the rates of Section 6.1 |
| `ngm_seir7.py` | the 51 x 16 next generation matrix H at `t = t_vacc` (start of vaccination) from the same data (option `params=`): `rho(H)`, `rho(F V^-1)`, the left eigenvector, the gradient |
| `plot_weekly_optimal.py` | the figures of the weekly optimal simulation, from `data/simulate_weekly_51x16_optimal__<tag>*.csv` (options `initial=`, `doses=`, `params=` as for `seir7_ode.py`), saved as `fig_weekly_optimal*.png` to the folder given as argument (or `out=`, required) |
| `plot_compare.py` | the comparison figures of the strategies run for one set of inputs (`python3 plot_compare.py RESULTS_DIR`), saved to `RESULTS_DIR/figures/fig_compare_*.png` |
| `weekly_report.py` | `python3 weekly_report.py RESULTS_DIR`: all figures of one results directory (`plot_weekly_optimal.py`, `plot_compare.py`), and the run values (input files, parameter set, days, weeks, results) written to `notes/run_values.tex`, included by `notes/DATA.tex` and `notes/simulation_weekly_optimal.tex`, and the values used by `notes/stratified_seir_ngm.tex` (r_p, omega, epsilon_1, the Delta_p bounds) written to its last appendix, "Parameter values", which is created if missing |

Order: `census_N_51x16.py`, `prem_U_16x16.py`, `geo_51.py`, `build_C_51x16.py`,
`glasser2023_inputs.py`, `oxcgrt_s_51.py`, `cdc_variants.py`, `sero_initial_51x16.py`,
`doses_51x16x16.py`, then

    python3 -u seir7_ode.py simulate optimal 2>&1 | tee intermediate/seir7_ode_simulate_optimal__state_t251_mock__doses_51x16x16_mock_55wk__2021.output

The inputs are `initial=FILE` (the state at `t0` of the parameter set, default `data/cdc/state_t251_mock.csv`;
a CSV file, 664 rows and 18 columns: row 1 `variable`, `location`, then the age groups 0-4,
5-9, ..., 75+; then 13 blocks of 51 rows, the compartments in the order `S`, `P`, `Es`, `Ep`,
`V`, `Ia`, `Ips`, `Ipm`, `Is`, `Im`, `Ih`, `R`, `M` (rows 2-52 `S`, rows 53-103 `P`, ..., rows
614-664 `M`); within a block, one row per location in FIPS order (AL, AK, AZ, ...); column A
the compartment, column B the location, then the 16 ages.  Entries are numbers of people.)
and `doses=FILE` (default `data/mock/doses_51x16x16_mock_55wk.xlsx`), e.g.
`simulate optimal initial=data/cdc/my_state.csv doses=data/mock/my_doses.xlsx`, and the
parameter set `params=NAME`, a column of `weekly_simulation/input/parameters.xlsx` (kept on
disk, not in the repository; default `2021`, the values of Unknown 2020/21; also `feng2026`,
the values of `table2.csv`, `beta0.csv`, `variant_days.csv` and Tables S1, S4 of the supplement
of Feng, Day, Glasser 2026).  The file has the sheets `scalars`, `by_age` and `over_time` (each
parameter that changes over time with its own switch days); a new set is a new column.  The day of the initial state `t0` and the start of vaccination `t_vacc` belong to the set
(`2021`: `t0` = 347, `t_vacc` = 354; `feng2026`: `t0` = 251, `t_vacc` = 349); the state is
integrated without vaccination from `t0` to `t_vacc`.  The outputs are named by the inputs, `tag = <initial>__<doses>__<params>` (file names
without directory and extension, and the set): `data/simulate_weekly_51x16_optimal__<tag>.csv`,
`data/simulate_weekly_51x16_optimal__<tag>_rho.csv`, and the checkpoint
`intermediate/simulate_optimal__<tag>_checkpoint.npz`, saved after each week.  After an
interruption (Ctrl-C), `resume` with the same inputs continues from the last completed week
(and stops if there is no checkpoint for these inputs):

    python3 -u seir7_ode.py simulate optimal resume 2>&1 | tee -a intermediate/seir7_ode_simulate_optimal__state_t251_mock__doses_51x16x16_mock_55wk__2021.output

and then `python3 plot_weekly_optimal.py` (with the same `initial=`, `doses=`, `params=`, if
not the defaults) for the figures.

The ODEs are solved by DOP853 (explicit Runge-Kutta of order 8, scipy), the default.  Adding
`method=SDIRK4` to any mode of `seir7_ode.py` uses `sdirk.py` instead.  On `[251, 600]` it
agrees with DOP853 (rtol `1e-12`) to `1.2e-4`, `1.3e-6`, `3.7e-7` (relative, max norm) at
rtol `1e-4`, `1e-6`, `1e-8`, but is slower: the system is not stiff there.

Logs are written as `*.output`, not `*.log`, so that they are not removed with the TeX logs.

## Conventions

Groups are ordered `p = i n_a + j` (location `i` in the order of the FIPS codes, age group `j`,
from 0; `n_a = 16` age groups, `n` in the code).  In H the rows and columns of group `p` are `2p` (susceptible rows `Y`, columns
`q_s`) and `2p + 1` (vaccinated or partially immune rows `Z`, columns `q_m`).  Time `t` is in
days from 31 December 2019.
