# MathBio

The age- and location-stratified SEIR model of Feng, Day, Glasser (J. Theor. Biol. 633 (2026)
112531): its next generation matrix (NGM), the minimization of the spectral radius of the NGM
over the vaccination rates `chi1, chi2, chi3` (first, second, third dose), and the weekly
simulation of the model with the optimal rates.  All data are published or public, except two
inputs, which are mock data.

    NGM = diag(d(chi)) Ac,    rho(NGM) = rho(A1 diag(g)),

with `Ac` and `A1 = diag(d_c) C diag(psi_I q_s)` independent of `chi`, `g = y + r z`,
`r = q_m / q_s`.

## Documents (`notes/`)

| file | |
| --- | --- |
| `stratified_seir_ngm.tex` | the main document: the NGM, the reduction to order `N = m n`, the optimality conditions, the fixed point iteration, the alternating method for three doses, the numerical results (Section 6.5), the simulation with the weekly optimization and the bound on the rates (Section 9), and the products with the contact matrix (appendix) |
| `suplemental/stratified_seir_ngm_supplement.tex` | the supplement (more results for the alternating method, a Newton method, the 51 x 16 data before vaccination) |
| `DATA.tex` | the data, each with its source, and the list of files (Section 1: F1-1 to F1-25); the model and the computation (Section 2: the program and its files F2-1 to F2-4) |
| `bib_mb.bib` | the bibliography |

Build each with `pdflatex`, `bibtex`, `pdflatex` twice.

## Data

* `data/Downloaded_from_source/`: the downloads, as downloaded (DATA.tex, Section 1.1.1):
  `papers/` (named `YYYYAuthor1...AuthorN.pdf`), `census/`, `prem2017/`, `oxcgrt/`, `cdc/`.
  The data files (kept gzipped, `.csv.gz`, read by pandas without unpacking), the two zip
  archives and the papers are tracked.
* `code/python/data/`: the inputs of the model, built from the downloads by the scripts below
  (DATA.tex, Section 1.1.2, F1-15 to F1-25).

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
| `mock_doses_51x16x16.py` | the weekly doses by location, age and dose (MOCK) |
| `seir7_ode.py` | the model and its solution: `simulate optimal` (the weekly optimization of Section 9 of the main document), `solve`, `verify` |
| `alternating.py` | the alternating method of Section 6, with the bound on the rates of Section 9.1 |
| `ngm_seir7.py` | the 51 x 16 NGM at `t = 349` (start of vaccination) from the same data: `rho(NGM)`, `rho(F V^-1)`, the left eigenvector, the gradient |

Order: `census_N_51x16.py`, `prem_U_16x16.py`, `geo_51.py`, `build_C_51x16.py`,
`glasser2023_inputs.py`, `oxcgrt_s_51.py`, `cdc_variants.py`, `sero_initial_51x16.py`,
`mock_doses_51x16x16.py`, then

    python3 -u seir7_ode.py simulate optimal | tee intermediate/seir7_ode_simulate_optimal.log

### The results of the main document and the supplement

| file | |
| --- | --- |
| `verify_stratified_seir_ngm.py` | checks the formulas and the reduction on the 4 x 4 and 51 x 16 data; ends with PASS or FAIL |
| `optimize_doses_51x16.py`, `three_doses_51x16_mock.py`, `mock_doses_51x16.py` | the three doses, week by week, with mock doses and mock group sizes: Section 6.5 and Table 2 of the main document; see `code/python/README_three_doses.md` |
| `figures_stratified_seir_ngm.py` | the figures, written to `notes/figures/` |
| `zf_jg.py`, `zf_jg_params.py`, `zf_jg_doses.py`, `compare_allocations_4x4.py` | the 4 x 4 model (four census regions, four age groups) |

## Other directories

* `archive/`: earlier versions, left as they are.

## Conventions

Groups are ordered `p = i n + j` (location `i` in the order of the FIPS codes, age group `j`,
from 0).  In the NGM the rows and columns of group `p` are `2p` (susceptible rows `Y`, columns
`q_s`) and `2p + 1` (vaccinated or partially immune rows `Z`, columns `q_m`).  Time `t` is in
days from 31 December 2019.
