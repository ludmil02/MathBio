# Weekly optimal vaccination: simulation package

The model of Feng, Day, Glasser (J. Theor. Biol. 633 (2026) 112531), 51 locations (50 states
and D.C.) x 16 age groups, solved from `t = 251` (7 September 2020) without vaccination to
the start of vaccination `t_vacc` (`t = 347`, 12 December 2020, for the parameter set `2021`;
`t = 349`, 14 December 2020, for `feng2026`), then week by week: at the start of each week the doses of each
location (all ages) are allocated among its age groups so as to minimize the spectral radius of
the next generation matrix, and the system is solved over the week with these rates.  Time `t`
is in days from 31 December 2019.

The methods are in `notes/stratified_seir_ngm.tex` (the optimization; Section 9 for the
simulation and the bound on the rates) and `notes/DATA.tex` (the data with their sources, the
model and the computation).  The program is in `code/python/` (`seir7_ode.py`,
`alternating.py`, `plot_weekly_optimal.py`).

## Contents

Only this README is in the repository.  `run.sh`, `input/` and `results/` are kept on disk
(listed in `.gitignore`) and copied between machines with `rsync`.  The mock inputs are also in
the repository: `code/python/data/mock/doses_51x16x16_mock.xlsx` and
`code/python/data/cdc/state_t251_mock.csv` (the files `seir7_ode.py` uses by default), and
`code/python/data/mock/doses_51x16x16_mock_5wk.xlsx`.

| | |
| --- | --- |
| `run.sh` | runs the simulation and then the figures |
| `input/doses_51x16x16_mock.xlsx` | weekly doses (MOCK) |
| `input/doses_51x16x16_mock_5wk.xlsx` | 5 weeks of mock doses, weeks 1-2 zero and weeks 3-5 those of the full mock file (a short test run: `./run.sh input/doses_51x16x16_mock_5wk.xlsx`) |
| `input/state_t251_mock.csv` | the state at `t = 251` (MOCK); format below |
| `results/<tag>/` | made by `run.sh`: data, log and figures of one run |

## Running

From this directory:

    ./run.sh                          # the files in input/
    ./run.sh DOSES.xlsx               # other doses
    ./run.sh DOSES.xlsx INITIAL.csv   # other doses and initial state
    PARAMS=feng2026 ./run.sh          # another parameter set

The parameters are a set (a column) of `input/parameters.xlsx`, `PARAMS=NAME`
(default `2021`, the values of Unknown 2020/21; also `feng2026`,
the values of Feng, Day, Glasser 2026).  The results go to `results/<tag>/`, `tag =
<INITIAL>__<DOSES>__<PARAMS>` (the two file names without extension, and the set), e.g.
`results/state_t251_mock__doses_51x16x16_mock__2021/`, so runs with different inputs are kept
apart.  After an interruption, `RESUME=1 ./run.sh` with the same arguments continues
from the last completed week.  The run with the 126 weeks of the mock doses took about 5000 s
(about 1.4 h); memory about 150 MB.

## The dose file

An Excel file; the entries are the numbers of people given the dose in the week.  The sheets
are taken by position, not by name: the optimal simulation reads the first sheet, whatever it
is called (`by_state` in the mock file, `doses` or any other name works), and a file with this
one sheet is enough.

* Row 1: empty cell(s) over the key columns, then `Week 1`, `Week 2`, ..., each in the middle
  one of its three columns (over `Dose 2`); the other two cells are not read.
* Row 2: the key columns, then `Dose 1`, `Dose 2`, `Dose 3+` under each week.
* Sheet 1 (`by_state` in the mock file): key column `State`, one row per location (51, in FIPS order: AL, AK,
  AZ, ...), the doses of all ages.  This is the sheet the optimal simulation uses.
* Sheet 2 (`by_state_age` in the mock file; optional): key columns `State`, `Age`, one row per location and age group
  (816; ages 0-4, 5-9, ..., 75+); used only for the actual strategy (not run).  Sheet 1 is the
  sum of sheet 2 over the ages.

Week `w` starts on day `t = t_vacc + 7 (w - 1)`.  The number of weeks is that of the file, and
the run ends at `t = t_vacc + 7 W` for `W` weeks; for the 126 weeks of the mock file, at `t =
1229` (13 May 2023) with `2021` and `t = 1231` (15 May 2023) with `feng2026`.  The mock doses are made by
`code/python/doses_51x16x16.py` (rules in `notes/DATA.tex`, Section 1.9.2); a file of
reported doses has to have the same layout.

## The initial state file

The state at `t = 251`, a CSV file without a header, 663 rows and 16 columns: 13 blocks of 51 rows, the
compartments in the order `S`, `P`, `Es`, `Ep`, `V`, `Ia`, `Ips`, `Ipm`, `Is`, `Im`, `Ih`, `R`,
`M` (rows 1-51 `S`, rows 52-102 `P`, ..., rows 613-663 `M`); within a block, one row per
location in FIPS order (AL, AK, AZ, ...); the columns are the age groups 0-4, 5-9, ..., 75+.
Entries are numbers of people.  This is the only format read (`seir7_ode.read_state`);
the size is checked.  The mock state is made by `code/python/sero_initial_51x16.py`.

## The results, in `results/<tag>/`

| file | |
| --- | --- |
| `simulate_weekly_51x16_optimal__<tag>.csv` | national totals of the 13 compartments (`S`, `P`, `Es`, `Ep`, `V`, `Ia`, `Ips`, `Ipm`, `Is`, `Im`, `Ih`, `R`, `M`) at the start of each week and at the end |
| `simulate_weekly_51x16_optimal__<tag>_rho.csv` | for each week: the spectral radius at the uniform allocation (`rho_uniform`), at the optimum (`rho_opt`, `rho_capped`), the largest rate (`chi_max_before_cap`, at most 0.658 per day), the start and sweeps of the optimization, the times and iterations of the steps for the three doses (`t_chi1`, ..., `it_chi3`), the time of the ODE solution (`t_ode`) and the memory used (`max_rss_MB`) |
| `simulate_optimal__<tag>.output` | the log of the run |
| `simulate_optimal__<tag>_checkpoint.npz` | the state after the last completed week (for `RESUME=1`) |
| `figures/fig_weekly_optimal.pdf` | the four panels below together |
| `figures/fig_weekly_optimal_rho.png` | spectral radius, uniform and optimal allocation |
| `figures/fig_weekly_optimal_compartments.png` | `S`, `P`, `V`, `R` (millions) |
| `figures/fig_weekly_optimal_infectious.png` | all infectious and hospitalized (thousands) |
| `figures/fig_weekly_optimal_deaths.png` | cumulative deaths (thousands) |

The initial state and the doses in `input/` are mock data; the results computed from them are
a test of the computation, not predictions.
