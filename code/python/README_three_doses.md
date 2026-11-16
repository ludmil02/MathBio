# Optimal allocation of three doses, week by week

`optimize_doses_51x16.py` minimizes the spectral radius `rho(NGM)` of the
51 x 16 model (50 states and D.C., 16 age groups, with the parameters at `t = 349`, the start
of vaccination, from the public data of `notes/DATA.tex`) over the
vaccination rates `chi1, chi2, chi3` (first, second, third dose, per day),
subject to the doses available at each location:

    sum over the age groups p of location i of  sigma_{k,p} chi_{k,p} = b_{k,i},   chi_{k,p} >= 0.

The method is that of Section 6 of `notes/stratified_seir_ngm.tex`
(`alternating.py`): the alternating method, minimizing over `chi1`, then
`chi3`, then `chi2`, each step by the damped fixed point iteration (damping
1/2), from five starts (uniform, left eigenvector, three random); the end point
with the smallest `rho` is the solution.  A step is done by spectral projected
gradient instead if C is not irreducible, if a location has no group with
`Delta_p > 0` (second dose), or if the fixed point iteration does not converge;
the log says so.

## What you need

* Python 3 with `numpy`, `scipy`, `pandas` and `openpyxl`.  No pip is needed:
  on Debian/Ubuntu,
  `sudo apt install python3-numpy python3-scipy python3-pandas python3-openpyxl`.
* The model data in `code/python/data/` (built from public data, `notes/DATA.tex`,
  Section 1; tracked in the repository).
* The Excel file of the weekly doses, `data/doses_51x16_mock.xlsx` (or another
  one, `--file`); its first sheet:
  * row 1: column A empty, `Week1` over columns B-D, `Week2` over E-G, ...,
    `Week55` over FH-FJ;
  * row 2: `State`, then `Dose1`, `Dose2`, `Dose3` for each week;
  * rows 3-53: the 51 locations, two-letter codes in column A
    (`AL AK AZ AR CA CO CT DE DC FL ...`), and for each week the number of people
    vaccinated with dose 1, 2, 3 at the location in that week.  The rows are
    matched to the locations by the code, so their order does not matter; a
    missing, repeated or unknown code stops the script.  The doses are divided
    by 7, since `chi` is a rate per day, and must be positive.

`python3 -u mock_doses_51x16.py` writes the file with **mock** doses for 55
weeks (from reference rates `chi_ref = (3e-3, 1/28, 1e-3)` per day; weeks 2 and
3 have 1.25 and 1.5 times the doses of week 1, and the later weeks 1.5 times
them with a random factor `exp(0.2 z)`).  Replace them with the real doses.

The group sizes `sigma_{k,p}` (the people of each group who can receive dose
`k`) are **mock** as well: `S, V, P` at the disease free equilibrium of
`chi_ref`, computed by `mock_doses_51x16.mock_sigma`, the same in every week.

The weeks are independent: the model is not advanced from one week to the next.

## Running

From `code/python/`:

    python3 -u optimize_doses_51x16.py --weeks all | tee intermediate/optimize_doses.output

| option | |
| --- | --- |
| `--weeks` | `1`, `2-5`, `1,3,4` or `all` (all weeks of the file); default `all` |
| `--file` | the Excel file; default `data/doses_51x16_mock.xlsx` |
| `--random-starts` | number of random starts besides the uniform and left eigenvector ones; default 3 |
| `--theta` | damping of the fixed point iteration, `0 < theta <= 1`; default `0.5` |
| `--out` | output prefix; default `intermediate/optimize_doses` |

Each week takes about 30 seconds (five starts), so all 55 weeks about half an
hour.

## Output

* `intermediate/optimize_doses_week<w>.csv`: the solution by state and age,
  the rates `chi1_per_day, chi2_per_day, chi3_per_day` and the doses per week
  `doses1_per_week, ...` (`7 sigma chi`).
* `intermediate/optimize_doses_summary.csv`: one row per week and start, `rho`
  at the end point, the number of sweeps, whether it is the solution, and the
  KKT spreads at the solution.

## Reading the log

* `[start] sweep s: after chi1 ..., chi3 ..., chi2 ...`: `rho` after each step,
  with the iterations of each fixed point iteration.
* `solution: rho = ... from the start ...; end points from the 5 starts: rho in [...]`.
* `KKT dose k: spread ..., ratio where no doses ..., dose residual ...`: the
  KKT conditions (4.1) at the solution; spread 0 and ratio at most 1 at a KKT
  point.
* `... dose k step: <reason>; spg instead`: a step fell back to spectral
  projected gradient.

The earlier code (one dose, comparisons, Newton, other methods for the steps)
is in `archive/code_python_2026-11-15/`.
