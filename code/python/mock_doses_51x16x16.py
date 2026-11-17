"""MOCK weekly doses by location, age group and dose, from the start of vaccination
(t = 349, 14 December 2020) through the end of the Public Health Emergency (11 May 2023),
126 weeks. To be replaced by the doses reported to the CDC (notes/DATA.tex, Section 1.9.2).

Excel file data/mock/doses_51x16x16_mock.xlsx, sheets "Dose1", "Dose2", "Dose3"; in each,
columns state, age, then one column per week headed by its first day t = 349, 356, ...;
816 rows (locations in FIPS order, ages 0-4, ..., 75+); entries: people given that dose in
that week.

Mock rules (all made up, chosen only to look plausible):
  dose 1: age group j receives a total of cov_j N_ij first doses, spread evenly over 12 weeks
          from its start week w_j:
            ages  75+, 70-74, 65-69: cov 0.90, w = 1 (14 Dec 2020)
                  50-64:             cov 0.80, w = 9
                  20-49:             cov 0.70, w = 15
                  15-19:             cov 0.55, w = 17
                  10-14:             cov 0.45, w = 22
                  5-9:               cov 0.30, w = 47
                  0-4:               cov 0.10, w = 79;
          after these 12 weeks, 0.002 N_ij first doses per week until the end, so that every
          location has first doses in every week (hence a group with Delta_p > 0, Section 6.3
          of notes/stratified_seir_ngm.tex);
  dose 2: 0.9 times dose 1 of four weeks earlier;
  dose 3: 0.5 times the total of dose 2 of age group j, spread evenly over 20 weeks from
          week 40 for ages 65+ and week 48 for ages 20-64; none below age 20;
  each location, dose and week is multiplied by exp(0.1 z), z standard normal (seed 0).

Usage:  python3 -u mock_doses_51x16x16.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "mock", "doses_51x16x16_mock.xlsx")
T_VACC, WEEKS = 349, 126
# coverage and start week of dose 1, by age group 0-4, ..., 75+
COV = np.array([0.10, 0.30, 0.45, 0.55] + [0.70] * 6 + [0.80] * 3 + [0.90] * 3)
START = np.array([79, 47, 22, 17] + [15] * 6 + [9] * 3 + [1] * 3)
SPREAD1, SPREAD3, LAG2, F2, F3 = 12, 20, 4, 0.9, 0.5
TAIL1 = 0.002          # first doses per week after the campaign, fraction of the group
START3 = np.array([0] * 4 + [48] * 9 + [40] * 3)        # 0 = none


def build():
    pop = pd.read_csv(os.path.join(HERE, "data", "census", "N_51x16.csv"))
    m, n = 51, 16
    N = pop.N.to_numpy(float).reshape(m, n)
    D = np.zeros((3, m, n, WEEKS))
    for j in range(n):
        w = np.arange(START[j] - 1, min(START[j] - 1 + SPREAD1, WEEKS))
        D[0, :, j, w] = (COV[j] * N[:, j] / SPREAD1)[None, :]
        later = np.arange(START[j] - 1 + SPREAD1, WEEKS)          # continued uptake after the campaign
        D[0, :, j, later] = (TAIL1 * N[:, j])[None, :]
    if not (D[0].sum(axis=1) > 0).all():                         # every location, every week
        raise ValueError("a location has no first doses in some week")
    D[1, :, :, LAG2:] = F2 * D[0, :, :, :-LAG2]
    for j in range(n):
        if START3[j]:
            w = np.arange(START3[j] - 1, min(START3[j] - 1 + SPREAD3, WEEKS))
            D[2, :, j, w] = (F3 * D[1, :, j, :].sum(axis=1) / SPREAD3)[None, :]
    rng = np.random.default_rng(0)
    D *= np.exp(0.1 * rng.standard_normal((3, m, 1, WEEKS)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    weeks = [str(T_VACC + 7 * w) for w in range(WEEKS)]
    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        for k in range(3):
            df = pd.DataFrame(D[k].reshape(m * n, WEEKS), columns=weeks)
            df.insert(0, "age", pop.age)
            df.insert(0, "state", pop.state)
            df.to_excel(xw, sheet_name=f"Dose{k + 1}", index=False)
    print(f"doses 1, 2, 3 in total: " + ", ".join(f"{D[k].sum() / 1e6:.0f} M" for k in range(3))
          + f" (population {N.sum() / 1e6:.0f} M); wrote {OUT}")


if __name__ == "__main__":
    build()
