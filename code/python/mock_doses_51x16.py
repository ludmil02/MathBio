"""Mock weekly doses of the 51 x 16 model, written to the Excel file data/doses_51x16_mock.xlsx,
and the mock group sizes (mock_sigma).

The Excel file has one sheet, "doses":
  row 1        column A empty; "Week1" over columns B-D, "Week2" over E-G, ...,
               "Week55" over FH-FJ (merged cells);
  row 2        "State", then "Dose1", "Dose2", "Dose3" for each week;
  rows 3-53    the 51 locations (two-letter codes in column A), and for each week the
               number of people vaccinated with dose 1, 2, 3 at the location in that week.

Mock values: reference rates chi_ref = (3e-3, 1/28, 1e-3) per day, the same for all
groups.  The group sizes sigma_{1,p}, sigma_{2,p}, sigma_{3,p} (mock_sigma) are the
susceptible, vaccinated and partially immune people of group p at the disease free
equilibrium with chi = chi_ref, the same in every week.  The doses of week 1 are
b_{k,i} = 7 sum_{p in I_i} sigma_{k,p} chi_ref_k per week; weeks 2 and 3 have 1.25 and 1.5
times these, and week w >= 4 has 1.5 exp(0.2 z) times these, z standard normal,
independent for each location, dose and week (seed 0).  Replace them with the real doses.

Usage:  python3 -u mock_doses_51x16.py
"""

import os

import numpy as np
from openpyxl import Workbook

import ngm_seir7 as S7

CHI_REF = (3.0e-3, 1.0 / 28.0, 1.0e-3)
WEEKS = 55
XLSX = os.path.join(S7.HERE, "data", "doses_51x16_mock.xlsx")


def mock_sigma():
    """[S, V, P] per group (length 816) at the disease free equilibrium with chi = chi_ref."""
    N = S7.m * S7.n
    c1, c2, c3 = (np.full(N, c) for c in CHI_REF)
    p1, p2, p3 = c1 / (S7.omv + S7.e2 * c2), S7.e2 * c2 / S7.om, 1.0 + S7.e3 * c3 / S7.om
    S = S7.mN.ravel() / (1.0 + p1 * (1.0 + p2 + p2 * p3))
    return [S, p1 * S, p2 * p1 * S]


def week_factors(rng):
    """Factors of the doses of week 1, shape (WEEKS, 51, 3)."""
    f = np.empty((WEEKS, S7.m, 3))
    f[0], f[1], f[2] = 1.0, 1.25, 1.5
    f[3:] = 1.5 * np.exp(0.2 * rng.standard_normal((WEEKS - 3, S7.m, 3)))
    return f


def main():
    m, n = S7.m, S7.n
    sig = mock_sigma()
    week1 = np.array([[7.0 * sig[k][i * n:(i + 1) * n].sum() * CHI_REF[k] for k in range(3)]
                      for i in range(m)])                              # (51, 3)
    doses = week_factors(np.random.default_rng(0)) * week1[None, :, :]  # (weeks, 51, 3)
    wb = Workbook()
    ws = wb.active
    ws.title = "doses"
    ws.cell(row=2, column=1, value="State")
    for w in range(WEEKS):
        c0 = 2 + 3 * w
        ws.cell(row=1, column=c0, value="Week%d" % (w + 1))
        ws.merge_cells(start_row=1, start_column=c0, end_row=1, end_column=c0 + 2)
        for k in range(3):
            ws.cell(row=2, column=c0 + k, value="Dose%d" % (k + 1))
    for i, s in enumerate(S7.STATES):
        ws.cell(row=3 + i, column=1, value=s)
        for w in range(WEEKS):
            for k in range(3):
                ws.cell(row=3 + i, column=2 + 3 * w + k, value=float(doses[w, i, k]))
    os.makedirs(os.path.dirname(XLSX), exist_ok=True)
    wb.save(XLSX)
    tot = doses.sum(axis=1)
    print("doses per week, US total: week 1: %.4e %.4e %.4e; week %d: %.4e %.4e %.4e"
          % (*tot[0], WEEKS, *tot[-1]))
    print("written", XLSX)


if __name__ == "__main__":
    main()
