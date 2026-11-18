"""The dose files: read_sheet reads a dose file, real or mock (seir7_ode.py uses it for every
run); write_sheet writes one; run as a script, it generates the MOCK doses below.

MOCK weekly doses by location, age group and dose, from the start of vaccination
(t_vacc of the parameter set in weekly_simulation/input/parameters.xlsx) for 126 weeks,
through about the end of the Public Health Emergency (11 May 2023).
To be replaced by the doses reported to the CDC (notes/DATA.tex, Section 1.9.2).

Excel file data/mock/doses_51x16x16_mock.xlsx, two sheets; entries: people given that dose in
that week. Row 1: empty cell(s) over the key columns, then "Week 1", ..., "Week 126", each
in the middle one of its three columns (over Dose 2), centered; week w starts on day t = t_vacc + 7 (w - 1), t_vacc the start of
vaccination of the parameter set of seir7_ode.py (2021: 347, feng2026: 349). Row 2: the key column
names, then "Dose 1", "Dose 2", "Dose 3+" under each week. Then:
  sheet 1 "by_state":     key column State; 51 rows (FIPS order), all ages together;
  sheet 2 "by_state_age": key columns State, Age; 816 rows (location, age 0-4, ..., 75+).
Sheet 1 is the sum of sheet 2 over the ages (checked after writing). seir7_ode.py reads
sheet 1 (simulate optimal needs only the doses of each location); sheet 2 only for
simulate actual.

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

Usage:  python3 -u doses_51x16x16.py      (writes the mock doses)
"""
import os

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "mock", "doses_51x16x16_mock.xlsx")
WEEKS = 126
# coverage and start week of dose 1, by age group 0-4, ..., 75+
COV = np.array([0.10, 0.30, 0.45, 0.55] + [0.70] * 6 + [0.80] * 3 + [0.90] * 3)
START = np.array([79, 47, 22, 17] + [15] * 6 + [9] * 3 + [1] * 3)
SPREAD1, SPREAD3, LAG2, F2, F3 = 12, 20, 4, 0.9, 0.5
TAIL1 = 0.002          # first doses per week after the campaign, fraction of the group
START3 = np.array([0] * 4 + [48] * 9 + [40] * 3)        # 0 = none


DOSE_LABELS = ["Dose 1", "Dose 2", "Dose 3+"]


def write_sheet(ws, keys, key_rows, values):
    """Row 1: len(keys) empty cells, then for each week "Week w" in the middle one of its three
    columns (the column of Dose 2), centered, the other two empty; row 2: the key names, then
    Dose 1, Dose 2, Dose 3+ for each week; then one row per key_rows entry with
    values[r] = (week 1: doses 1, 2, 3+, week 2: ..., ...)."""
    nk, nw = len(keys), values.shape[1] // 3
    ws.append([None] * nk + [x for w in range(nw) for x in (None, f"Week {w + 1}", None)])
    ws.append(keys + DOSE_LABELS * nw)
    for w in range(nw):
        ws.cell(row=1, column=nk + 3 * w + 2).alignment = Alignment(horizontal="center")
    for kr, v in zip(key_rows, values):
        ws.append(list(kr) + v.tolist())


def excel_col(j):
    """Excel column name of the 0-based column index j: 0 -> A, 25 -> Z, 26 -> AA."""
    s = ""
    j += 1
    while j:
        j, r = divmod(j - 1, 26)
        s = chr(65 + r) + s
    return s


def header_mismatch(df, nk):
    """The first difference of the two header rows of df (after the nk key columns) from the
    layout of write_sheet, as text with the Excel cell, or "" if there is none."""
    rows, cols = df.index, df.columns
    h1, h2 = df.iloc[0, nk:], df.iloc[1, nk:]
    if len(h1) % 3:
        return (f"{len(h1)} columns after the key column(s), not a multiple of 3; the last"
                f" non-empty column is {excel_col(cols[-1])} (row {rows[0] + 1}: {h1.iloc[-1]!r},"
                f" row {rows[1] + 1}: {h2.iloc[-1]!r})")
    for c in range(len(h1)):
        cell1, want1 = h1.iloc[c], (f"Week {c // 3 + 1}" if c % 3 == 1 else None)
        if want1 is not None and cell1 != want1:
            return (f"cell {excel_col(cols[nk + c])}{rows[0] + 1} is {cell1!r}, expected"
                    f" {want1!r}")
        cell2, want2 = h2.iloc[c], DOSE_LABELS[c % 3]
        if cell2 != want2:
            return (f"cell {excel_col(cols[nk + c])}{rows[1] + 1} is {cell2!r}, expected"
                    f" {want2!r}")
    return ""


def read_sheet(path, sheet, nk):
    """The inverse of write_sheet, with checks of the two header rows. Returns (key rows,
    number of weeks, values: rows x (3 * weeks), week-major)."""
    xl = pd.ExcelFile(path)
    name = xl.sheet_names[sheet] if isinstance(sheet, int) else sheet
    df = pd.read_excel(xl, sheet_name=sheet, header=None)
    r0, c0 = df.shape
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")     # empty columns and rows
    print(f"{path}: sheet {sheet} ('{name}', of {len(xl.sheet_names)}): {r0} rows x {c0} columns"
          f" read; {r0 - df.shape[0]} empty rows and {c0 - df.shape[1]} empty columns removed",
          flush=True)
    h1, h2 = df.iloc[0, nk:], df.iloc[1, nk:]
    nw = len(h1) // 3
    bad = header_mismatch(df, nk)
    if bad:
        raise ValueError(f"{path}, sheet {sheet}: header rows not Week w / Dose 1, Dose 2, Dose 3+:"
                         f" {bad}")
    keys, val = df.iloc[2:, :nk].values.tolist(), df.iloc[2:, nk:].to_numpy(float)
    tot = val.reshape(len(keys), nw, 3).sum(axis=(0, 1))
    print(f"   {nw} weeks (Week 1 ... Week {nw}), {len(keys)} rows of data ({keys[0]} ... "
          f"{keys[-1]}); doses in total: dose 1 {tot[0]:,.0f}, dose 2 {tot[1]:,.0f}, dose 3+"
          f" {tot[2]:,.0f}; empty cells {int(np.isnan(val).sum())}, negative {int((val < 0).sum())}",
          flush=True)
    rows, cols = df.index, df.columns
    for i, j in zip(*np.nonzero(np.isnan(val))):          # row-major: by key, then week, dose
        print(f"   empty cell {excel_col(cols[nk + j])}{rows[2 + i] + 1}: {keys[i]}, Week"
              f" {j // 3 + 1}, {DOSE_LABELS[j % 3]}", flush=True)
    return keys, nw, val


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
    D = np.rint(D)                          # floats with integer values; sheet 1 sums these exactly
    if not (D[0].sum(axis=1) > 0).all():
        raise ValueError("after rounding, a location has no first doses in some week")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    locs = list(dict.fromkeys(pop.state))
    wb = Workbook()
    wb.remove(wb.active)
    D1 = D.sum(axis=2)                                                        # sum over ages
    write_sheet(wb.create_sheet("by_state"), ["State"], [[s] for s in locs],
                D1.transpose(1, 2, 0).reshape(m, 3 * WEEKS))
    write_sheet(wb.create_sheet("by_state_age"), ["State", "Age"],
                [[s, a] for s, a in zip(pop.state, pop.age)],
                D.transpose(1, 2, 3, 0).reshape(m * n, 3 * WEEKS))
    wb.save(OUT)
    # check: the two sheets as read back, sheet 1 = sum of sheet 2 over the ages
    a1 = read_sheet(OUT, 0, 1)[2].reshape(m, WEEKS, 3)
    a2 = read_sheet(OUT, 1, 2)[2].reshape(m, n, WEEKS, 3).sum(axis=1)
    err = np.abs(a1 - a2).max()
    if err != 0 or (a1 != np.rint(a1)).any() or (a2 != np.rint(a2)).any():
        raise ValueError(f"sheets out of sync ({err:g}) or values not integers")
    print(f"doses 1, 2, 3 in total: " + ", ".join(f"{D[k].sum() / 1e6:.0f} M" for k in range(3))
          + f" (population {N.sum() / 1e6:.0f} M); integer values, sheets in sync exactly; wrote {OUT}", flush=True)


if __name__ == "__main__":
    build()
