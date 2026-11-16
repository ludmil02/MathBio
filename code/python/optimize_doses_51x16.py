"""Optimal allocation of three doses, 51 x 16 model, week by week, data read from an Excel file.

The doses are read from the first sheet of an Excel file (default
data/doses_51x16_mock.xlsx, see mock_doses_51x16.py):
  row 1        "Week1" over columns B-D, "Week2" over E-G, ... (column A empty);
  row 2        "State", then "Dose1", "Dose2", "Dose3" for each week;
  rows 3-53    the 51 locations (two-letter codes in column A, matched to ngm_seir7.STATES
               by code), and for each week the number of people vaccinated with dose 1, 2, 3
               at the location in that week: b_{1,i}, b_{2,i}, b_{3,i}, divided by 7, since
               chi is a rate per day.
The group sizes sigma_{1,p}, sigma_{2,p}, sigma_{3,p} are the MOCK ones of
mock_doses_51x16.mock_sigma (the disease free equilibrium at chi_ref), the same in every
week, until real ones are available.  The model data are those of ngm_seir7 (the public
data of notes/DATA.tex, with the parameters at t = 349).

The method (alternating.py, Section 6 of notes/stratified_seir_ngm.tex): the alternating
method, each step by the damped fixed point iteration, from the uniform allocation, the
left eigenvector allocation and --random-starts random allocations; the end point with the
smallest rho is the solution, and the KKT conditions (4.1) are checked there.  The weeks are
independent.

Output, for each week w:
  <out>_week<w>.csv   the solution by state and age: rates chi1, chi2, chi3 per day and
                      doses per week 7 sigma chi;
and <out>_summary.csv, one row per week and start: rho at the end point, sweeps, and
whether it is the solution, with the KKT spreads at the solution.

Usage, from code/python:
    python3 -u mock_doses_51x16.py                       # once: the mock Excel file
    python3 -u optimize_doses_51x16.py --weeks all | tee intermediate/optimize_doses.output
"""

import argparse
import os

import numpy as np
import pandas as pd

import alternating as A
import ngm_seir7 as S7
from mock_doses_51x16 import mock_sigma

say = A.say
m, n, N = A.m, A.n, A.N


XLSX = os.path.join(S7.HERE, "data", "doses_51x16_mock.xlsx")


def read_doses(path):
    """All weeks of the first sheet: a dict week -> [b1, b2, b3] per day (each of length 51,
    in the order of ngm_seir7.STATES).  Checks the two header rows and the state codes."""
    raw = pd.read_excel(path, sheet_name=0, header=None)
    head1, head2 = raw.iloc[0], raw.iloc[1]
    if str(head2[0]).strip() != "State":
        raise ValueError("%s: cell A2 must be 'State'" % path)
    states = [str(s).strip() for s in raw.iloc[2:, 0]]
    if sorted(states) != sorted(S7.STATES):
        raise ValueError("%s: rows 3-53 of column A must be the 51 codes %s" % (path, " ".join(S7.STATES)))
    order = [states.index(s) for s in S7.STATES]          # row of each location of the model
    data = raw.iloc[2:, :].to_numpy()[order]
    out = {}
    for c0 in range(1, raw.shape[1], 3):
        label = str(head1[c0]).strip()
        if not (label.lower().startswith("week") and label[4:].strip().isdigit()):
            raise ValueError("%s: expected 'Week<w>' in row 1, column %d, found %r" % (path, c0 + 1, label))
        if [str(head2[c0 + k]).strip() for k in range(3)] != ["Dose1", "Dose2", "Dose3"]:
            raise ValueError("%s: expected Dose1, Dose2, Dose3 in row 2 under %s" % (path, label))
        w = int(label[4:])
        b = [data[:, c0 + k].astype(float) / 7.0 for k in range(3)]
        if not all(np.all(x > 0) for x in b):
            raise ValueError("%s, %s: the doses must be positive at every location" % (path, label))
        out[w] = b
    return out


def parse_weeks(spec, available):
    """'all', 'w', 'w1-w2' or 'w1,w2,...'."""
    if spec == "all":
        return available
    weeks = []
    for part in spec.split(","):
        if "-" in part:
            a, b = (int(x) for x in part.split("-"))
            weeks += list(range(a, b + 1))
        else:
            weeks.append(int(part))
    missing = [w for w in weeks if w not in available]
    if missing:
        raise ValueError("no week(s) %s in the doses file (weeks found: %s)" % (missing, available))
    return weeks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weeks", default="all", help="'all', 'w', 'w1-w2' or 'w1,w2,...' (default all)")
    ap.add_argument("--file", default=XLSX, help="the Excel file of the weekly doses "
                                                  "(default data/doses_51x16_mock.xlsx)")
    ap.add_argument("--random-starts", type=int, default=3, help="random starts (default 3)")
    ap.add_argument("--theta", type=float, default=0.5, help="damping, 0 < theta <= 1 (default 0.5)")
    ap.add_argument("--out", default=os.path.join(S7.HERE, "intermediate", "optimize_doses"),
                    help="output prefix (default intermediate/optimize_doses)")
    a = ap.parse_args()
    A.THETA = a.theta
    doses = read_doses(a.file)
    weeks = parse_weeks(a.weeks, sorted(doses))
    sig = mock_sigma()
    say("doses %s; weeks %s; %d starts; damping %g; group sizes: mock (disease free equilibrium at chi_ref)"
        % (a.file, ", ".join(map(str, weeks)), 2 + a.random_starts, a.theta))
    rows = []
    for w in weeks:
        say("=" * 20 + " week %d " % w + "=" * 20)
        b = doses[w]
        say("doses per week, US total: b1 %.4e, b2 %.4e, b3 %.4e" % tuple(7 * x.sum() for x in b))
        ends = []
        for name, chi in A.starts(b, sig, a.random_starts, seed=w):
            cm, rm, sw, _ = A.alternate(chi, b, sig, label="[%s]" % name)
            ends.append((rm, name, cm, sw))
        best = min(ends, key=lambda e: e[0])
        rmin, bname, chi, _ = best
        say("solution: rho = %.12f from the start %s; end points from the %d starts: rho in [%.12f, %.12f]"
            % (rmin, bname, len(ends), rmin, max(e[0] for e in ends)))
        say("rho(NGM) of order 2N at the solution: %.12f" % A.rho_ngm(chi))
        say("KKT conditions at the solution:")
        kkt = A.kkt_table(chi, b, sig)
        for rm, name, _, sw in ends:
            row = dict(week=w, start=name, rho=rm, sweeps=sw, solution=(name == bname))
            for k in range(3):
                row["kkt_spread_dose%d" % (k + 1)] = kkt[k][0] if name == bname else np.nan
            rows.append(row)
        tab = {"state": [s for s in S7.STATES for _ in range(n)], "age": list(S7.AGES) * m}
        for k in range(3):
            tab["chi%d_per_day" % (k + 1)] = chi[k]
        for k in range(3):
            tab["doses%d_per_week" % (k + 1)] = 7 * sig[k] * chi[k]
        pd.DataFrame(tab).to_csv("%s_week%d.csv" % (a.out, w), index=False)
    pd.DataFrame(rows).to_csv(a.out + "_summary.csv", index=False)
    say("wrote %s_week<w>.csv and %s_summary.csv" % (a.out, a.out))


if __name__ == "__main__":
    main()
