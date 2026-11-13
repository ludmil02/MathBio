"""Optimal allocation of one dose or of three doses, 51 x 16 model, week by week,
with the doses b_{k,i} read from an Excel file.

For each week w chosen, the sheet "week<w>" of the doses file gives, per location
(state), the first, second and third doses b1, b2, b3 available in that week (see
data/doses_51x16_mock.xlsx and README_three_doses.md).  The doses are divided by 7,
since chi is a rate per day.  The group sizes sigma_{k,p} = S_p, V_p, P_p are read
from the sheet "sigma_week<w>" (columns state, age, S, V, P, 816 rows in the order
of the groups) if it exists, and otherwise are the MOCK sizes of
three_doses_51x16_mock.py (the disease free equilibrium at chi_ref).

The weeks are independent: the model is not advanced from one week to the next,
so each week is solved with its own doses and the group sizes above.

  --mode 1   one dose: chi_2 = chi_3 = 0, minimization over chi_1 by the fixed point
             iteration of Section 5 of notes/stratified_seir_ngm.tex, from the uniform
             allocation (only b1 is used);
  --mode 3   three doses: the alternating method of Section 6 (chi_1, chi_3 by the
             fixed point iteration, chi_2 by spectral projected gradient), from the
             uniform allocation, the left eigenvector allocation and --random-starts
             random allocations; the end point with the smallest rho is kept.

Every end point is checked by the KKT conditions (4.1) of the notes, and rho(NGM)
is compared for chi = 0, the uniform allocation (the same rate for all groups of a
location), the left eigenvector allocation of the paper (doses in proportion to
the susceptible rows of the dominant left eigenvector of NGM at the uniform
allocation) and the minimizer.

Output:
  <out>.xlsx   sheet "summary", one row per week; sheet "week<w>" per week, the
               minimizer by state and age: rates chi1, chi2, chi3 (per day) and
               doses per week sigma chi 7;
  <out>_week<w>.npz   the allocations, the group sizes, the doses per day and the
               histories of rho, for each week.

Usage, from code/python:
    python3 -u optimize_doses_51x16.py --mode 3 --weeks 1 | tee intermediate/optimize_doses.log
    python3 -u optimize_doses_51x16.py --mode 1 --weeks 1-3 --file data/doses_51x16_mock.xlsx
    python3 -u optimize_doses_51x16.py --mode 3 --weeks all --random-starts 0
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

_argv, sys.argv = sys.argv, sys.argv[:1]       # three_doses_51x16_mock reads sys.argv when imported
import three_doses_51x16_mock as T
sys.argv = _argv
S7 = T.S7
N, n, m, LOC = T.N, T.n, T.m, T.LOC
T0 = time.time()


def say(*args):
    print("[%7.2fs]" % (time.time() - T0), *args, flush=True)


# ----------------------------------------------------------------------------- input
def week_sheets(path):
    """The weeks w with a sheet "week<w>" in path, in increasing order."""
    names = pd.ExcelFile(path).sheet_names
    return sorted(int(s[4:]) for s in names if s.startswith("week") and s[4:].isdigit())


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
        raise ValueError("no sheet week%s in the doses file (weeks found: %s)"
                         % (", week".join(map(str, missing)), available))
    return weeks


def mock_sigma():
    ref = [np.full(N, c) for c in T.CHI_REF]
    p1, p2, _ = T.pis(*ref)
    _, _, den = T.g_of(*ref)
    St = T.Nflat / den
    return [St, p1 * St, p2 * p1 * St]


def read_sigma(path, w):
    """[S, V, P] from the sheet "sigma_week<w>", or None if there is no such sheet."""
    sheet = "sigma_week%d" % w
    if sheet not in pd.ExcelFile(path).sheet_names:
        return None
    df = pd.read_excel(path, sheet_name=sheet)
    states = [s for s in S7.STATES for _ in range(n)]
    ages = list(S7.AGES) * m
    if list(df["state"]) != states or [str(a) for a in df["age"]] != ages:
        raise ValueError("%s, sheet %s: rows must be the 816 groups, state by state in the order "
                         "of ngm_seir7.STATES, ages %s within each state" % (path, sheet, " ".join(S7.AGES)))
    sig = [df[c].to_numpy(dtype=float) for c in ("S", "V", "P")]
    if not all(np.all(s > 0) for s in sig):
        raise ValueError("%s, sheet %s: S, V, P must be positive" % (path, sheet))
    return sig


# ----------------------------------------------------------------------------- allocations
def uniform(b, sig, ks):
    """The same rate for all groups of a location; zero for the doses not in ks."""
    return [np.concatenate([np.full(n, b[k][i] / sig[k][I].sum()) for i, I in enumerate(LOC)])
            if k in ks else np.zeros(N) for k in range(3)]


def left_ev(b, sig, at, ks):
    """Doses in proportion to the susceptible rows of the dominant left eigenvector
    of NGM at the allocation `at`."""
    Y, Z, _ = S7.yz(*(c.reshape(m, n) for c in at))
    H = S7.h_operator(Y, Z, S7.Qs, S7.Qm)[0]
    wS = np.abs(S7.perron(H, transpose=True)[2].reshape(N, 2)[:, 0])
    out = []
    for k in range(3):
        x = np.zeros(N)
        if k in ks:
            for i, I in enumerate(LOC):
                x[I] = b[k][i] * wS[I] / wS[I].sum() / sig[k][I]
        out.append(x)
    return out


def random_alloc(b, sig, rng, ks):
    out = []
    for k in range(3):
        x = np.zeros(N)
        if k in ks:
            for i, I in enumerate(LOC):
                u = rng.random(n)
                x[I] = b[k][i] * u / u.sum() / sig[k][I]
        out.append(x)
    return out


# ----------------------------------------------------------------------------- one week
def solve_week(b, sig, mode, nrand, seed):
    """Returns a dict with the allocations, rho, the histories and the KKT check."""
    ks = (0,) if mode == 1 else (0, 1, 2)
    uni = uniform(b, sig, ks)
    lev = left_ev(b, sig, uni, ks)
    res = dict(uniform=uni, left_ev=lev, hist={})
    if mode == 1:
        h = []
        c1, it = T.t_block(uni, 0, b, sig, tol=1e-13, kmax=1000, hist=h)
        h.append(T.rho_of(c1))
        h = np.array(h)
        inc = np.diff(h) / h[1:]
        say("fixed point iteration from the uniform allocation: %d iterations, rho = %.12f; "
            "iterations that increase rho by more than 1e-12 (relative): %d of %d"
            % (it, h[-1], int((inc > 1e-12).sum()), inc.size))
        res.update(minimizer=c1, best_start="uniform", steps=it, increases=int((inc > 1e-12).sum()))
        res["hist"]["uniform"] = h
    else:
        rng = np.random.default_rng(seed)
        starts = [("uniform", uni), ("left ev", lev)]
        starts += [("random %d" % s, random_alloc(b, sig, rng, ks)) for s in range(nrand)]
        ends = []
        for name, ch in starts:
            cm, rm, h = T.alternate([c.copy() for c in ch], b, sig, label="[%s]" % name)
            inc = np.diff(h) / h[1:]
            ends.append((rm, name, cm, (h.size - 1) // 3, int((inc > 1e-12).sum())))
            res["hist"][name] = h
        ends.sort(key=lambda x: x[0])
        say("end points from the %d starts: rho in [%.12f, %.12f]; smallest from the start %s"
            % (len(ends), ends[0][0], ends[-1][0], ends[0][1]))
        rm, name, cm, sw, nin = ends[0]
        res.update(minimizer=cm, best_start=name, steps=sw,
                   increases=sum(e[4] for e in ends), ends=[(e[1], e[0]) for e in ends])
    say("KKT conditions at the minimizer:")
    res["kkt"] = T.kkt_table(res["minimizer"], b, sig, doses=ks)
    zero = [np.zeros(N)] * 3
    res["rho"] = {name: T.rho_ngm(ch) for name, ch in
                  (("chi = 0", zero), ("uniform", uni), ("left ev", lev), ("minimizer", res["minimizer"]))}
    rmin = res["rho"]["minimizer"]
    for name, rr in res["rho"].items():
        say("  %-10s rho(NGM) = %.6f   excess over the minimizer: %7.3f%%" % (name, rr, 100 * (rr - rmin) / rmin))
    return res


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", type=int, choices=(1, 3), required=True, help="one dose or three doses")
    ap.add_argument("--weeks", default="1", help="'all', 'w', 'w1-w2' or 'w1,w2,...' (default 1)")
    ap.add_argument("--file", default=T.DOSES_XLSX, help="the doses file (default %(default)s)")
    ap.add_argument("--random-starts", type=int, default=3, help="three doses: random starts (default 3)")
    ap.add_argument("--chi2", choices=("spg", "tc", "pg"), default="spg",
                    help="three doses: method of the chi_2 step (default spg)")
    ap.add_argument("--out", default=None, help="output prefix (default intermediate/optimize_doses_mode<M>)")
    a = ap.parse_args()
    T.CHI2 = a.chi2
    out = a.out or os.path.join(S7.HERE, "intermediate", "optimize_doses_mode%d" % a.mode)

    if not os.path.exists(a.file) and a.file == T.DOSES_XLSX:
        T.write_mock_doses(a.file, mock_sigma())
    weeks = parse_weeks(a.weeks, week_sheets(a.file))
    say("doses file %s; mode %d (%s); weeks %s"
        % (a.file, a.mode, "one dose" if a.mode == 1 else "three doses", ", ".join(map(str, weeks))))
    ks = (1,) if a.mode == 1 else (1, 2, 3)

    rows, sheets = [], {}
    for w in weeks:
        say("=" * 20 + " week %d " % w + "=" * 20)
        b = T.read_doses(a.file, sheet="week%d" % w, doses=ks)
        sig = read_sigma(a.file, w)
        say("group sizes: %s" % ("sheet sigma_week%d" % w if sig is not None else "mock (disease free equilibrium at chi_ref)"))
        if sig is None:
            sig = mock_sigma()
        say("doses per week, US total: %s" % ", ".join("b%d %.4e" % (k, 7 * b[k - 1].sum()) for k in ks))
        res = solve_week(b, sig, a.mode, a.random_starts, seed=w)

        rmin = res["rho"]["minimizer"]
        row = dict(week=w, mode=a.mode,
                   rho_chi0=res["rho"]["chi = 0"], rho_uniform=res["rho"]["uniform"],
                   rho_left_ev=res["rho"]["left ev"], rho_minimizer=rmin,
                   excess_uniform_pct=100 * (res["rho"]["uniform"] - rmin) / rmin,
                   excess_left_ev_pct=100 * (res["rho"]["left ev"] - rmin) / rmin,
                   best_start=res["best_start"],
                   iterations_or_sweeps=res["steps"], steps_increasing_rho=res["increases"])
        for k, (sp, ra, dr, nz) in res["kkt"].items():
            row["kkt_spread_dose%d" % (k + 1)] = sp
            row["kkt_ratio_no_doses_dose%d" % (k + 1)] = ra if np.isfinite(ra) else np.nan
            row["groups_no_doses_dose%d" % (k + 1)] = nz
        rows.append(row)

        cm = res["minimizer"]
        tab = {"state": [s for s in S7.STATES for _ in range(n)], "age": list(S7.AGES) * m}
        for k in range(3):
            tab["chi%d_per_day" % (k + 1)] = cm[k]
        for k in range(3):
            tab["doses%d_per_week" % (k + 1)] = 7 * sig[k] * cm[k]
        sheets["week%d" % w] = pd.DataFrame(tab)
        np.savez("%s_week%d.npz" % (out, w), minimizer=np.array(cm), uniform=np.array(res["uniform"]),
                 left_ev=np.array(res["left_ev"]), sig=np.array(sig), b=np.array(b),
                 **{"hist_" + k.replace(" ", "_"): v for k, v in res["hist"].items()})

    with pd.ExcelWriter(out + ".xlsx", engine="openpyxl") as xw:
        pd.DataFrame(rows).to_excel(xw, sheet_name="summary", index=False)
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False)
    say("wrote %s.xlsx and %s_week<w>.npz" % (out, out))
    say("done")


if __name__ == "__main__":
    main()
