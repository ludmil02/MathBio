"""From one results directory of the weekly simulation: the figures, and the run values in the
notes.

Usage:  python3 weekly_report.py RESULTS_DIR [nofigures]
RESULTS_DIR is weekly_simulation/results/<tag>/ (tag = <initial>__<doses>__<params>), with at
least the optimal run (simulate_optimal__<tag>.output and its two CSV files), and optionally
the eigenvector run (notes/DATA.tex, Section 2.3). It
  1. makes the figures in RESULTS_DIR/figures/: the four panels of the optimal run
     (plot_weekly_optimal.py) and, with at least two strategies, the comparison
     (plot_compare.py);
     (not with nofigures, when the figures are made separately, as in weekly_simulation/run.sh);
  2. rewrites, in notes/DATA.tex and notes/simulation_weekly_optimal.tex, the lines between
        % BEGIN run values ... and % END run values
     with \\newcommand definitions of the values of the run (input files, parameter set, days,
     weeks, figure folder, results); the rest of each file is not touched.
The values come from the log of the optimal run (input paths, parameter set, t_vacc, switch
days), its CSV files, the initial state file (seropositive share) and the dose file (totals).
"""
import datetime
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))          # ~/claude/MathBio
NOTES = os.path.join(ROOT, "notes")
TEX = ["DATA.tex", "simulation_weekly_optimal.tex"]
BEGIN, END = "% BEGIN run values", "% END run values"
INFECTIOUS = ["Ia", "Ips", "Ipm", "Is", "Im", "Ih"]


def date(t):
    """Day t of the model (days from 31 December 2019) as '1 January 2022'."""
    d = datetime.date(2019, 12, 31) + datetime.timedelta(days=int(round(float(t))))
    return f"{d.day} {d:%B %Y}"


def sig(x, n=2):
    """x to n significant digits, without an exponent: 5.53 -> 5.5, 40.4 -> 40, 131.9 -> 130."""
    if x == 0:
        return "0"
    r = round(x, n - 1 - int(np.floor(np.log10(abs(x)))))
    return f"{r:.{max(0, n - 1 - int(np.floor(np.log10(abs(x)))))}f}"


def tt(path):
    """A file name for \\texttt: relative to MathBio if inside it, with _ escaped and a possible
    line break after each / and _."""
    p = os.path.relpath(path, ROOT) if os.path.abspath(path).startswith(ROOT + os.sep) else path
    return p.replace("_", r"\_\allowbreak{}").replace("/", r"/\allowbreak{}")


def read_log(resdir, tag):
    """Input paths, parameter set and file, t_vacc and switch days from the optimal run's log."""
    log = os.path.join(resdir, f"simulate_optimal__{tag}.output")
    txt = open(log).read()
    get = lambda pat: re.search(pat, txt, re.M).group(1).strip()
    out = dict(initial=get(r"^initial state (.+)$"), doses=get(r"^doses (.+)$"),
               params=get(r"^parameters (\S+) of "), paramfile=get(r"^parameters \S+ of (.+?) \(vacc"),
               tvacc=float(get(r"vaccination from t = ([0-9.]+)")))
    m = re.search(r"switch days ([0-9., ]+);", txt)
    out["switch"] = [float(s) for s in m.group(1).split(",")] if m else []
    return out


def values(resdir):
    """The run values: {macro name: LaTeX text}."""
    tag = os.path.basename(os.path.normpath(resdir))
    L = read_log(resdir, tag)
    w = pd.read_csv(os.path.join(resdir, f"simulate_weekly_51x16_optimal__{tag}.csv"))
    r = pd.read_csv(os.path.join(resdir, f"simulate_weekly_51x16_optimal__{tag}_rho.csv"))
    W, tv = len(r), L["tvacc"]
    v = {"RunTag": tt(tag),
         "RunInitialFile": tt(L["initial"]), "RunDosesFile": tt(L["doses"]),
         "RunParams": L["params"], "RunTvacc": f"{tv:g}", "RunTvaccDate": date(tv),
         "RunWeeks": f"{W}", "RunTlast": f"{tv + 7 * (W - 1):g}", "RunTend": f"{tv + 7 * W:g}",
         "RunTendDate": date(tv + 7 * W), "RunDaysNoVacc": f"{tv - 251:g}",
         "RunFigDir": os.path.relpath(os.path.join(resdir, "figures"), NOTES) + "/"}
    sw = L["switch"]
    v["RunSwitch"] = ("no change of the parameters" if not sw else
                      ", ".join(f"$t={s:g}$" for s in sw[:-1]) + (" and " if len(sw) > 1 else "")
                      + f"$t={sw[-1]:g}$ (" + ", ".join(date(s) for s in sw) + ")")
    cap = np.log(100) / 7
    v.update(RunRhoOptAbove=f"{int((r.rho_opt > 1).sum())}",
             RunCapWeeks=f"{int(np.isclose(r.chi_max_before_cap, cap, rtol=1e-6).sum())}")
    inf = w[INFECTIOUS].sum(axis=1)
    for name, s, scale, n in (("V", w.V, 1e6, 3), ("Inf", inf, 1e6, 3), ("Hosp", w.Ih, 1e6, 3)):
        i = int(s.idxmax())
        v[f"Run{name}Peak"], v[f"Run{name}PeakDate"] = sig(s[i] / scale, n), date(w.t[i])
    v["RunDeathsStart"] = f"{w.M[np.isclose(w.t, tv)].iloc[0] / 1e3:.0f}"
    v["RunDeathsEnd"] = f"{w.M.iloc[-1] / 1e3:.0f}"
    sys.path.insert(0, HERE)
    import seir7_ode as so
    from doses_51x16x16 import read_sheet
    mod = so.Model(initial=L["initial"], params=L["params"], params_file=L["paramfile"])
    X = mod.x251.reshape(13, mod.m, mod.n)
    v["RunSeroPct"] = sig(100 * X[so.NAMES.index("R")].sum() / X.sum())
    _, nw, val = read_sheet(L["doses"], 0, 1)
    tot = val.reshape(-1, nw, 3).sum(axis=(0, 1)) / 1e6
    v.update(RunDosesOne=f"{tot[0]:.0f}", RunDosesTwo=f"{tot[1]:.0f}", RunDosesThree=f"{tot[2]:.0f}")
    for s in ("eigenvector",):                       # the other strategy, if run
        p = os.path.join(resdir, f"simulate_weekly_51x16_{s}__{tag}.csv")
        if os.path.exists(p):
            ws, rs = pd.read_csv(p), pd.read_csv(p.replace(".csv", "_rho.csv"))
            key = s.capitalize()
            v[f"Cmp{key}DeathsEnd"] = f"{ws.M.iloc[-1] / 1e3:.0f}"
            v[f"Cmp{key}RhoAbove"] = f"{int((rs.rho_applied > 1).sum())}"
            v[f"Cmp{key}InfPeak"] = sig(ws[INFECTIOUS].sum(axis=1).max() / 1e6, 3)
            v[f"Cmp{key}HospPeak"] = sig(ws.Ih.max() / 1e6, 3)
    v["CmpOptimalDeathsEnd"], v["CmpOptimalRhoAbove"] = v["RunDeathsEnd"], v["RunRhoOptAbove"]
    v["CmpOptimalInfPeak"], v["CmpOptimalHospPeak"] = v["RunInfPeak"], v["RunHospPeak"]
    return tag, v


def write_block(path, tag, v):
    """Replace the lines between BEGIN and END in path by the definitions of v."""
    lines = open(path).read().split("\n")
    b = [i for i, l in enumerate(lines) if l.startswith(BEGIN)]
    e = [i for i, l in enumerate(lines) if l.startswith(END)]
    if len(b) != 1 or len(e) != 1 or e[0] < b[0]:
        raise ValueError(f"{path}: needs exactly one '{BEGIN}' line and one '{END}' line after it")
    block = [f"{BEGIN}: written by code/python/weekly_report.py from weekly_simulation/results/{tag}/"]
    block += [f"\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{x}}}" for k, x in v.items()]
    lines[b[0]:e[0]] = block
    open(path, "w").write("\n".join(lines))
    print(f"wrote {len(v)} values to {path}", flush=True)


def main(resdir, figures=True):
    tag = os.path.basename(os.path.normpath(resdir))
    sys.path.insert(0, HERE)
    if figures:
        import plot_weekly_optimal
        import plot_compare
        L = read_log(resdir, tag)
        plot_weekly_optimal.main(os.path.join(resdir, "figures"), L["initial"], L["doses"],
                                 L["params"], resdir)
        try:
            plot_compare.main(resdir)
        except FileNotFoundError as err:
            print(f"no comparison figures: {err}", flush=True)
    tag, v = values(resdir)
    for name in TEX:
        p = os.path.join(NOTES, name)
        if os.path.exists(p):
            write_block(p, tag, v)
        else:
            print(f"{p}: not found, skipped", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) not in (1, 2) or (len(args) == 2 and args[1] != "nofigures"):
        raise SystemExit("usage: python3 weekly_report.py RESULTS_DIR [nofigures]")
    main(os.path.abspath(args[0]), figures=len(args) == 1)
