"""From one results directory of the weekly simulation: the figures, and the run values in the
notes.

Usage:  python3 weekly_report.py RESULTS_DIR [nofigures]
RESULTS_DIR is weekly_simulation/results/<tag>/ (tag = <initial>__<doses>__<params>__<order>),
with at least the optimal run (simulate_optimal__<tag>.output and its two CSV files), and
optionally the eigenvector run (notes/DATA.tex, Section 2.3). <order> is the sweep order of the
run, the last part of the tag (for a tag without it, the order in the log, or 132). It
  1. makes the figures in RESULTS_DIR/figures/: the four panels of the optimal run
     (plot_weekly_optimal.py) and, with at least two strategies, the comparison
     (plot_compare.py);
     (not with nofigures, when the figures are made separately, as in weekly_simulation/run.sh);
  2. writes notes/run_values_<order>.tex, the \\newcommand definitions of the values of the run
     (input files, parameter set, days, weeks, figure folder, results);
  3. writes notes/DATA_<order>.tex and notes/simulation_weekly_optimal_<order>.tex, the
     templates notes/templates/DATA.tex and notes/templates/simulation_weekly_optimal.tex with \\input{run_values}
     replaced by \\input{run_values_<order>}.
Each file of an order is overwritten by the next report of a run of the same order; the files of
the other orders are kept. notes/stratified_seir_ngm.tex is not changed: it includes
run_values_<order>.tex of the order set in its preamble by hand.
The values come from the log of the optimal run (input paths, parameter set, t0, t_vacc, sweep
order, switch days), its CSV files, the initial state file (seropositive share) and the dose file (totals).
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
TEMPLATES = ["DATA", "simulation_weekly_optimal"]   # notes/templates/<name>.tex -> notes/<name>_<order>.tex
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
    """Input paths, parameter set and file, t0, t_vacc, sweep order and switch days from the
    optimal run's log (t0 = 251 and order 132 for a log written before these were reported). If
    the dose file of the log does not exist (renamed after the run), the file named by the tag
    (its second part) in the same directory."""
    log = os.path.join(resdir, f"simulate_optimal__{tag}.output")
    txt = open(log).read()
    get = lambda pat: re.search(pat, txt, re.M).group(1).strip()
    out = dict(initial=get(r"^initial state (.+)$"), doses=get(r"^doses (.+)$"),
               params=get(r"^parameters (\S+) of "), paramfile=get(r"^parameters \S+ of (.+?) \(vacc"),
               tvacc=float(get(r"vaccination from t = ([0-9.]+)")))
    m0 = re.search(r"initial state at t0 = ([0-9.]+)", txt)
    out["t0"] = float(m0.group(1)) if m0 else 251.0
    mo = re.search(r"^sweep order ([123]{3})$", txt, re.M)
    out["order"] = mo.group(1) if mo else "132"
    m = re.search(r"switch days ([0-9., ]+);", txt)
    out["switch"] = [float(s) for s in m.group(1).split(",")] if m else []
    if not os.path.isfile(out["doses"]):
        # the dose file renamed after the run: the file named by the tag, in the same directory
        parts = tag.split("__")
        alt = os.path.join(os.path.dirname(out["doses"]), parts[1] + os.path.splitext(out["doses"])[1]) \
            if len(parts) >= 3 else None
        if alt is None or not os.path.isfile(alt):
            raise FileNotFoundError(f"{log}: dose file {out['doses']} not found, nor {alt}")
        print(f"dose file of the log {out['doses']} not found; using {alt}, named by the tag", flush=True)
        out["doses"] = alt
    return out


def values(resdir):
    """The run values: {macro name: LaTeX text}."""
    tag = os.path.basename(os.path.normpath(resdir))
    L = read_log(resdir, tag)
    w = pd.read_csv(os.path.join(resdir, f"simulate_weekly_51x16_optimal__{tag}.csv"))
    r = pd.read_csv(os.path.join(resdir, f"simulate_weekly_51x16_optimal__{tag}_rho.csv"))
    W, tv, t0 = len(r), L["tvacc"], L["t0"]
    v = {"RunTag": tag.replace("_", r"\_\allowbreak{}"),
         "RunInitialFile": tt(L["initial"]), "RunDosesFile": tt(L["doses"]),
         "RunParams": L["params"], "RunTzero": f"{t0:g}", "RunTzeroDate": date(t0),
         "RunOrder": ", ".join(f"$\\chi_{c}$" for c in L["order"]),
         "RunTvacc": f"{tv:g}", "RunTvaccDate": date(tv),
         "RunWeeks": f"{W}", "RunTlast": f"{tv + 7 * (W - 1):g}", "RunTend": f"{tv + 7 * W:g}",
         "RunTendDate": date(tv + 7 * W), "RunDaysNoVacc": f"{tv - t0:g}",
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
    # the two inputs not from public sources
    v["RunInitialText"] = (r"the initial state at $t=\RunTzero$, \texttt{\RunInitialFile};"
                           r" $\RunSeroPct\%$ seropositive")
    v["RunDosesText"] = (r"the weekly doses by location and dose, \texttt{\RunDosesFile}: \RunWeeks{}"
                         r" weeks, with $\RunDosesOne$, $\RunDosesTwo$ and $\RunDosesThree$ million"
                         r" first, second and third doses")
    v["RunOtherInputs"] = "not from public sources"
    for s in ("eigenvector",):                       # the other strategy, if run
        p = os.path.join(resdir, f"simulate_weekly_51x16_{s}__{tag}.csv")
        if os.path.exists(p):
            ws, rs = pd.read_csv(p), pd.read_csv(p.replace(".csv", "_rho.csv"))
            key = s.capitalize()
            v[f"Cmp{key}DeathsEnd"] = f"{ws.M.iloc[-1] / 1e3:.0f}"
            v[f"Cmp{key}RhoAbove"] = f"{int((rs.rho_applied > 1).sum())}"
            v[f"Cmp{key}InfPeak"] = sig(ws[INFECTIOUS].sum(axis=1).max() / 1e6, 3)
            v[f"Cmp{key}HospPeak"] = sig(ws.Ih.max() / 1e6, 3)
    v.update(delta_values(mod, resdir, tag, tv, W))
    v["CmpOptimalDeathsEnd"], v["CmpOptimalRhoAbove"] = v["RunDeathsEnd"], v["RunRhoOptAbove"]
    v["CmpOptimalInfPeak"], v["CmpOptimalHospPeak"] = v["RunInfPeak"], v["RunHospPeak"]
    return tag, v


def delta_values(mod, resdir, tag, tv, W):
    """The bounds of stratified_seir_ngm.tex, Section 5.3, for the parameter set of the run:
    Delta_p > 0 (with chi_3 = 0) iff 0 < chi_1 < omega x*, x* = (1 - r eps_1) / (r (1 - 2 (1 -
    eps_1))), over the weeks of the run, for the first and the last eps_1; and the weeks of the
    optimal run in which a location has no group with Delta_p > 0 (from its log)."""
    taus = [tv + 7 * w for w in range(W)]
    ctx = [mod.ngm_context(t) for t in taus]
    r = np.concatenate([c[5] for c in ctx])
    om = ctx[0][6]
    eps = [c[8] for c in ctx]
    first_change = next((t for t, e in zip(taus, eps) if e != eps[0]), None)
    last_from = next(t for t, e in zip(taus, eps) if e == eps[-1])

    def bounds(e):
        if 2 * (1 - e) >= 1:                       # Delta_p > 0 for all chi_1 > 0
            return r"\infty", r"\infty"
        xs = lambda rr: (1 - rr * e) / (rr * (1 - 2 * (1 - e)))
        lo, hi = om * xs(r.max()), om * xs(r.min())
        tex = lambda x: f"{x / 10 ** np.floor(np.log10(x)):.1f}\\cdot10^{{{int(np.floor(np.log10(x)))}}}"
        return tex(lo), tex(hi)

    v = {"RunOmegaInv": f"{1 / om:g}", "RunRMin": f"{r.min():.3f}", "RunRMax": f"{r.max():.3f}",
         "RunEpsFirst": f"{eps[0]:g}"}
    v["RunDeltaBoundLo"], v["RunDeltaBoundHi"] = bounds(eps[0])
    v["RunEpsFirstTo"] = (f" in the weeks before $t={first_change:g}$" if first_change is not None else "")
    if eps[-1] != eps[0]:
        lo, hi = bounds(eps[-1])
        v["RunDeltaLast"] = (f" (with $\\varepsilon_1={eps[-1]:g}$, in the weeks from $t={last_from:g}$: between"
                             f" ${lo}$ and ${hi}$)")
    else:
        v["RunDeltaLast"] = ""
    txt = open(os.path.join(resdir, f"simulate_optimal__{tag}.output")).read()
    ts = sorted({int(t) for t in re.findall(r"\[t=(\d+) [^\]]*\] sweep \d+ dose 2 step: a location has no"
                                            r" group with Delta_p", txt)})
    v["RunDeltaText"] = (f"in {len(ts)} of the {W} weeks, the first at $t={ts[0]}$" if ts
                         else f"in none of the {W} weeks")
    return v


def write_values(tag, v, order):
    """notes/run_values_<order>.tex: the definitions of v."""
    path = os.path.join(NOTES, f"run_values_{order}.tex")
    users = ", ".join(f"notes/{t}_{order}.tex" for t in TEMPLATES) + " and notes/stratified_seir_ngm.tex"
    lines = ["% " + "=" * 92,
             "% THIS FILE IS GENERATED AUTOMATICALLY. DO NOT EDIT IT BY HAND: EVERY CHANGE IS LOST",
             "% THE NEXT TIME IT IS GENERATED.",
             "%",
             "% IT IS WRITTEN BY code/python/weekly_report.py (RUN BY weekly_simulation/run.sh, OR BY",
             "%     python3 code/python/weekly_report.py weekly_simulation/results/<tag> [nofigures]",
             f"% ) FROM THE RESULTS OF ONE RUN WITH SWEEP ORDER {order}, AND OVERWRITTEN BY THE NEXT ONE.",
             f"% THIS VERSION: weekly_simulation/results/{tag}/, {datetime.datetime.now():%Y-%m-%d %H:%M}.",
             "%",
             f"% IT IS INCLUDED BY {users}",
             f"% (IF ITS PREAMBLE INCLUDES run_values_{order}). RERUN pdflatex ON THEM AFTER IT IS GENERATED AGAIN.",
             "%",
             "% TO CHANGE A VALUE OR ITS WORDING, CHANGE weekly_report.py (FUNCTIONS values AND",
             "% delta_values) AND GENERATE THE FILE AGAIN. A NEW \\Run... MACRO USED IN THE NOTES MUST",
             "% BE ADDED THERE TOO, OR THE NOTES DO NOT COMPILE.",
             "% " + "=" * 92]
    lines += [f"\\newcommand{{\\{k}}}{{{x}}}" for k, x in v.items()]
    open(path, "w").write("\n".join(lines) + "\n")
    print(f"wrote {len(v)} values to {path}", flush=True)


def write_copy(name, order):
    """notes/<name>_<order>.tex: the template notes/templates/<name>.tex with \\input{run_values}
    replaced by \\input{run_values_<order>}."""
    src = os.path.join(NOTES, "templates", f"{name}.tex")
    lines = open(src).read().split("\n")
    i = [k for k, l in enumerate(lines) if l.strip() == "\\input{run_values}"]
    if len(i) != 1:
        raise ValueError(f"{src}: needs exactly one line \\input{{run_values}}")
    lines[i[0]] = f"\\input{{run_values_{order}}}"
    head = [f"% GENERATED by code/python/weekly_report.py from the template templates/{name}.tex for the",
            f"% sweep order {order}; do not edit by hand: edit templates/{name}.tex and generate again."]
    out = os.path.join(NOTES, f"{name}_{order}.tex")
    open(out, "w").write("\n".join(head + lines))
    print(f"wrote {out}", flush=True)


def run_order(resdir, tag):
    """The sweep order of the run: the last part of the tag, which must agree with the log; for a
    tag without it, the order in the log (132 if the log has none)."""
    log = read_log(resdir, tag)["order"]
    last = tag.rsplit("__", 1)[-1]
    if sorted(last) == ["1", "2", "3"]:
        if last != log:
            raise ValueError(f"{resdir}: order {last} in the tag, {log} in the log")
        return last
    return log


def main(resdir, figures=True):
    tag = os.path.basename(os.path.normpath(resdir))
    sys.path.insert(0, HERE)
    order = run_order(resdir, tag)
    if figures:
        import plot_weekly_optimal
        import plot_compare
        L = read_log(resdir, tag)
        plot_weekly_optimal.main(os.path.join(resdir, "figures"), L["initial"], L["doses"],
                                 L["params"], resdir, order)
        try:
            plot_compare.main(resdir)
        except FileNotFoundError as err:
            print(f"no comparison figures: {err}", flush=True)
    tag, v = values(resdir)
    write_values(tag, v, order)
    for name in TEMPLATES:
        write_copy(name, order)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) not in (1, 2) or (len(args) == 2 and args[1] != "nofigures"):
        raise SystemExit("usage: python3 weekly_report.py RESULTS_DIR [nofigures]")
    main(os.path.abspath(args[0]), figures=len(args) == 1)
