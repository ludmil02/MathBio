#!/usr/bin/env python3
"""Compare the optimal runs that differ only in the sweep order of the alternating method.

Usage (from anywhere):
    python3 comparison_script.py [RESULTS] [OUT] [CFILE]

RESULTS  the results directory of run.sh (default: results/ next to this script)
OUT      the output directory (default: RESULTS/comparison/)
CFILE    the contact matrix (default: ../code/python/data/model/C_816x816.csv.gz from this script)

Before the comparison, the rows of the contact matrix C are summed exactly (math.fsum of the
stored entries); the largest |row sum - 1| is reported, in units of the machine epsilon, and
called 1 to machine precision when it is at most n eps (n the order of C).

The run directories RESULTS/<base>__<ORDER>/ (ORDER a permutation of 123) with the same <base>
(<initial>__<doses>__<params>) form a group; each group with two or more orders is compared
against the reference order (132 when present, otherwise the first).  Written to OUT:
    comparison_orders.tex      one document, a section per group: summary table, per-week table,
                               figures
    figures/<base>_*.png       rho at the optimum, the work of the optimization, the national
                               totals relative to the reference
Build with:  cd OUT && pdflatex comparison_orders.tex
Needs only numpy, pandas, matplotlib (no project modules).
"""
import glob
import math
import os
import sys
from itertools import permutations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDERS = {"".join(p) for p in permutations("123")}
REF = "132"
COLOURS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
COMPARTMENTS = ["S", "P", "Es", "Ep", "V", "Ia", "Ips", "Ipm", "Is", "Im", "Ih", "R", "M"]
DAY0 = pd.Timestamp("2019-12-31")          # t in days from 31 December 2019


def tex(s):
    return str(s).replace("_", r"\_")


def sci(x, digits=2):
    if x is None or not np.isfinite(x):
        return "--"
    if x == 0:
        return "$0$"
    m, e = f"{x:.{digits - 1}e}".split("e")
    return rf"${m}\times 10^{{{int(e)}}}$"


def check_C(path):
    """row sums of C, each summed exactly; lines of the report"""
    if not os.path.isfile(path):
        msg = f"contact matrix {path} not found; row sums not checked"
        print(msg)
        return [rf"Contact matrix \texttt{{{tex(path)}}} not found; row sums not checked.", ""]
    C = pd.read_csv(path, index_col=0, float_precision="round_trip").to_numpy(float)
    n = C.shape[0]
    eps = np.finfo(float).eps
    dev = np.array([math.fsum(r) for r in C]) - 1.0
    k = int(np.argmax(np.abs(dev)))
    worst = abs(dev[k])
    ok = worst <= n * eps
    verdict = "1 to machine precision" if ok else "NOT 1 to machine precision"
    print(f"contact matrix {path}: {C.shape[0]} x {C.shape[1]}; max |row sum - 1| = {worst:.2e}"
          f" = {worst / eps:.1f} eps (row {k + 1}); bound n eps = {n * eps:.2e}: {verdict}",
          flush=True)
    return [r"\section*{Contact matrix}", "",
            rf"\texttt{{{tex(path)}}}, {C.shape[0]} $\times$ {C.shape[1]}, the stored entries of "
            r"each row summed exactly (\texttt{math.fsum}): the largest "
            rf"$|\sum_{{pq}} c_{{ij,pq}} - 1|$ is {sci(worst)} $= {worst / eps:.1f}\,\varepsilon$ "
            rf"(row {k + 1}; $\varepsilon = {eps:.3g}$), against the bound "
            rf"$n\varepsilon = {sci(n * eps)[1:-1]}$ ($n = {n}$): the row sums are "
            rf"\textbf{{{verdict}}}.", ""]


def read_run(d, tag):
    def one(strategy, suffix):
        f = os.path.join(d, f"simulate_weekly_51x16_{strategy}__{tag}{suffix}.csv")
        return pd.read_csv(f) if os.path.isfile(f) else None
    run = {"rho": one("optimal", "_rho"), "tot": one("optimal", ""),
           "ev_rho": one("eigenvector", "_rho")}
    if run["rho"] is None or run["tot"] is None:
        return None
    r = run["rho"]
    r["it"] = r[[c for c in r.columns if c.startswith("it_chi")]].sum(axis=1)
    r["time"] = r[[c for c in r.columns if c.startswith("t_chi")]].sum(axis=1)
    return run


def groups(results):
    g = {}
    for d in sorted(glob.glob(os.path.join(results, "*__*"))):
        if not os.path.isdir(d):
            continue
        tag = os.path.basename(d)
        base, order = tag.rsplit("__", 1)
        if order in ORDERS:
            g.setdefault(base, {})[order] = (d, tag)
    return g


def rel(a, b):
    """max over rows of |b - a| / max |a|, per column"""
    return (b - a).abs().max() / a.abs().max().replace(0, np.nan)


def compare(base, members, figdir):
    runs = {o: read_run(d, t) for o, (d, t) in members.items()}
    missing = sorted(o for o, r in runs.items() if r is None)
    runs = {o: r for o, r in runs.items() if r is not None}
    if len(runs) < 2:
        return None
    orders = sorted(runs, key=lambda o: (o != REF, o))
    ref = orders[0]
    colour = dict(zip(orders, COLOURS))

    # the weeks completed by all orders
    t_rho = sorted(set.intersection(*(set(r["rho"]["t"]) for r in runs.values())))
    t_tot = sorted(set.intersection(*(set(r["tot"]["t"]) for r in runs.values())))
    rho = {o: r["rho"].set_index("t").loc[t_rho] for o, r in runs.items()}
    tot = {o: r["tot"].set_index("t").loc[t_tot] for o, r in runs.items()}
    comp = [c for c in COMPARTMENTS if c in tot[ref].columns]
    doses = [c for c in rho[ref].columns if c[:3] in ("D1_", "D2_", "D3_")]

    summary = []
    for o in orders:
        R, T = rho[o], tot[o]
        d_rho = (R["rho_opt"] - rho[ref]["rho_opt"]) / rho[ref]["rho_opt"]
        d_tot = rel(tot[ref][comp], T[comp])
        d_dos = ((R[doses] - rho[ref][doses]).abs().to_numpy().max()
                 / rho[ref][doses].abs().to_numpy().max()) if doses else np.nan
        ev = runs[o]["ev_rho"]
        ev_ref = runs[ref]["ev_rho"]
        d_ev = np.nan
        if ev is not None and ev_ref is not None and len(ev) == len(ev_ref):
            d_ev = ((ev["rho_applied"] - ev_ref["rho_applied"]).abs() / ev_ref["rho_applied"]).max()
        summary.append(dict(
            order=o, weeks=len(runs[o]["rho"]), sweeps=int(R["sweeps"].sum()),
            it=int(R["it"].sum()), time=R["time"].sum(),
            ode=R["t_ode"].sum() if "t_ode" in R else np.nan,
            mem=R["max_rss_MB"].max() if "max_rss_MB" in R else np.nan,
            d_rho=d_rho.abs().max(), lower=int((d_rho < 0).sum()), higher=int((d_rho > 0).sum()),
            deaths=T["M"].iloc[-1] if "M" in T else np.nan,
            d_tot=d_tot.max(), d_tot_worst=d_tot.idxmax() if d_tot.notna().any() else "--",
            d_dos=d_dos, d_ev=d_ev))

    weeks = np.arange(1, len(t_rho) + 1)
    figs = []

    # rho at the optimum and its difference relative to the reference
    fig, ax = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
    for o in orders:
        ax[0].plot(weeks, rho[o]["rho_opt"], "o-", ms=3, color=colour[o], label=o,
                   lw=2.5 if o == ref else 1.2, alpha=0.8)
        if o != ref:
            ax[1].plot(weeks, (rho[o]["rho_opt"] - rho[ref]["rho_opt"]) / rho[ref]["rho_opt"],
                       "o-", ms=3, color=colour[o], label=f"{o} vs {ref}")
    ax[0].set_ylabel("rho at the optimum")
    ax[0].legend(title="order")
    ax[1].axhline(0, color="0.6", lw=0.8)
    ax[1].set_ylabel(f"relative difference to {ref}")
    ax[1].set_xlabel("week")
    ax[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax[1].legend()
    fig.tight_layout()
    figs.append((f"{base}_rho.png", "Spectral radius at the optimum, per week, and its "
                 f"difference relative to order {ref}."))
    fig.savefig(os.path.join(figdir, figs[-1][0]), dpi=150)
    plt.close(fig)

    # the work of the optimization
    fig, ax = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
    w = 0.8 / len(orders)
    for k, o in enumerate(orders):
        x = weeks + (k - (len(orders) - 1) / 2) * w
        ax[0].bar(x, rho[o]["it"], w, color=colour[o], label=o)
        ax[1].bar(x, rho[o]["time"], w, color=colour[o], label=o)
    ax[0].set_ylabel("iterations (three doses)")
    ax[0].legend(title="order")
    ax[1].set_ylabel("optimization time (s)")
    ax[1].set_xlabel("week")
    fig.tight_layout()
    figs.append((f"{base}_work.png", "Iterations and time of the optimization, per week "
                 "(sum over the steps for the three doses)."))
    fig.savefig(os.path.join(figdir, figs[-1][0]), dpi=150)
    plt.close(fig)

    # national totals relative to the reference
    fig, ax = plt.subplots(figsize=(7, 3.5))
    tw = np.arange(len(t_tot))
    for o in orders[1:]:
        d = ((tot[o][comp] - tot[ref][comp]).abs() / tot[ref][comp].abs().max()
             .replace(0, np.nan)).max(axis=1)
        ax.semilogy(tw, d.where(d > 0), "o-", ms=3, color=colour[o], label=f"{o} vs {ref}")
    ax.set_xlabel("week (0: start of vaccination)")
    ax.set_ylabel("max relative difference")
    ax.legend()
    fig.tight_layout()
    figs.append((f"{base}_totals.png", "National totals: the largest over the compartments of "
                 f"the difference to order {ref}, relative to the largest value of the "
                 "compartment, at the start of each week and at the end."))
    fig.savefig(os.path.join(figdir, figs[-1][0]), dpi=150)
    plt.close(fig)

    return dict(base=base, orders=orders, ref=ref, missing=missing, summary=summary,
                rho=rho, t_rho=t_rho, figs=figs)


def section(c):
    ref, orders = c["ref"], c["orders"]
    L = [rf"\section{{{tex(c['base'])}}}", ""]
    L.append(f"Orders {', '.join(orders)}; reference order {ref}; "
             f"{len(c['t_rho'])} weeks completed by all.")
    if c["missing"]:
        L.append(f" Without results (not compared): {', '.join(c['missing'])}.")
    L += ["", r"\begin{center}\small", r"\begin{tabular}{lrrrrrrr}", r"\toprule",
          r"order & weeks & sweeps & iterations & optimization (s) & ODE (s) & memory (MB)"
          r" & deaths at the end \\", r"\midrule"]
    for s in c["summary"]:
        L.append(f"{s['order']} & {s['weeks']} & {s['sweeps']} & {s['it']} & {s['time']:.0f} & "
                 f"{s['ode']:.0f} & {s['mem']:.0f} & {s['deaths']:,.0f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{center}", ""]
    L += [r"\begin{center}\small", r"\begin{tabular}{lrrrrrr}", r"\toprule",
          rf"order & max rel.\ diff.\ of rho & weeks lower & weeks higher & totals (worst) & "
          rf"doses by age & eigenvector run \\", r"\midrule"]
    for s in c["summary"][1:]:
        L.append(f"{s['order']} & {sci(s['d_rho'])} & {s['lower']} & {s['higher']} & "
                 f"{sci(s['d_tot'])} ({s['d_tot_worst']}) & {sci(s['d_dos'])} & "
                 f"{sci(s['d_ev'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{center}", "",
          rf"Differences relative to order {ref}: rho at the optimum, the largest over the weeks "
          r"of the difference divided by the value of the reference; the weeks in which it is "
          r"lower or higher than with the reference; the national totals, the largest over the "
          r"weeks and compartments of the difference divided by the largest value of the "
          r"compartment; the doses by age, the largest difference divided by the largest dose; "
          r"the eigenvector run (the same strategy in both, a check), rho of the applied rates.",
          ""]

    # per week
    others = orders[1:]
    cols = "rl" + "r" + "r" * len(others) + "r" * len(orders) * 2 + "l" * len(orders)
    head = (["week", "date", f"rho {ref}"] + [f"{o}" for o in others]
            + [f"it.\\ {o}" for o in orders] + [f"s {o}" for o in orders]
            + [f"start {o}" for o in orders])
    L += [r"\begin{center}\scriptsize", rf"\begin{{longtable}}{{{cols}}}", r"\toprule",
          " & ".join(head) + r" \\", r"\midrule", r"\endhead"]
    abbrev = {"uniform": "u", "eigenvector": "e", "eigenvector run": "r"}
    R = c["rho"]
    for k, t in enumerate(c["t_rho"]):
        row = [str(k + 1), (DAY0 + pd.Timedelta(days=int(t))).strftime("%Y-%m-%d"),
               f"{R[ref].at[t, 'rho_opt']:.10f}"]
        row += [sci((R[o].at[t, "rho_opt"] - R[ref].at[t, "rho_opt"]) / R[ref].at[t, "rho_opt"])
                for o in others]
        row += [str(int(R[o].at[t, "it"])) for o in orders]
        row += [f"{R[o].at[t, 'time']:.1f}" for o in orders]
        row += [abbrev.get(str(R[o].at[t, "start"]), tex(R[o].at[t, "start"]))
                if "start" in R[o] else "--" for o in orders]
        L.append(" & ".join(row) + r" \\")
    L += [r"\bottomrule", r"\end{longtable}", r"\end{center}", "",
          rf"Per week: rho at the optimum with order {ref}, and with the other orders its "
          r"difference relative to it; the iterations and the time (s) of the optimization; the "
          r"start of the best result (u: uniform, e: eigenvector strategy at the state of the "
          r"run, r: the eigenvector run of the same week).", ""]
    for f, cap in c["figs"]:
        L += [r"\begin{figure}[htbp]\centering",
              rf"\includegraphics[width=0.85\textwidth]{{figures/{f}}}",
              rf"\caption{{{cap}}}", r"\end{figure}", ""]
    L.append(r"\clearpage")
    return L


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    results = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "results"))
    out = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else os.path.join(results, "comparison"))
    cfile = os.path.abspath(sys.argv[3] if len(sys.argv) > 3 else
                            os.path.join(here, "..", "code", "python", "data", "model",
                                         "C_816x816.csv.gz"))
    figdir = os.path.join(out, "figures")
    os.makedirs(figdir, exist_ok=True)

    c_lines = check_C(cfile)
    done, single = [], []
    for base, members in groups(results).items():
        c = compare(base, members, figdir) if len(members) > 1 else None
        if c is None:
            single.append((base, sorted(members)))
        else:
            done.append(c)
            print(f"{base}: orders {', '.join(c['orders'])}")
    L = [r"\documentclass[11pt]{article}",
         r"\usepackage[margin=2cm]{geometry}",
         r"\usepackage{booktabs,longtable,graphicx}",
         r"\title{Sweep orders of the alternating method compared}",
         r"\author{}", r"\date{\today}",
         r"\begin{document}", r"\maketitle", "",
         rf"Results in \texttt{{{tex(results)}}}.  Runs with the same initial state, doses and "
         r"parameter set and different sweep orders are compared.", ""]
    L += c_lines
    if single:
        L += [r"Not compared (fewer than two orders with results):", r"\begin{itemize}"]
        L += [rf"\item \texttt{{{tex(b)}}}: {', '.join(o)}" for b, o in single]
        L += [r"\end{itemize}", ""]
    for c in done:
        L += section(c)
    L.append(r"\end{document}")
    f = os.path.join(out, "comparison_orders.tex")
    with open(f, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"wrote {f} ({len(done)} groups compared); build: cd {out} && pdflatex comparison_orders.tex")


if __name__ == "__main__":
    main()
