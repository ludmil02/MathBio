"""Compare the weekly strategies of seir7_ode.py simulate: optimal and eigenvector
(notes/DATA.tex, Section 2.3), from the results of one set of inputs.

Usage:  python3 plot_compare.py RESULTS_DIR
RESULTS_DIR is weekly_simulation/results/<tag>/ (tag = <initial>__<doses>__<params>); it holds,
for each strategy that was run,
  simulate_weekly_51x16_<strategy>__<tag>.csv      national totals at the start of each week
  simulate_weekly_51x16_<strategy>__<tag>_rho.csv  rho of the applied rates and doses by age, per week
Output, in RESULTS_DIR/figures/, each figure alone:
  fig_compare_rho.png            rho of the applied rates, per week
  fig_compare_infectious.png     all infectious Ia+Ips+Ipm+Is+Im+Ih (thousands)
  fig_compare_hospitalized.png   hospitalized Ih (thousands)
  fig_compare_deaths.png         cumulative deaths M (thousands)
  fig_compare_doses_age_dose<k>.png   doses of type k by age group, summed over the weeks (millions)
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_weekly_optimal import C, day, dates

STRATEGIES = [("eigenvector", "eigenvector", C[2]), ("optimal", "optimal", C[1])]       # fixed colour per strategy, in any figure
INFECTIOUS = ["Ia", "Ips", "Ipm", "Is", "Im", "Ih"]


def load(resdir):
    """{strategy: (weekly totals, per-week rho and doses)} for the strategies found."""
    tag = os.path.basename(os.path.normpath(resdir))
    out = {}
    for s, _, _ in STRATEGIES:
        p = os.path.join(resdir, f"simulate_weekly_51x16_{s}__{tag}.csv")
        if os.path.exists(p):
            r = pd.read_csv(p.replace(".csv", "_rho.csv"))
            if "rho_applied" not in r:                     # runs made before the column existed
                r["rho_applied"] = r["rho_capped"]
            out[s] = (pd.read_csv(p), r)
    if len(out) < 2:
        raise FileNotFoundError(f"{resdir}: fewer than two strategies to compare ({sorted(out)})")
    return tag, out


def lines(ax, data, f, title, one=False):
    """One line per strategy, f(table) against the table's t; the strategies are named in the
    legend only (the curves can be too close for labels at their ends)."""
    for s, label, c in STRATEGIES:
        if s in data:
            ax.plot(day(data[s].t), f(data[s]), color=c, label=label)
    if one:
        ax.axhline(1.0, color="#8a8984", lw=0.8, ls="--")
    ax.set_title(title, loc="left")
    ax.legend(loc="best")
    dates(ax)


def save(fig, outdir, name):
    out = os.path.join(outdir, name)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out, flush=True)


def main(resdir):
    tag, runs = load(resdir)
    outdir = os.path.join(resdir, "figures")
    os.makedirs(outdir, exist_ok=True)
    print(f"{tag}: strategies {', '.join(s for s, _, _ in STRATEGIES if s in runs)}", flush=True)
    size = (5.6, 3.25)

    fig, a = plt.subplots(figsize=size, constrained_layout=True)
    lines(a, {s: r for s, (w, r) in runs.items()}, lambda y: y.rho_applied,
          "Spectral radius of the next-generation matrix", one=True)
    save(fig, outdir, "fig_compare_rho.png")

    for name, f, title in [
            ("infectious", lambda y: y[INFECTIOUS].sum(axis=1) / 1e3, "All infectious (thousands)"),
            ("hospitalized", lambda y: y.Ih / 1e3, "Hospitalized Ih (thousands)"),
            ("deaths", lambda y: y.M / 1e3, "Deaths M, cumulative (thousands)")]:
        fig, a = plt.subplots(figsize=size, constrained_layout=True)
        lines(a, {s: w for s, (w, r) in runs.items()}, f, title)
        save(fig, outdir, f"fig_compare_{name}.png")

    with_doses = [s for s, _, _ in STRATEGIES if s in runs and any(c.startswith("D1_") for c in runs[s][1])]
    for k in (1, 2, 3):
        if len(with_doses) < 2:
            print("doses by age: fewer than two strategies with doses by age; skipped", flush=True)
            break
        cols = [c for c in runs[with_doses[0]][1] if c.startswith(f"D{k}_")]
        ages = [c.split("_", 1)[1] for c in cols]
        fig, a = plt.subplots(figsize=(6.4, 3.25), constrained_layout=True)
        wbar = 0.8 / len(with_doses)
        x = np.arange(len(ages))
        for i, s in enumerate(with_doses):
            label, c = next((l, c) for t, l, c in STRATEGIES if t == s)
            a.bar(x + (i - (len(with_doses) - 1) / 2) * wbar, runs[s][1][cols].sum() / 1e6,
                  width=wbar * 0.92, color=c, label=label)
        a.set_xticks(x, ages, rotation=45, ha="right", fontsize=7)
        a.set_title(f"Dose {k}" + ("+" if k == 3 else "") + " by age group, all weeks (millions)",
                    loc="left")
        a.grid(axis="x", visible=False)
        a.legend()
        save(fig, outdir, f"fig_compare_doses_age_dose{k}.png")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 plot_compare.py RESULTS_DIR")
    main(os.path.abspath(sys.argv[1]))
