"""Weekly results of seir7_ode.py simulate optimal: rho, compartments, infections, deaths.

Inputs (produced by seir7_ode.py simulate optimal):
  data/simulate_weekly_51x16_optimal.csv      national totals at the start of each week
  data/simulate_weekly_51x16_optimal_rho.csv  rho per week, uniform and optimal allocation
Output, in notes/figures/ or in the directory given as the first argument:
  fig_weekly_optimal.pdf                      the four panels together
  fig_weekly_optimal_{rho,compartments,infectious,deaths}.png   each panel alone
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUTDIR = os.path.normpath(os.path.join(HERE, "..", "..", "notes", "figures"))
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "lines.linewidth": 1.6, "legend.frameon": False})


def day(t):
    """Day t of the model (t = 1 is 1 January 2020) as a date."""
    return np.datetime64("2020-01-01") + (np.asarray(t) - 1).astype(int).astype("timedelta64[D]")


def panel_rho(a, w, r):
    a.plot(day(r.t), r.rho_uniform, color=C[0], label="uniform allocation")
    a.plot(day(r.t), r.rho_opt, color=C[1], label="optimal allocation")
    a.axhline(1.0, color="#8a8984", lw=0.8, ls="--")
    a.set_title("Spectral radius of the next-generation matrix", loc="left")
    a.legend()


def panel_compartments(a, w, r):
    for k, (c, nm) in enumerate([("S", "susceptible S"), ("P", "P"), ("V", "vaccinated V"),
                                 ("R", "recovered R")]):
        a.plot(day(w.t), w[c] / 1e6, color=C[k], label=nm)
    a.set_title("Compartments (millions)", loc="left")
    a.legend()


def panel_infectious(a, w, r):
    infectious = w[["Ia", "Ips", "Ipm", "Is", "Im", "Ih"]].sum(axis=1)
    a.plot(day(w.t), infectious / 1e3, color=C[0], label="all infectious Ia+Ips+Ipm+Is+Im+Ih")
    a.plot(day(w.t), w.Ih / 1e3, color=C[1], label="hospitalized Ih")
    a.set_title("Infectious (thousands)", loc="left")
    a.legend(loc="upper left")


def panel_deaths(a, w, r):
    a.plot(day(w.t), w.M / 1e3, color=C[0])
    a.set_title("Deaths M, cumulative (thousands)", loc="left")


PANELS = {"rho": panel_rho, "compartments": panel_compartments,
          "infectious": panel_infectious, "deaths": panel_deaths}


def dates(a):
    a.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    a.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))


def main(outdir):
    w = pd.read_csv(os.path.join(DATA, "simulate_weekly_51x16_optimal.csv"))
    r = pd.read_csv(os.path.join(DATA, "simulate_weekly_51x16_optimal_rho.csv"))

    fig, ax = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True, constrained_layout=True)
    for a, f in zip(ax.ravel(), PANELS.values()):
        f(a, w, r)
    for a in ax[1]:
        dates(a)
    fig.suptitle(f"seir7_ode.py, optimal strategy, weekly from 14 Dec 2020 (t = {w.t.iloc[0]:g})"
                 f" to t = {w.t.iloc[-1]:g}")
    out = os.path.join(outdir, "fig_weekly_optimal.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)

    for name, f in PANELS.items():
        fig, a = plt.subplots(figsize=(5, 3.25), constrained_layout=True)
        f(a, w, r)
        dates(a)
        out = os.path.join(outdir, f"fig_weekly_optimal_{name}.png")
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else OUTDIR)
