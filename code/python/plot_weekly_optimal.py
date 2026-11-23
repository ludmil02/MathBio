"""Weekly results of seir7_ode.py simulate optimal: rho, compartments, infections, deaths.

Usage:  python3 plot_weekly_optimal.py [initial=FILE] [doses=FILE] [params=NAME] [out=DIR] [FIGDIR]
with initial=, doses=, params= and out= as given to seir7_ode.py simulate optimal (default: the
default files and set, and the results in data/). The figures go to FIGDIR; without it, to DIR
(out=); one of the two is required.

Inputs (produced by seir7_ode.py simulate optimal, tag = <initial>__<doses>__<params>__<order>, seir7_ode.run_tag):
  data/simulate_weekly_51x16_optimal__<tag>.csv      national totals at the start of each week
  data/simulate_weekly_51x16_optimal__<tag>_rho.csv  rho per week (rho_opt is plotted)
  data/simulate_weekly_51x16_optimal__<tag>_age.csv  compartments by age group (if the run saved them)
Output, in FIGDIR or DIR:
  fig_weekly_optimal_{rho,compartments,infectious,deaths}.png   one panel each
  fig_weekly_optimal_compartments_age<jj>.png   S, P, V, R of age group jj = 01, 02, ..., one each
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
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "lines.linewidth": 1.6, "legend.frameon": False})


def day(t):
    """Day t of the model (t = 1 is 1 January 2020) as a date."""
    return np.datetime64("2020-01-01") + (np.asarray(t) - 1).astype(int).astype("timedelta64[D]")


def panel_rho(a, w, r):
    a.plot(day(r.t), r.rho_opt, color=C[1])
    a.axhline(1.0, color="#8a8984", lw=0.8, ls="--")
    a.set_title("Spectral radius ρ(H), optimal", loc="left")


def panel_compartments(a, w, r):
    for k, (c, nm) in enumerate([("S", "susceptible S"), ("P", "partially immune P"), ("V", "vaccinated V"),
                                 ("R", "recovered R")]):
        a.plot(day(w.t), w[c] / 1e6, color=C[k], label=nm)
    a.set_title("Compartments, optimal (millions)", loc="left")
    a.legend()


def panel_infectious(a, w, r):
    infectious = w[["Ia", "Ips", "Ipm", "Is", "Im", "Ih"]].sum(axis=1)
    a.plot(day(w.t), infectious / 1e3, color=C[0], label="all infectious Ia+Ips+Ipm+Is+Im+Ih")
    a.plot(day(w.t), w.Ih / 1e3, color=C[1], label="hospitalized Ih")
    a.set_title("Infectious, optimal (thousands)", loc="left")
    a.legend(loc="upper left")


def panel_deaths(a, w, r):
    a.plot(day(w.t), w.M / 1e3, color=C[0])
    a.set_title("Deaths M, optimal (thousands)", loc="left")


PANELS = {"rho": panel_rho, "compartments": panel_compartments,
          "infectious": panel_infectious, "deaths": panel_deaths}


def dates(a):
    """Date ticks, by the span of the plotted days: every month up to about 7 months, every 3
    months up to about 14 months, else January and July."""
    lo, hi = a.get_xlim()
    span = hi - lo                                   # days
    months = (range(1, 13) if span <= 215 else (1, 4, 7, 10) if span <= 430 else (1, 7))
    a.xaxis.set_major_locator(mdates.MonthLocator(bymonth=months))
    a.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))


def main(outdir, initial, doses, params, resdir=None, order="132"):
    import seir7_ode as so
    tag = so.run_tag(so.input_path(initial), so.input_path(doses), params, order)
    path, path_rho = so.output_paths("optimal", tag, resdir)
    os.makedirs(outdir, exist_ok=True)
    print("reading", path, "and", path_rho, flush=True)
    w = pd.read_csv(path)
    r = pd.read_csv(path_rho)

    for name, f in PANELS.items():
        fig, a = plt.subplots(figsize=(5, 3.25), constrained_layout=True)
        f(a, w, r)
        dates(a)
        out = os.path.join(outdir, f"fig_weekly_optimal_{name}.png")
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print("wrote", out, flush=True)

    path_age = so.age_path("optimal", tag, resdir)
    if os.path.exists(path_age):
        compartments_by_age(pd.read_csv(path_age), outdir)
    else:
        print(f"no {path_age} (a run before the compartments by age were saved); "
              "figures by age skipped", flush=True)


def compartments_by_age(wa, outdir):
    """fig_weekly_optimal_compartments_age<jj>.png, jj = 01, 02, ...: S, P, V, R of the age
    group jj (summed over the locations), in thousands; one figure per age group."""
    names = [("S", "susceptible S"), ("P", "partially immune P"), ("V", "vaccinated V"), ("R", "recovered R")]
    for j, age in enumerate(dict.fromkeys(wa.age), start=1):
        g = wa[wa.age == age]
        fig, a = plt.subplots(figsize=(5, 3.25), constrained_layout=True)
        for k, (c, nm) in enumerate(names):
            a.plot(day(g.t), g[c] / 1e3, color=C[k], label=nm)
        a.set_title(f"Compartments, age {age}, optimal (thousands)", loc="left")
        a.legend(fontsize=7)
        dates(a)
        out = os.path.join(outdir, f"fig_weekly_optimal_compartments_age{j:02d}.png")
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print("wrote", out, flush=True)


if __name__ == "__main__":
    import seir7_ode as so
    opts = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    bad = set(opts) - {"initial", "doses", "params", "out", "order"}
    if bad:
        raise ValueError(f"unknown options {sorted(bad)}: initial=, doses=, params=, out=, order=")
    pos = [a for a in sys.argv[1:] if "=" not in a]
    resdir = os.path.abspath(opts["out"]) if "out" in opts else None
    if not pos and resdir is None:
        raise SystemExit("plot_weekly_optimal.py: give the figure folder FIGDIR or out=DIR")
    main(pos[0] if pos else resdir, opts.get("initial", so.INITIAL),
         opts.get("doses", so.DOSES), opts.get("params", so.DEFAULT_PARAMS), resdir,
         opts.get("order", "132"))
