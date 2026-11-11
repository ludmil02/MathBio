"""Figures of notes/min_spectral_radius_vaccination.tex, one plot per png.

Reads the .dat files in notes/figures/dat/, which plot_chi1_single_dose.py and
plot_yogi_three_dose.py write, and writes to notes/figures/:

  fig_chi1_{min,john}                Figure 3.1, the minimizing chi1 and the
                                     chi1 of John's method;
  fig_yogi_conv_{excess,r}           Figure 4.1, projected Yogi on the weekly
                                     instance: relative excess and r;
  fig_alloc_s2_{joint,yogi}_k{1,2,3} Figures 4.2-4.4, chi_k at the endpoints of
                                     the interior point method and of Yogi,
                                     sigma2, sigma3 of seed 2;
  fig_alloc_p_{joint,yogi}_k{1,2,3}  Figures 4.5-4.7, the same with
                                     sigma2 = sigma3 = P.

Each plot is WIDTH = 2.6 in wide, 0.4 of the \\textwidth of the document,
6.5 in, and is included at that width, so at its natural size.  Plots shown
side by side share their y scale.

Usage:
    python3 -u make_figures.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DAT = os.path.join(ROOT, "notes", "figures", "dat")
OUT = os.path.join(ROOT, "notes", "figures")

WIDTH = 2.6                     # in; 0.4 of the 6.5 in \textwidth
LOC = [("NE", "Northeast", "#2A78D6", "o"), ("MW", "Midwest", "#EB6834", "s"),
       ("S", "South", "#1BAF7A", "^"), ("W", "West", "#EDA100", "D")]
AGES = ["0–19", "20–39", "40–59", "60+"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "cm",
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "0.88", "grid.linewidth": 0.6,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def read(name):
    path = os.path.join(DAT, name)
    with open(path) as fh:
        cols = fh.readline().split()
    data = np.loadtxt(path, skiprows=1, ndmin=2)
    return {c: data[:, k] for k, c in enumerate(cols)}


def save(fig, stem):
    path = os.path.join(OUT, stem + ".png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print("written", path)


def age_axis(ax, labels=True):
    ax.set_xticks(range(4))
    ax.set_xticklabels(AGES if labels else [])
    ax.set_ylim(bottom=0)


def fig_chi1():
    for key, name, legend in [("min", "fig_chi1_single_dose.dat", True),
                              ("john", "fig_chi1_john.dat", False)]:
        fig, ax = plt.subplots(figsize=(WIDTH, 2.1))
        d = read(name)
        for _, full, col, mk in LOC:
            ax.plot(d["age"], d[full], color=col, marker=mk, lw=1.4, ms=4.5, label=full)
        age_axis(ax)
        ax.set_ylim(0, 5.6)
        ax.set_xlabel("age group (years)")
        ax.set_ylabel(r"$\chi_1$ ($10^{-3}$ per day)")
        if legend:
            ax.legend(loc="lower left")
        save(fig, "fig_chi1_%s" % key)


def fig_yogi_conv():
    runs = [("bary", "barycenter", LOC[0][2]), ("k3", r"$(3,1,4)$", LOC[1][2]),
            ("k2", r"$(2,4,1)$", LOC[2][2])]
    hist = {key: read("fig_yogi_hist_%s.dat" % key) for key, _, _ in runs}
    for col_name, ylabel, legend in [("excess", r"$(\rho-\rho^*)/\rho^*$", True),
                                     ("r", r"$r$", False)]:
        fig, ax = plt.subplots(figsize=(WIDTH, 2.1))
        for key, lab, col in runs:
            d = hist[key]
            keep = d["it"] > 0                  # iteration 0 is not shown on the log scale
            ax.loglog(d["it"][keep], d[col_name][keep], color=col, lw=1.0, label=lab)
        ax.set_ylim(1e-16, 2)
        ax.set_xlabel("iteration")
        ax.set_ylabel(ylabel)
        if legend:
            ax.legend(loc="lower left")
        save(fig, "fig_yogi_conv_%s" % col_name)


def fig_alloc(stem):
    ymax = [6, 60, 100]                 # the same in the two plots of a row
    for kind in ("joint", "yogi"):
        d = read("fig_alloc_%s_%s.dat" % (stem, kind))
        for k in range(3):
            fig, ax = plt.subplots(figsize=(WIDTH, 1.8))
            for short, full, col, mk in LOC:
                ax.plot(d["age"], d["k%d%s" % (k + 1, short)], color=col, marker=mk,
                        lw=1.0, ms=3.5, label=full)
            age_axis(ax)
            ax.set_ylim(0, ymax[k])
            ax.set_xlabel("age group (years)")
            ax.set_ylabel(r"$\chi_%d$ ($10^{-3}$ per day)" % (k + 1))
            if kind == "joint" and k == 0:
                ax.legend(loc="lower left")
            save(fig, "fig_alloc_%s_%s_k%d" % (stem, kind, k + 1))


def fig_tralt_conv():
    """Section 4.3: the trust region phase (excess per iteration) and the
    alternating phase (excess and r per sweep), weekly b2 = b1/8, b3 = b1/2, P."""
    runs = [("bary", "barycenter", LOC[0][2]), ("k1", r"$(1,1,4)$", LOC[1][2]),
            ("k2", r"$(2,4,4)$", LOC[2][2])]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.1))
    for key, lab, col in runs:
        d = read("fig_tralt_tr_%s.dat" % key)
        ax.semilogy(d["it"], d["excess"], color=col, lw=1.0, label=lab)
    ax.set_ylim(1e-16, 2)
    ax.set_xlabel("trust region iteration")
    ax.set_ylabel(r"$(\rho-\rho_{\mathrm{ref}})/\rho_{\mathrm{ref}}$")
    ax.legend(loc="lower left")
    save(fig, "fig_tralt_tr")
    fig, ax = plt.subplots(figsize=(WIDTH, 2.1))
    for key, lab, col in runs[1:]:
        d = read("fig_tralt_alt_%s.dat" % key)
        ax.semilogy(d["sweep"], d["excess"], color=col, lw=1.0, marker="o", ms=3, label=lab + r", excess")
        ax.semilogy(d["sweep"], d["r"], color=col, lw=1.0, ls="--", marker="s", ms=3, label=lab + r", $r$")
    ax.set_ylim(1e-16, 2)
    ax.set_xlabel("sweep")
    ax.legend(loc="upper right")
    save(fig, "fig_tralt_alt")


def fig_alloc_tralt():
    ymax = [6, 60, 100]
    for stem in ("q", "p"):
        d = read("fig_alloc_tralt_%s.dat" % stem)
        for k in range(3):
            fig, ax = plt.subplots(figsize=(WIDTH, 1.8))
            for short, full, col, mk in LOC:
                ax.plot(d["age"], d["k%d%s" % (k + 1, short)], color=col, marker=mk,
                        lw=1.0, ms=3.5, label=full)
            age_axis(ax)
            ax.set_ylim(0, ymax[k])
            ax.set_xlabel("age group (years)")
            ax.set_ylabel(r"$\chi_%d$ ($10^{-3}$ per day)" % (k + 1))
            if stem == "q" and k == 0:
                ax.legend(loc="lower left")
            save(fig, "fig_alloc_tralt_%s_k%d" % (stem, k + 1))


if __name__ == "__main__":
    import sys
    os.makedirs(OUT, exist_ok=True)
    if "--tralt" in sys.argv:          # only the figures of Section 4.3
        fig_tralt_conv()
        fig_alloc_tralt()
        sys.exit(0)
    fig_chi1()
    fig_yogi_conv()
    fig_alloc("s2")
    fig_alloc("p")
    fig_tralt_conv()
    fig_alloc_tralt()
