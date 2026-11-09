"""Figures of notes/min_spectral_radius_vaccination.tex, as pdf and png.

Reads the .dat files in notes/suplemental/, which plot_chi1_single_dose.py and
plot_yogi_three_dose.py write, and writes to notes/figures/:

  fig_chi1        Figure 3.1, the minimizing chi1 and the chi1 of John's method;
  fig_yogi_conv   Figure 4.1, projected Yogi on the weekly instance;
  fig_alloc_s2    Figure 4.2, chi_k at the endpoints, sigma2, sigma3 of seed 2;
  fig_alloc_p     Figure 4.3, the same with sigma2 = sigma3 = P.

The width is the \\textwidth of the document, 6.5 in, so the figures are
included at their natural size.

Usage:
    python3 -u make_figures.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAT = os.path.join(ROOT, "notes", "suplemental")
OUT = os.path.join(ROOT, "notes", "figures")

WIDTH = 6.5
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
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(os.path.join(OUT, "%s.%s" % (stem, ext)), **kw)
    plt.close(fig)
    print("written", os.path.join(OUT, stem + ".{pdf,png}"))


def age_axis(ax, labels=True):
    ax.set_xticks(range(4))
    ax.set_xticklabels(AGES if labels else [])
    ax.set_ylim(bottom=0)


def fig_chi1():
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 2.7), sharey=True)
    for ax, name, title in zip(axs, ["fig_chi1_single_dose.dat", "fig_chi1_john.dat"],
                               ["minimizer\n" r"$\rho=2.9059$", "one dose from eigenvalues_H_ZF_JG.nb\n" r"$\rho=2.9312$"]):
        d = read(name)
        for _, full, col, mk in LOC:
            ax.plot(d["age"], d[full], color=col, marker=mk, lw=1.4, ms=4.5, label=full)
        age_axis(ax)
        ax.set_ylim(0, 5.6)
        ax.set_title(title)
        ax.set_xlabel("age group (years)")
    axs[0].set_ylabel(r"$\chi_1$ ($10^{-3}$ per day)")
    axs[0].legend(loc="lower left")
    save(fig, "fig_chi1")


def fig_yogi_conv():
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 2.5))
    runs = [("bary", "barycenter", LOC[0][2]), ("k3", r"$(3,1,4)$", LOC[1][2]),
            ("k2", r"$(2,4,1)$", LOC[2][2])]
    for key, lab, col in runs:
        d = read("fig_yogi_hist_%s.dat" % key)
        keep = d["it"] > 0                      # iteration 0 is not shown on the log scale
        axs[0].loglog(d["it"][keep], d["excess"][keep], color=col, lw=1.0, label=lab)
        axs[1].loglog(d["it"][keep], d["r"][keep], color=col, lw=1.0)
    axs[0].set_ylim(1e-16, 2)
    axs[1].set_ylim(1e-16, 2)
    axs[0].set_title("relative excess")
    axs[1].set_title("stationarity measure")
    axs[0].set_ylabel(r"$(\rho-\rho^*)/\rho^*$")
    axs[1].set_ylabel(r"$r$")
    for ax in axs:
        ax.set_xlabel("iteration")
    axs[0].legend(loc="lower left")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig_yogi_conv")


def fig_alloc(stem, rho_ip, rho_yogi):
    fig, axs = plt.subplots(3, 2, figsize=(WIDTH, 6.0), sharex=True)
    ymax = [6, 60, 100]
    for c, (kind, title) in enumerate([("joint", "interior point method, $\\rho=%s$" % rho_ip),
                                       ("yogi", "projected Yogi, $\\rho=%s$" % rho_yogi)]):
        d = read("fig_alloc_%s_%s.dat" % (stem, kind))
        for k in range(3):
            ax = axs[k, c]
            for short, full, col, mk in LOC:
                ax.plot(d["age"], d["k%d%s" % (k + 1, short)], color=col, marker=mk,
                        lw=1.0, ms=3.5, label=full)
            age_axis(ax, labels=(k == 2))
            ax.set_ylim(0, ymax[k])
            if c == 0:
                ax.set_ylabel(r"$\chi_%d$" % (k + 1))
            if k == 2:
                ax.set_xlabel("age group (years)")
        axs[0, c].set_title(title)
    axs[0, 0].legend(loc="lower left")
    fig.tight_layout(h_pad=1.2, w_pad=2.0)
    save(fig, "fig_alloc_%s" % stem)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_chi1()
    fig_yogi_conv()
    fig_alloc("s2", "1.7118", "1.7095")
    fig_alloc("p", "1.3088", "1.3085")
