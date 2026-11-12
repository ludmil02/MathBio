"""Figures of notes/stratified_seir_ngm.tex, 51 x 16 data at t = 347 (chi = 0).

  fig_weights.pdf    q_s, q_m by age (left) and r = q_m / q_s (right);
  fig_left_ev.pdf    S part of the dominant left eigenvector of the NGM by age:
                     share of the total (left), per person relative to the
                     largest age group (right);
  fig_grad_hhs.pdf   -d rho(NGM) / d chi_1 by age for the ten HHS region
                     headquarter states.

Colors: slots 1 and 2 of the reference categorical palette (blue, orange),
neutral gray for context lines; ink and grid in text and muted tones.

Usage:
    python3 -u figures_stratified_seir_ngm.py | tee intermediate/figures_stratified_seir_ngm.log
"""

import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ngm_seir7 as S7

T0 = time.time()
OUT = os.path.join(os.path.dirname(S7.HERE), "..", "notes", "figures")
OUT = os.path.normpath(OUT)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY = "#b5b3ad"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e4e3de"

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.5, "lines.markersize": 4,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})

AGE_TICKS = S7.AGES


def say(*args):
    print("[%7.2fs]" % (time.time() - T0), *args, flush=True)


def age_axis(ax):
    x = np.arange(S7.n)
    ax.set_xticks(x)
    ax.set_xticklabels(AGE_TICKS, rotation=60, ha="right", fontsize=7)
    ax.set_xlabel("age group (years)")
    ax.grid(axis="x", visible=False)
    return x


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    say("wrote", path)


def fig_weights():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 2.6), gridspec_kw=dict(wspace=0.35))
    x = age_axis(a1)
    a1.plot(x, S7.Qs, color=BLUE, marker="o", label=r"$q_s$ (naive)")
    a1.plot(x, S7.Qm, color=ORANGE, marker="s", label=r"$q_m$ (partially immune)")
    a1.text(x[-1] + 0.4, S7.Qs[-1], r"$q_s$", color=INK, va="center")
    a1.text(x[-1] + 0.4, S7.Qm[-1], r"$q_m$", color=INK, va="center")
    a1.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2,
              borderaxespad=0.2, handlelength=1.8, columnspacing=1.2)
    a1.set_ylabel("weight (days)")
    a1.set_xlim(-0.5, S7.n + 0.6)
    x = age_axis(a2)
    a2.plot(x, S7.Qm / S7.Qs, color=BLUE, marker="o")
    a2.set_ylabel(r"$r=q_m/q_s$")
    save(fig, "fig_weights.pdf")


def fig_left_ev(wS):
    share = wS.sum(axis=0)
    percap = share / S7.mN.sum(axis=0)
    percap = percap / percap.max()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 2.6), gridspec_kw=dict(wspace=0.35))
    for ax, val, lab in ((a1, share, "share of the total"),
                         (a2, percap, "per person, relative")):
        x = age_axis(ax)
        ax.bar(x, val, width=0.72, color=BLUE, edgecolor="white", linewidth=1.0)
        ax.set_ylabel(lab)
        ax.set_ylim(0, val.max() * 1.15)
        k = int(np.argmax(val))
        ax.text(x[k], val[k] * 1.03, "%.2f" % val[k], ha="center", va="bottom", fontsize=8, color=INK2)
    save(fig, "fig_left_ev.pdf")


def fig_grad_hhs(grad):
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    x = age_axis(ax)
    rows = [(city, st, -grad[S7.STATES.index(st)]) for city, st in S7.HHS]
    peak = [r[2].max() for r in rows]
    hi, lo = int(np.argmax(peak)), int(np.argmin(peak))
    first = True
    for k, (city, st, g) in enumerate(rows):
        if k not in (hi, lo):
            ax.plot(x, g, color=GRAY, linewidth=1.0,
                    label="the other eight" if first else None)
            first = False
    for k, col in ((hi, BLUE), (lo, ORANGE)):
        city, st, g = rows[k]
        ax.plot(x, g, color=col, marker="o", label="%s (%s)" % (city, st))
    h, l = ax.get_legend_handles_labels()
    order = [1, 0, 2]                                  # largest, the others, smallest
    ax.legend([h[i] for i in order], [l[i] for i in order], frameon=False,
              loc="upper right", fontsize=8)
    ax.set_ylabel(r"$-\partial\rho/\partial\chi_1$")
    say("largest peak: %s (%s) %.3f; smallest: %s (%s) %.3f"
        % (rows[hi][0], rows[hi][1], peak[hi], rows[lo][0], rows[lo][1], peak[lo]))
    save(fig, "fig_grad_hhs.pdf")


def fig_doses_4x4():
    """First doses chi1 S by age, per location, for three allocations of Section 6."""
    import compare_allocations_4x4 as CA
    allocs, _ = CA.allocations()
    show = [("actual", "actual", BLUE),
            ("left ev, doses", "left eigenvector, doses", ORANGE),
            ("minimizer", "minimizer", AQUA)]
    x = np.arange(CA.n)
    wbar = 0.26
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.4), sharex=True, sharey=True,
                             gridspec_kw=dict(hspace=0.35, wspace=0.12))
    for i, ax in enumerate(axes.ravel()):
        for k, (key, label, col) in enumerate(show):
            doses = allocs[key][i] * CA.S[i] / 1e3
            ax.bar(x + (k - 1) * wbar, doses, width=wbar, color=col, edgecolor="white",
                   linewidth=0.8, label=label)
        ax.set_title(CA.LOCS[i], fontsize=9, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels(CA.AGES)
        ax.grid(axis="x", visible=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("first doses (thousands)")
    for ax in axes[1, :]:
        ax.set_xlabel("age group (years)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=3, frameon=False)
    for key, _, _ in show:
        say("%-15s doses by location: %s" % (key, np.array2string((allocs[key] * CA.S).sum(axis=1), precision=0)))
    save(fig, "fig_doses_4x4.pdf")


def main():
    os.makedirs(OUT, exist_ok=True)
    zero = np.zeros((S7.m, S7.n))
    Y, Z, pref = S7.yz(zero, zero, zero)
    rho, wS, _, _, grad = S7.left_and_gradient(Y, Z, pref)
    say("rho(NGM) = %.12f at chi = 0" % rho)
    fig_weights()
    fig_left_ev(wS)
    fig_grad_hhs(grad)
    fig_doses_4x4()
    say("done")


if __name__ == "__main__":
    main()
