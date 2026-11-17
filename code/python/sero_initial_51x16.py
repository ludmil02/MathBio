"""MOCK initial state at t = 251 (7 September 2020) of the 51 x 16 model, from the CDC
"Nationwide Commercial Laboratory Seroprevalence Survey" (data.cdc.gov, id d2tw-32xv;
Bajema et al. 2021), rounds 3 (26 Aug - 9 Sep 2020) and 4 (9 - 18 Sep 2020).

The survey gives the percentage seropositive by state and 4 age groups (0-17, 18-49, 50-64,
65+). The model needs 16 age groups and the infected compartments; both steps below are
mock rules (notes/DATA.tex, Section 1.9.1), to be replaced by the procedure of Glasser et al.
2023, Section 3:

1. 4 -> 16 age groups: each 5-year group takes the value of the survey group that contains
   it; 15-19 takes 3/5 of 0-17 and 2/5 of 18-49 (by years of age). A value that is missing
   or suppressed (code 777) is replaced by the national value (site "US") of the same round
   and age group.
2. Seropositive fraction r_ij at t = 251 from round 3; force of infection
   lambda_ij = max(-ln((1 - r4)/(1 - r3)) / 11.5, lambda_floor), 11.5 days between the
   midpoints of rounds 3 and 4, lambda_floor = 1e-4 per day (the 4-round differences are
   noisy and often negative).
3. R = r N, and the infected compartments in quasi-steady state with this lambda (the
   continuous form of the 12-day sums of Glasser et al. 2023, Section 3):
   E_s = lambda S / k, I_a = k p E_s / gamma_a, I_ps = k (1 - p) E_s / xi,
   I_s = xi I_ps / (gamma_s + delta_s), S = N - R - E_s - I_a - I_ps - I_s (solved for S),
   and V = P = E_p = I_pm = I_m = I_h = M = 0.
->  data/cdc/state_t251_mock.csv (columns state, age, S, P, Es, Ep, V, Ia, Ips, Ipm, Is, Im, Ih, R, M)

Usage:  python3 -u sero_initial_51x16.py
"""
import os
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.normpath(os.path.join(HERE, "..", "..", "data", "Downloaded_from_source", "cdc", "seroprevalence.csv.gz"))
URL = "https://data.cdc.gov/resource/d2tw-32xv.csv?$limit=100000"
OUT = os.path.join(HERE, "data", "cdc", "state_t251_mock.csv")
GROUPS = ["0_17", "18_49", "50_64", "65"]
# weights of the 4 survey groups for each of the 16 model groups
W = np.zeros((16, 4))
W[0:3, 0] = 1                       # 0-4, 5-9, 10-14
W[3] = [0.6, 0.4, 0, 0]             # 15-19
W[4:10, 1] = 1                      # 20-24, ..., 45-49
W[10:13, 2] = 1                     # 50-54, 55-59, 60-64
W[13:16, 3] = 1                     # 65-69, 70-74, 75+
DT = 11.5
LAMBDA_FLOOR = 1e-4


def fractions(s, rnd, locs):
    x = s[s["round"] == rnd].set_index("site")
    R = np.empty((len(locs), 4))
    for g, grp in enumerate(GROUPS):
        col = pd.to_numeric(x[f"rate_{grp}_prevalence"], errors="coerce")
        us = col.get("US")
        for i, st in enumerate(locs):
            v = col.get(st, np.nan)
            R[i, g] = (v if (v == v and 0 <= v <= 100) else us) / 100
    return R @ W.T                      # 51 x 16


def build():
    import seir7_ode as O
    mod = O.Model(initial="none")
    if not os.path.exists(RAW):
        from census_N_51x16 import fetch_gz
        fetch_gz(URL, RAW)
    s = pd.read_csv(RAW)
    r3, r4 = fractions(s, 3, mod.locs), fractions(s, 4, mod.locs)
    lam = np.maximum(-np.log((1 - r4) / (1 - r3)) / DT, LAMBDA_FLOOR)
    N = mod.N
    R = r3 * N
    # E_s = lam S / k etc.; S + E + I = N - R with all of them proportional to S
    es = lam / mod.k
    ia = mod.k * mod.p * es / mod.gamma_a
    ips = mod.k * (1 - mod.p) * es / mod.xi
    is_ = mod.xi * ips / (mod.gamma_s + mod.delta_s)
    S = (N - R) / (1 + es + ia + ips + is_)
    X = np.zeros((13, mod.m, mod.n))
    X[O.S], X[O.ES], X[O.IA], X[O.IPS], X[O.IS], X[O.R] = S, es * S, ia * S, ips * S, is_ * S, R
    df = pd.DataFrame({"state": np.repeat(mod.locs, mod.n), "age": np.tile(mod.ages, mod.m)})
    for c, nm in enumerate(O.NAMES):
        df[nm] = X[c].ravel()
    df.to_csv(OUT, index=False, float_format="%.10g")
    print(f"seropositive at t = 251: {R.sum() / N.sum():.2%} of N; lambda in [{lam.min():.2e}, {lam.max():.2e}]"
          f" (at the floor in {(lam == LAMBDA_FLOOR).sum()} of {lam.size} groups);"
          f" infected (E + I) {X[[O.ES, O.IA, O.IPS, O.IS]].sum():,.0f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
