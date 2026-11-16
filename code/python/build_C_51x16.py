"""The contact matrix C = (c_{ij,pq}) of the 51 x 16 model (notes/ODE.tex, (2.2)), formed and
saved to data/model/C_816x816.csv.gz (gzipped CSV; row (i,j) and column (p,q) labelled "state age",
locations in FIPS order, ages 0-4, ..., 75+), from the public inputs of notes/DATA.tex only:

    c_{ij,pq} = L_p U_jq exp(-b_q d_ip) / sum_{p',q'} L_p' U_jq' exp(-b_q' d_ip'),

U from prem_U_16x16.py, L and d from geo_51.py, b_q from U and d (seir7_ode.decay_rates).
Checks: every row adds up to 1, and C x agrees with its computation without forming C
(appendix of notes/stratified_seir_ngm.tex), for random x.

Usage:  python3 -u build_C_51x16.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(DATA, "model", "C_816x816.csv.gz")


def build():
    from seir7_ode import decay_rates
    pop = pd.read_csv(os.path.join(DATA, "census", "N_51x16.csv"))
    locs = list(dict.fromkeys(pop.state))
    U = pd.read_csv(os.path.join(DATA, "prem2017", "U_16x16.csv"), index_col=0)
    ages = list(U.index)
    U = U.to_numpy(float)
    d = pd.read_csv(os.path.join(DATA, "census", "d_51x51.csv"), index_col=0).to_numpy(float)
    L = pd.read_csv(os.path.join(DATA, "census", "L_51.csv")).L.to_numpy(float)
    m, n = len(locs), len(ages)
    b = decay_rates(U, d)
    E = np.exp(-b[None, None, :] * d[:, :, None])                     # E[i, p, q]
    K = L[None, None, :, None] * U[None, :, None, :] * E[:, None, :, :]  # K[i, j, p, q]
    C = (K / K.sum(axis=(2, 3), keepdims=True)).reshape(m * n, m * n)
    # checks: row sums, and (W U^T)/Z without forming C (appendix of stratified_seir_ngm.tex)
    Kq = np.exp(-b[:, None, None] * d[None, :, :])                    # K^(q)[i, p]
    cy = lambda y: np.einsum("qip,pq->iq", Kq, L[:, None] * y) @ U.T
    Z = cy(np.ones((m, n)))
    x = np.random.default_rng(0).random((m, n))
    err = np.abs(C @ x.ravel() - (cy(x) / Z).ravel()).max()
    print(f"C: {m * n} x {m * n}, entries in [{C.min():.3g}, {C.max():.3g}]; "
          f"max |row sum - 1| = {np.abs(C.sum(axis=1) - 1).max():.1e}; "
          f"max |C x - factored| = {err:.1e}")
    labels = [f"{s} {a}" for s in locs for a in ages]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pd.DataFrame(C, index=labels, columns=labels).to_csv(OUT, index_label="state age",
                                                         float_format="%.17g")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
