"""Three doses: projected Yogi, Algorithm T5.1 of math_bio_three_doses.tex.

The instance, the objective function log rho(A1 diag(g)), its gradient (T1.2),
the 49 starts and the KKT check are those of gs_three_dose.py, imported from it.

Projected Yogi in all 3 N variables u-hat:
    q  = grad F(u)
    mu = beta1 mu + (1 - beta1) q
    s  = s - (1 - beta2) sign(s - q^2) q^2          (s starts at q0^2)
    D  = diag(sqrt(s) + eps)
    u  = Pi^D_S(u - eta D^{-1} mu / (1 - beta1^t))
with Pi^D_S the projection onto the product of simplices in the metric D,
computed exactly per factor (Proposition T5.1(b)).  Stops when the
stationarity measure r(u) = ||u - Pi_S(u - grad F(u))||_inf (T5.4) is below
TOL, or after TMAX iterations.

Writes the log to stdout and the per-start results to yogi_three_dose_results.txt,
or to the file given by --out.  With --only nothing is written unless --out is given.

Usage:
    python3 -u yogi_three_dose.py [--eta E] [--beta1 B1] [--beta2 B2]
                                  [--eps EPS] [--tol T] [--tmax N] [--only NAME]
                                  [--out FILE]
"""

import argparse
import time
import numpy as np

import gs_three_dose as T

OUT = "yogi_three_dose_results.txt"
m, n, N = T.m, T.n, T.N


def proj(z, d):
    """Pi^d of (T5.2) for each row of z, with weights d > 0; rows are simplices."""
    t = d * z
    order = np.argsort(-t, axis=1)
    ts = np.take_along_axis(t, order, axis=1)
    zs = np.take_along_axis(z, order, axis=1)
    ids = np.take_along_axis(1.0 / d, order, axis=1)
    lam = (np.cumsum(zs, axis=1) - 1.0) / np.cumsum(ids, axis=1)
    k = (ts > lam).sum(axis=1) - 1          # the valid candidates form a prefix
    lam = lam[np.arange(z.shape[0]), k]
    return np.maximum(z - lam[:, None] / d, 0.0)


def stationarity(x, gr):
    """r of (T5.4): Euclidean projected gradient step, infinity norm."""
    xr = x.reshape(3 * m, n)
    return np.abs(xr - proj(xr - gr.reshape(3 * m, n), np.ones_like(xr))).max()


def yogi(x0, eta, beta1, beta2, eps, tol, tmax, hist=None):
    """hist, if a list, receives (t, rho, r) at the start and after every iteration."""
    x = np.array(x0, dtype=float)
    f, q = T.value_grad(x)
    mu, s = np.zeros_like(q), q * q
    if hist is not None:
        hist.append((0, np.exp(f), stationarity(x, q)))
    for t in range(1, tmax + 1):
        mu = beta1 * mu + (1.0 - beta1) * q
        s = s - (1.0 - beta2) * np.sign(s - q * q) * q * q
        d = np.sqrt(s) + eps
        z = x - eta * mu / (1.0 - beta1 ** t) / d
        x = proj(z.reshape(3 * m, n), d.reshape(3 * m, n)).ravel()
        f, q = T.value_grad(x)              # the gradient of the next iteration
        r = stationarity(x, q)
        if hist is not None:
            hist.append((t, np.exp(f), r))
        if r < tol:
            break
    return x, t, r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eta", type=float, default=1e-2)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--tmax", type=int, default=200000)
    ap.add_argument("--only", default=None, help="run one start, e.g. barycenter or 'k=3 i=1 j=4'")
    ap.add_argument("--out", default=None, help="results file, default " + OUT)
    a = ap.parse_args()
    print("eta=%g beta1=%g beta2=%g eps=%g tol=%g tmax=%d"
          % (a.eta, a.beta1, a.beta2, a.eps, a.tol, a.tmax), flush=True)

    rows = []
    t_all = time.time()
    for name, x0 in T.starts():
        if a.only is not None and name != a.only:
            continue
        t0 = time.time()
        x, it, r = yogi(x0, a.eta, a.beta1, a.beta2, a.eps, a.tol, a.tmax)
        el = time.time() - t0
        rho = np.exp(T.value_grad(x)[0])
        sp, ex = T.kkt(x)
        rows.append((name, rho, it, el, r, sp, ex))
        print("%-12s rho=%.12f it=%6d %6.1fs r=%.1e kkt %.1e %+.1e" % rows[-1], flush=True)
    print("total %.1f s" % (time.time() - t_all))
    out = a.out if a.out is not None else (OUT if a.only is None else None)
    if out is not None:
        with open(out, "w") as fh:
            fh.write("# eta=%g beta1=%g beta2=%g eps=%g tol=%g tmax=%d\n"
                     % (a.eta, a.beta1, a.beta2, a.eps, a.tol, a.tmax))
            fh.write("# start rho it seconds r spread excess\n")
            for rw in rows:
                fh.write("%s %.12f %d %.2f %.3e %.3e %.3e\n" % ((rw[0].replace(" ", "_"),) + rw[1:]))
        print("written", out)
    vals = np.array([rw[1] for rw in rows])
    print("starts %d, min %.12f max %.12f, distinct to 1e-8: %s"
          % (len(vals), vals.min(), vals.max(), np.unique(np.round(vals, 8))))
