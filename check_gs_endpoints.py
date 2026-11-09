"""Two Gauss-Seidel endpoints of gs_three_dose.py with rho above the joint minimum.

For the starts "k=3 i=1 j=4" and "k=3 i=2 j=1": the endpoint, the joint method
started from it (moved 1e-6 inward), 2000 random feasible directions at steps
1e-5 ... 1e-2 (how many decrease log rho), and the convexity inequality at the
midpoint of the segment from the endpoint to the joint minimizer.

Usage:
    python3 -u check_gs_endpoints.py
"""

import numpy as np

import gs_three_dose as T

NAMES = ["k=3 i=1 j=4", "k=3 i=2 j=1"]

if __name__ == "__main__":
    st = dict(T.starts())
    rng = np.random.default_rng(1)
    xj, _ = T.joint(st["barycenter"])
    fj = T.value_grad(xj)[0]
    print("joint minimizer from the barycenter: rho %.10f" % np.exp(fj))
    for name in NAMES:
        xg, sw = T.gauss_seidel(st[name])
        f0 = T.value_grad(xg)[0]
        print("== %s: GS rho %.10f after %d sweeps" % (name, np.exp(f0), sw))
        xr, _ = T.joint((1 - 1e-6) * xg + 1e-6 / T.n)
        print("   joint method from the endpoint moved 1e-6 inward: rho %.10f"
              % np.exp(T.value_grad(xr)[0]))
        for r in (1e-5, 1e-4, 1e-3, 1e-2):
            dec = 0
            for _ in range(2000):
                y = rng.dirichlet(np.ones(T.n), size=3 * T.m).ravel()
                dec += T.value_grad((1 - r) * xg + r * y)[0] - f0 < -1e-13
            print("   step %.0e: %d of 2000 random feasible directions decrease log rho" % (r, dec))
        mid = T.value_grad(0.5 * (xg + xj))[0]
        print("   midpoint of the segment to the joint minimizer: log rho %.6f, chord %.6f, "
              "difference %+.2e" % (mid, 0.5 * (f0 + fj), mid - 0.5 * (f0 + fj)))
