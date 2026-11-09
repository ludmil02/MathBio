"""The dependence of g_p in (2.4) on the second dose, u_2, with u_1 and u_3 fixed.

Samples (u_1, u_2, u_3) log-uniformly in [1e-5, 10^-1.5]^3 with the parameters
of the model (eps_1 = eps_2 = eps_3 = 0.8, omega = omega_v = 1/182.5) and reports
the smallest u_2^2 d^2 log g / du_2^2 (negative: log g is not convex in u_2) and
the fraction of samples where g increases in u_2 (positive: g is not monotone).

Usage:
    python3 -u check_block2.py
"""

import numpy as np

from zf_jg_params import default_params

SAMPLES = 200000

par = default_params()
e1, e2, e3, om, omv = par["eps1"], par["eps2"], par["eps3"], par["omega"], par["omega_v"]


def g(u1, u2, u3):
    p1, p2, p3 = u1 / (omv + e2 * u2), e2 * u2 / om, 1.0 + e3 * u3 / om
    return (1.0 + (1.0 - e1) * p1 + p1 * p2) / (1.0 + p1 + p1 * p2 + p1 * p2 * p3)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    u1, u2, u3 = 10.0 ** rng.uniform(-5, -1.5, size=(3, SAMPLES))
    h = 1e-4 * u2
    f = lambda x: np.log(g(u1, x, u3))
    d1 = (f(u2 + h) - f(u2 - h)) / (2 * h)
    d2 = (f(u2 + h) - 2 * f(u2) + f(u2 - h)) / h**2
    print("parameters: eps = %.2f %.2f %.2f, omega = %.6f, omega_v = %.6f" % (e1, e2, e3, om, omv))
    print("min u2^2 d2(log g)/du2^2 = %.3f" % (u2**2 * d2).min())
    print("fraction with dg/du2 > 0  = %.3f" % (d1 > 0).mean())
