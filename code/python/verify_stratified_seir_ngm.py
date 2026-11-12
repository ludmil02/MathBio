"""Reproduce the numbers of Section 5 and Table 1 of notes/stratified_seir_ngm.tex.

  Table 1   rho(F V^-1) of the full 8-state system and rho(A1 diag(g)) of
            Theorem 3.1, for the 4 x 4 data (zf_jg_params.py) at chi = 0 and at
            the chi of F_and_V_ZF_18Jan_orig.nb, and for the 51 x 16 data
            (intermediate/seir7_t347/, see ngm_seir7.py) at chi = 0 and, as
            rho(NGM) vs rho(A1 diag(g)), at a random chi;
  caption   the ranges of r_p = q_m / q_s;
  Sect. 5   at a random chi: the closed form (4.1) of g against y + r z, the two
            forms (4.3) of e_p and 0 < e_p <= 1, g_p = (1 - e_p) + e_p / t_p, and
            the derivatives (4.6) against central differences.

Each value is printed next to the one in the document; the script ends with
PASS or FAIL.  All eigenvalues by Arnoldi; F V^-1 is applied matrix free with V
factored by sparse LU.

Usage:
    python3 -u verify_stratified_seir_ngm.py | tee intermediate/verify_stratified_seir_ngm.log
"""

import sys
import time

import numpy as np
import scipy.sparse.linalg as spsl

import zf_jg as Z
from zf_jg_params import default_params
import ngm_seir7 as S7

T0 = time.time()
FAILS = []


def say(*args):
    print("[%7.2fs]" % (time.time() - T0), *args, flush=True)


def check(label, value, expected, tol):
    ok = abs(value - expected) <= tol
    if not ok:
        FAILS.append(label)
    say("%-58s %.12f   document %.12f   %s" % (label, value, expected, "ok" if ok else "FAIL"))


def check_small(label, value, bound):
    ok = value <= bound
    if not ok:
        FAILS.append(label)
    say("%-58s %.2e   (bound %.0e)   %s" % (label, value, bound, "ok" if ok else "FAIL"))


def rho_of(mat):
    return float(abs(spsl.eigs(mat, k=1, which="LM", tol=1e-14, return_eigenvectors=False)[0]))


TOL = 1e-11      # the document gives 12 decimals

# ------------------------------------------------------------------ 4 x 4 data
say("4 x 4 data (zf_jg_params.py)")
par = default_params()
m4, n4 = par["m"], par["n"]
qs4, qm4 = Z.q_vectors(par)
prm4 = dict(k=0.2, xi=par["xi"], vp=par["vp"], vq=par["vq"], ga=par["vgamma_a"],
            gs=par["vgamma_s"], gm=par["vgamma_m"], gh=par["vgamma_h"],
            ds=par["vdelta_s"], dm=par["vdelta_m"],
            eta=(par["eta_a"], par["eta_ps"], par["eta_s"], par["eta_h"], par["eta_pm"], par["eta_m"]))
C4 = par["mc"].reshape(m4 * n4, m4 * n4)
psi4 = par["NPII"].ravel()

# chi of F_and_V_ZF_18Jan_orig.nb, cell 19 (chi3 = 0)
chi1_fv = np.array([[0.25, 0.5, 0.16, 0.12], [0.14, 0.27, 0.095, 0.07],
                    [0.17, 0.35, 0.12, 0.09], [0.16, 0.31, 0.11, 0.08]])
chi2_fv = np.array([[0.002, 0.007, 0.0006, 0.0003], [0.003, 0.012, 0.001, 0.0007],
                    [0.006, 0.024, 0.002, 0.001], [0.015, 0.054, 0.005, 0.003]])
cases4 = [("chi = 0", Z.zero_chi(par), 4.440677536267),
          ("chi of F_and_V nb", (chi1_fv, chi2_fv, np.zeros((m4, n4))), 1.570568657459)]
for label, chi, doc in cases4:
    y, z = Z.compute_yz(*chi, par)
    check("4x4, %s: rho(F V^-1)" % label, S7.full_fv(y, z, C4, psi4, n4, prm4, verbose=False), doc, TOL)
    check("4x4, %s: rho(A1 diag(g))" % label, rho_of(Z.ngm_reduced(*chi, par=par)), doc, TOL)

# ------------------------------------------------------------------ 51 x 16 data
m, n = S7.m, S7.n
say("51 x 16 data (%s)" % S7.DATA)
A1 = S7.C * np.tile(S7.psiI * S7.Qs, m)[None, :]          # diag(d_c) moved into g below
r_age = S7.Qm / S7.Qs


def reduced_rho(chi):
    """rho(A1 diag(g)) with A1 = diag(d_c) C diag(psi_I q_s), g = y + r z.

    diag(d_c) A1' diag(g) and A1' diag(g) diag(d_c) have the same spectrum, so
    d_c is carried with g: rho(C diag(psi_I q_s) diag(d_c g)).
    """
    Y, Zz, _ = S7.yz(*chi)
    dg = (Y + np.tile(r_age, (m, 1)) * Zz).ravel()          # = d_c * g
    return rho_of(A1 * dg[None, :])


zero = np.zeros((m, n))
Y0, Z0, _ = S7.yz(zero, zero, zero)
check("51x16, chi = 0: rho(F V^-1)",
      S7.full_fv(Y0, Z0, S7.C, np.tile(S7.psiI, m), n, S7.AGE_PRM, verbose=False), 1.307390501274, TOL)
check("51x16, chi = 0: rho(A1 diag(g))", reduced_rho((zero, zero, zero)), 1.307390501274, TOL)

rng = np.random.default_rng(1)
chi_r = [rng.random((m, n)) * s for s in (0.05, 0.01, 0.01)]
Yr, Zr, _ = S7.yz(*chi_r)
check("51x16, random chi: rho(NGM)", S7.perron(S7.h_operator(Yr, Zr, S7.Qs, S7.Qm)[0])[0],
      0.492928980848, TOL)
check("51x16, random chi: rho(A1 diag(g))", reduced_rho(chi_r), 0.492928980848, TOL)

# ------------------------------------------------------------------ ratios r_p
r4 = qm4 / qs4
say("r_p, 4 x 4:   [%.4f, %.4f]   document [0.732, 0.977]" % (r4.min(), r4.max()))
say("r_p, 51 x 16: [%.4f, %.4f]   document [0.714, 0.985]" % (r_age.min(), r_age.max()))
for label, lo, hi, dlo, dhi in (("4x4", r4.min(), r4.max(), 0.732, 0.977),
                                ("51x16", r_age.min(), r_age.max(), 0.714, 0.985)):
    if round(lo, 3) != dlo or round(hi, 3) != dhi or not (0 < lo <= hi <= 1):
        FAILS.append("r_p range " + label)
        say("r_p range %s: FAIL" % label)

# ------------------------------------------------------------------ Section 5 formulas
say("Section 5 formulas at a random chi (51 x 16 data)")
om, omv, e1, e2, e3 = S7.om, S7.omv, S7.e1, S7.e2, S7.e3
r = np.tile(r_age, (m, 1))
rng = np.random.default_rng(3)
chi = [rng.random((m, n)) * s for s in (0.05, 0.01, 0.01)]


def gclosed(c1, c2, c3):
    p1 = c1 / (omv + e2 * c2)
    p2 = e2 * c2 / om
    p3 = 1 + e3 * c3 / om
    return (1 + r * p1 * (1 - e1 + p2)) / (1 + p1 * (1 + p2 + p2 * p3))


Y, Zz, pref = S7.yz(*chi)
g = gclosed(*chi)
check_small("(4.1) vs (y + r z): max abs diff", np.abs(g - (Y + r * Zz) / pref).max(), 1e-14)
if not ((g > 0).all() and (g <= 1).all()):
    FAILS.append("0 < g <= 1")
say("0 < g <= 1: %s, range [%.4f, %.4f]" % (bool((g > 0).all() and (g <= 1).all()), g.min(), g.max()))

c1, th, ze = chi
omp = omv + e2 * th
alpha = 1 + 2 * e2 * th / om + e2 * e3 * th * ze / om**2
pi2 = e2 * th / om
e = 1 - r * (1 - e1 + pi2) / alpha
e_alt = ((1 - r) + r * e1 + (2 - r) * pi2 + pi2 * e3 * ze / om) / alpha
t = 1 + alpha / omp * c1
check_small("(4.3): the two forms of e_p, max abs diff", np.abs(e - e_alt).max(), 1e-14)
if not ((e > 0).all() and (e <= 1).all()):
    FAILS.append("0 < e <= 1")
say("0 < e_p <= 1: %s" % bool((e > 0).all() and (e <= 1).all()))
check_small("g_p = (1 - e_p) + e_p / t_p, max abs diff", np.abs(g - ((1 - e) + e / t)).max(), 1e-14)

p1 = c1 / omp
p3 = 1 + e3 * ze / om
Nn = 1 + r * p1 * (1 - e1 + pi2)
D = 1 + p1 * (1 + pi2 + pi2 * p3)
d1 = -alpha * e / (omp * D**2)
d3 = -Nn * p1 * pi2 * e3 / (om * D**2)
# the closed forms of (4.6)
dN = r * e2 * p1 * (1 / om - (1 - e1 + pi2) / omp)
dD = e2 * p1 * ((1 + p3) / om - alpha / omp)
d2 = (dN * D - Nn * dD) / D**2
h = 1e-7
for kk, d in ((0, d1), (1, d2), (2, d3)):
    cp = [x.copy() for x in chi]
    cm = [x.copy() for x in chi]
    cp[kk] += h
    cm[kk] -= h
    fd = (gclosed(*cp) - gclosed(*cm)) / (2 * h)
    check_small("(4.6) dg/dchi%d vs central difference, max rel diff" % (kk + 1),
                np.abs(fd - d).max() / np.abs(d).max(), 2e-9)

say("PASS" if not FAILS else "FAIL: " + ", ".join(FAILS))
sys.exit(1 if FAILS else 0)
