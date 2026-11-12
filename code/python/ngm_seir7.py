"""Next generation matrix of the 51 x 16 age- and location-stratified SEIR model at chi = 0.

The model is that of Feng, Day, Glasser (J. Theor. Biol. 633 (2026) 112531) and of
notes/stratified_seir_ngm.tex.  Data: intermediate/seir7_t347/, exported from
"Age & Location Stratified SEIR 7 Actual Vax 3 with RE_orig.nb" by
Mathematica_Wolfram/export_seir7.wls at t = 347 (just before vaccination began).

The NGM has size 2 m n = 1632 and is never formed:

    NGM[(i,j,a), (p,q,b)] = Omega[i,j,a] mc[i,j,p,q] psiI[q] Q[q,b],

a, b = 0 (rows Y, columns q_s) and 1 (rows Z, columns q_m).  Products with NGM
and NGM^T cost one product with the 816 x 816 contact matrix mc.

Importing the module loads the data and defines the functions; running it prints
  1. mc formed here, checked against two slices computed in Mathematica;
  2. rho(NGM) by Arnoldi;
  3. rho(F V^-1) from the full 8-state F and V (matrix free, V by sparse LU),
     which must equal rho(NGM);
  4. the dominant left eigenvector, S part, by age and by state;
  5. -d rho / d chi1 at chi = 0, by age, for the ten HHS region headquarter
     states of Fig. 3A of the paper, with a finite difference check.

Usage:
    python3 -u ngm_seir7.py | tee intermediate/ngm_seir7.log
"""

import os
import time
from fractions import Fraction

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "intermediate", "seir7_t347")
T0 = time.time()

STATES = ("AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN "
          "MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA "
          "WV WI WY").split()
AGES = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44",
        "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75+"]
# Fig. 3 of the paper: HHS region headquarters
HHS = [("Boston", "MA"), ("New York", "NY"), ("Philadelphia", "PA"), ("Atlanta", "GA"),
       ("Chicago", "IL"), ("Dallas", "TX"), ("Kansas City", "MO"), ("Denver", "CO"),
       ("San Francisco", "CA"), ("Seattle", "WA")]


def say(*args):
    print("[%7.2fs]" % (time.time() - T0), *args, flush=True)


def load(name):
    """A CSV written by Mathematica; entries may be exact rationals such as 1/5."""
    with open(os.path.join(DATA, name + ".csv")) as f:
        rows = [[float(Fraction(x.strip().strip('"'))) for x in line.split(",")]
                for line in f if line.strip()]
    return np.array(rows)


def load_scalars():
    out = {}
    with open(os.path.join(DATA, "scalars.csv")) as f:
        for line in f:
            key, val = (x.strip().strip('"') for x in line.split(","))
            out[key] = float(Fraction(val))
    return out


# ----------------------------------------------------------------------------- data
mN = load("mN")
m, n = mN.shape
usc = load("USC")
lr = load("LR")[:, 0]
dist = load("dist")
b = load("b")[:, 0]
ma_nb = load("ma")
vS = load("vS")[:, 0]
psiS = load("vpsiS")[:, 0]
psiI = load("vpsiI")[:, 0]
beta = load("vbeta")[:, 0]
vp = load("vp")[:, 0]
vq = load("vq")[:, 0]
ga, gs, gm, gh = load("vgamma").T
ds, dm = load("vdelta").T
sc = load_scalars()

# cell 53 of the notebook:  mC[p] = LR[p] USC,  ma[p,q] = sum_j mC[p,j,q],
# mc[i,j,p,q] = mC[p,j,q] exp(-b[q] dist[i,p]) / sum_{s,r} mC[s,j,r] exp(-b[r] dist[i,s])
ma = lr[:, None] * usc.sum(axis=0)[None, :]
_expo = np.exp(-b[None, None, :] * dist[:, :, None])            # (i, p, q)
_K = lr[None, None, :, None] * usc[None, :, None, :] * _expo[:, None, :, :]
mc = _K / _K.sum(axis=(2, 3), keepdims=True)
del _K, _expo
C = mc.reshape(m * n, m * n)       # row (i,j), column (p,q)

om, omv = sc["omega"], sc["omega_v"]
e1, e2, e3 = sc["eps1"], sc["eps2"], sc["eps3"]
k, xi = sc["k"], sc["xi"]
ea, eps_, epm, es, em, eh = (sc[s] for s in ("eta_a", "eta_ps", "eta_pm", "eta_s", "eta_m", "eta_h"))

Qs = ea * vp / ga + eps_ * (1 - vp) / xi + es * (1 - vp) / (gs + ds) + eh * (1 - vp) * ds / (gs + ds) / gh
Qm = ea * vq / ga + epm * (1 - vq) / xi + em * (1 - vq) / (gm + dm) + eh * (1 - vq) * dm / (gm + dm) / gh

# the age dependent rates, in the form full_fv takes them
AGE_PRM = dict(k=k, xi=xi, vp=vp, vq=vq, ga=ga, gs=gs, gm=gm, gh=gh, ds=ds, dm=dm,
               eta=(ea, eps_, es, eh, epm, em))


def yz(chi1, chi2, chi3):
    """Y, Z (including the factor d_c) and d_c, shape (m, n) each."""
    c1 = chi1 / (omv + e2 * chi2)
    c2 = e2 * chi2 / om
    c3 = e3 * chi3 / om
    sdfe = mN / (1.0 + c1 * (1.0 + c2 + c2 * (1.0 + c3)))
    vdfe = c1 * sdfe
    pdfe = c2 * vdfe
    pref = vS[:, None] * ma * (psiS * beta)[None, :]
    return pref * sdfe / mN, pref * ((1.0 - e1) * vdfe + pdfe) / mN, pref


def h_operator(Y, Z, qs, qm):
    """NGM and NGM^T as a LinearOperator on vectors indexed (i, j, a), size 2 m n."""
    Om = np.stack([Y, Z], axis=-1).reshape(m * n, 2)
    Qd = np.stack([np.broadcast_to(qs, (m, n)), np.broadcast_to(qm, (m, n))], axis=-1).reshape(m * n, 2)
    Qd = Qd * np.tile(psiI, m)[:, None]

    def mv(x):
        x = x.reshape(m * n, 2)
        return (Om * (C @ (Qd * x).sum(axis=1))[:, None]).ravel()

    def rmv(x):
        x = x.reshape(m * n, 2)
        return (Qd * (C.T @ (Om * x).sum(axis=1))[:, None]).ravel()

    N = 2 * m * n
    return spsl.LinearOperator((N, N), matvec=mv, rmatvec=rmv, dtype=float), Om, Qd


def perron(op, transpose=False):
    a = op.adjoint() if transpose else op
    lam, vec = spsl.eigs(a, k=1, which="LM", tol=1e-14)
    v = np.real(vec[:, 0])
    v = v * np.sign(v.sum())
    return float(np.real(lam[0])), float(np.abs(np.imag(lam[0]))), v


def full_fv(Y, Z, Cmat, psi_flat, n_age, prm, verbose=True):
    """rho(F V^-1) of the 8-state system, states (Es, Ep, Ia, Ips, Is, Ih, Ipm, Im).

    Y, Z: shape (G,) or (G/n_age, n_age), G groups ordered p = i n_age + j;
    Cmat: G x G contact proportions; psi_flat: psi_I per group, shape (G,);
    prm: k, xi and the age dependent rates vp, vq, ga, gs, gm, gh, ds, dm
    (length n_age), and eta = (eta_a, eta_ps, eta_s, eta_h, eta_pm, eta_m).
    The value does not depend on k > 0.
    """
    yv, zv = np.ravel(Y), np.ravel(Z)
    G = yv.size
    kk, xx = prm["k"], prm["xi"]
    eta = np.array(prm["eta"])                       # columns Ia, Ips, Is, Ih, Ipm, Im
    if verbose:
        say("  assembling V, size %d, block diagonal 8 x 8" % (8 * G))
    rows, cols, vals = [], [], []
    for g in range(G):
        j = g % n_age
        r = 8 * g
        pj, qj = prm["vp"][j], prm["vq"][j]
        ent = [(0, 0, kk), (1, 1, kk),
               (2, 0, -kk * pj), (2, 1, -kk * qj), (2, 2, prm["ga"][j]),
               (3, 0, -kk * (1 - pj)), (3, 3, xx),
               (4, 3, -xx), (4, 4, prm["gs"][j] + prm["ds"][j]),
               (5, 4, -prm["ds"][j]), (5, 7, -prm["dm"][j]), (5, 5, prm["gh"][j]),
               (6, 1, -kk * (1 - qj)), (6, 6, xx),
               (7, 6, -xx), (7, 7, prm["gm"][j] + prm["dm"][j])]
        for a_, b_, v_ in ent:
            rows.append(r + a_)
            cols.append(r + b_)
            vals.append(v_)
    V = sps.csc_matrix((vals, (rows, cols)), shape=(8 * G, 8 * G))
    lu = spsl.splu(V)
    if verbose:
        say("  sparse LU of V done, nnz(V) = %d" % V.nnz)

    def fmv(x):
        x = x.reshape(G, 8)
        s = Cmat @ (psi_flat * (x[:, 2:] @ eta))
        out = np.zeros((G, 8))
        out[:, 0] = yv * s
        out[:, 1] = zv * s
        return out.ravel()

    op = spsl.LinearOperator((8 * G, 8 * G), matvec=lambda x: fmv(lu.solve(x)), dtype=float)
    lam = spsl.eigs(op, k=1, which="LM", tol=1e-14, return_eigenvectors=False)
    return float(np.real(lam[0]))


def main():
    say("data from", DATA)
    say("m = %d locations, n = %d age groups, N = %.6e people" % (m, n, mN.sum()))
    say("scalars:", ", ".join("%s = %g" % kv for kv in sc.items()))
    say("ma vs notebook: max rel diff = %.3e" % (np.abs(ma - ma_nb).max() / ma_nb.max()))
    sl = load("mc_slices").reshape(2, n, m, n)
    for kk, i in enumerate((0, 32)):
        say("mc slice i = %s vs Mathematica: max rel diff = %.3e"
            % (STATES[i], np.abs(mc[i] - sl[kk]).max() / sl[kk].max()))
    say("row sums of mc: min %.15f, max %.15f" % (mc.sum(axis=(2, 3)).min(), mc.sum(axis=(2, 3)).max()))
    say("Q_s by age:", np.array2string(Qs, precision=4, max_line_width=200))
    say("Q_m by age:", np.array2string(Qm, precision=4, max_line_width=200))

    zero = np.zeros((m, n))
    Y, Z, pref = yz(zero, zero, zero)
    say("chi = 0: Y in [%.4e, %.4e], max |Z| = %.1e" % (Y.min(), Y.max(), np.abs(Z).max()))

    say("Arnoldi for rho(NGM) ...")
    H, Om, Qd = h_operator(Y, Z, Qs, Qm)
    rho, im, v = perron(H)
    say("rho(NGM) = %.12f   (|imag| = %.1e)" % (rho, im))
    say("check NGM v = rho v: |NGM v - rho v| / |v| = %.2e"
        % (np.linalg.norm(H @ v - rho * v) / np.linalg.norm(v)))

    say("rho(F V^-1), full 8-state system, matrix free ...")
    rho_fv = full_fv(Y, Z, C, np.tile(psiI, m), n, AGE_PRM)
    say("rho(F V^-1) = %.12f   (rel. diff to rho(NGM): %.3e)" % (rho_fv, (rho_fv - rho) / rho))

    say("Arnoldi for the left eigenvector (NGM^T) ...")
    rho_l, _, w = perron(H, transpose=True)
    say("rho(NGM^T) = %.12f, min component of w = %.3e" % (rho_l, w.min()))
    w = np.abs(w)
    wS = w.reshape(m, n, 2)[:, :, 0]
    wS = wS / wS.sum()
    say("left eigenvector, S part, share by age (sum over 51 locations; total 1):")
    for a_, s_ in zip(AGES, wS.sum(axis=0)):
        print("    %-6s %.4f  %s" % (a_, s_, "#" * int(round(400 * s_))), flush=True)
    state_share = wS.sum(axis=1)
    order = np.argsort(-state_share)
    say("left eigenvector, S part, share by location, largest 10 and smallest 5:")
    for i in list(order[:10]) + list(order[-5:]):
        print("    %s %.4f  (population share %.4f)" % (STATES[i], state_share[i], mN[i].sum() / mN.sum()),
              flush=True)
    percap_age = wS.sum(axis=0) / mN.sum(axis=0)
    percap_age /= percap_age.max()
    say("left eigenvector, S part, per capita by age (relative to max):")
    print("    " + "  ".join("%s:%.2f" % (a_, s_) for a_, s_ in zip(AGES, percap_age)), flush=True)

    # d rho / d Omega_k = w_k (A v)_k / <w, v>, with A v = NGM v / Omega (computed without dividing)
    Av = C @ (Qd * v.reshape(m * n, 2)).sum(axis=1)
    wv = float(w @ v)
    wr = w.reshape(m * n, 2)
    # at chi = 0:  dY/dchi1 = -pref / omega_v,  dZ/dchi1 = (1 - eps1) pref / omega_v
    dY = -pref.ravel() / omv
    dZ = (1 - e1) * pref.ravel() / omv
    grad = ((wr[:, 0] * dY + wr[:, 1] * dZ) * Av / wv).reshape(m, n)
    say("gradient d rho / d chi1 at chi = 0: all entries negative: %s, range [%.4e, %.4e]"
        % (bool((grad < 0).all()), grad.min(), grad.max()))
    i0, j0 = STATES.index("GA"), 3
    hstep = 1e-6
    ch = zero.copy()
    ch[i0, j0] = hstep
    Yp, Zp, _ = yz(ch, zero, zero)
    rp = perron(h_operator(Yp, Zp, Qs, Qm)[0])[0]
    say("finite difference check at (GA, 15-19): adjoint %.6e, forward difference (h = %g) %.6e"
        % (grad[i0, j0], hstep, (rp - rho) / hstep))

    say("-d rho / d chi1 at chi = 0 by age, HHS region headquarter states (Fig. 3A of the paper):")
    print("    %-18s" % "" + " ".join("%6s" % a_ for a_ in AGES), flush=True)
    for city, st in HHS:
        i = STATES.index(st)
        print("    %-18s" % ("%s (%s)" % (city, st)) + " ".join("%6.3f" % (-g) for g in grad[i]), flush=True)
    say("done")


if __name__ == "__main__":
    main()
