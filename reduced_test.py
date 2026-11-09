"""Reduced problem (A1 diag(g), order 16) against the original one (NGM, order 32).

Reference for matrix entries and eigenvalues: eigenvalues_H_ZF_JG.nb, its own
input cells evaluated in a Wolfram kernel with chi substituted.  The only files
written are cases.json and ref_h.json, in intermediate/.

Usage:
    python3 -u reduced_test.py
"""
import json, os, subprocess, sys
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
from scipy.optimize import minimize, LinearConstraint

D = os.path.dirname(os.path.abspath(__file__))     # ref_h.wls and the notebook are here
S = os.path.join(D, "intermediate")                # cases.json, ref_h.json
from zf_jg_params import default_params
from zf_jg_doses import dose_budgets
import gp_single_dose as G

par = default_params()
m, n = par["m"], par["n"]
N = m * n
e1, e2, e3 = par["eps1"], par["eps2"], par["eps3"]
om, omv = par["omega"], par["omega_v"]

ac, _, dc = G.build_a1(par)                      # Ac of the tex, order 2N, sparse
A1 = ac[0::2][:, 0::2].tocsr()                   # order N
P = sps.kron(sps.identity(N), np.ones((2, 1))).tocsr()


def pis(u1, u2, u3):
    return u1 / (omv + e2 * u2), e2 * u2 / om, 1.0 + e3 * u3 / om


def yz(u1, u2, u3):
    """(fractions) and (yz-def) of the tex."""
    p1, p2, p3 = pis(u1, u2, u3)
    s = 1.0 / (1.0 + p1 * (1.0 + p2 + p2 * p3))
    return s, ((1.0 - e1) + p2) * p1 * s


def gfun(u1, u2, u3):
    """(g-def) of the tex, the closed form."""
    p1, p2, p3 = pis(u1, u2, u3)
    return (1.0 + (1.0 - e1) * p1 + p1 * p2) / (1.0 + p1 + p1 * p2 + p1 * p2 * p3)


def perron(mat):
    lam, v = spsl.eigs(mat, k=1, which="LR", tol=0)
    _, w = spsl.eigs(mat.T.tocsr(), k=1, which="LR", tol=0)
    v, w = np.real(v[:, 0]), np.real(w[:, 0])
    return float(np.real(lam[0])), v * np.sign(v.sum()), w * np.sign(w.sum())


def full_matrix(u):
    y, z = yz(*u)
    d = np.empty(2 * N); d[0::2] = y; d[1::2] = z
    return (sps.diags(d) @ ac).tocsr()


def red_matrix(u):
    return (A1 @ sps.diags(gfun(*u))).tocsr()


class Problem:
    """min log rho over the normalized variables chi-hat, sum_j chi-hat = 1."""

    def __init__(self, mode, scales):
        self.mode, self.scales, self.K = mode, [s.ravel() for s in scales], len(scales)

    def u(self, x):
        x = np.asarray(x).reshape(self.K, N)
        u = [np.zeros(N), np.zeros(N), np.zeros(N)]
        for k in range(self.K):
            u[k] = self.scales[k] * x[k]
        return u

    def value_grad(self, x):
        u = self.u(x)
        h = 1e-30
        grad = np.empty((self.K, N))
        if self.mode == "reduced":
            rho, v, w = perron(red_matrix(u))
            drho = (A1.T @ w) * v / float(w @ v)
            for k in range(self.K):
                uc = [a.astype(complex) for a in u]; uc[k] = uc[k] + 1j * h
                grad[k] = drho * np.imag(gfun(*uc)) / h
        else:
            rho, v, w = perron(full_matrix(u))
            sens = w * (ac @ v) / float(w @ v)
            for k in range(self.K):
                uc = [a.astype(complex) for a in u]; uc[k] = uc[k] + 1j * h
                yc, zc = yz(*uc)
                grad[k] = sens[0::2] * np.imag(yc) / h + sens[1::2] * np.imag(zc) / h
        for k in range(self.K):
            grad[k] *= self.scales[k] / rho
        return np.log(rho), grad.ravel()

    def solve(self, x0):
        e = np.zeros((self.K * m, self.K * N))
        for r in range(self.K * m):
            e[r, r * n:(r + 1) * n] = 1.0
        return minimize(self.value_grad, np.asarray(x0).ravel(), jac=True, method="trust-constr",
                        constraints=[LinearConstraint(e, np.ones(self.K * m), np.ones(self.K * m))],
                        bounds=[(0.0, None)] * (self.K * N),
                        options={"gtol": 1e-14, "xtol": 1e-15, "maxiter": 20000, "verbose": 0})


def chis(prob, x):
    return [a.reshape(m, n) for a in prob.u(x)]


# ------------------------------------------------------------------ the data
s_mat = np.loadtxt(os.path.join(D, G.S_CSV), delimiter=",")
p_mat = np.loadtxt(os.path.join(D, "old_files/solution_nb_Data/p1Matrix.csv"), delimiter=",")
b1_week, _, _ = dose_budgets()
OPT_H = np.array([[0.00300457816105132022, 0.006063747923999046954, 0.003545425576831823678, 2.236626067229857952e-14],
                  [0.004398722516730560125, 0.004449697286182037123, 0.005958627241133407706, 6.642807744610886198e-11],
                  [0.003537310173540902242, 0.004151002856664137187, 0.003181218010565318565, 2.718994501058789159e-14],
                  [0.006054774009199111705, 0.003829806663471072622, 0.004735452868049963744, 4.5967569049017104e-11]])
zero = np.zeros((m, n))
rng = np.random.default_rng(20260930)

print("=== 0. gradients of the two objectives against central differences ===")
for mode in ("reduced", "full"):
    pr = Problem(mode, [b1_week[:, None] / 7 / s_mat, b1_week[:, None] / 28 / p_mat, b1_week[:, None] / 28 / p_mat])
    x = rng.dirichlet(np.ones(n), size=3 * m).ravel()
    f, gr = pr.value_grad(x)
    eye = np.eye(3 * N)
    fd = np.array([(pr.value_grad(x + 1e-6 * eye[i])[0] - pr.value_grad(x - 1e-6 * eye[i])[0]) / 2e-6
                   for i in range(3 * N)])
    print("  %-8s max |grad - fd| / max|grad| = %.2e" % (mode, np.abs(gr - fd).max() / np.abs(gr).max()))

# ------------------------------------------------------------ optimizations
runs = []      # (label, chi_red triple, chi_full triple, rho_red, rho_full)

print("\n=== 1. one dose (b2 = b3 = 0): the convex program of the tex against the 32 x 32 problem ===")
for lab, days in (("b1/7 (daily)", 7.0), ("b1 (weekly, tex)", 1.0), ("3 b1 (test)", 1.0 / 3.0)):
    b1 = b1_week / days
    red = G.Reduced(A1.toarray(), s_mat, b1, e1, omv)            # the algorithm of the tex, as is
    ra = G.solve_route_a(red, np.full(N, 1.0 / n))
    chi_r = omv * (red.t_of(ra.x).reshape(m, n) - 1.0)
    rho_r = perron(red_matrix([chi_r.ravel(), np.zeros(N), np.zeros(N)]))[0]
    pf = Problem("full", [b1[:, None] / s_mat])
    rf = pf.solve(np.full(N, 1.0 / n))
    chi_f = chis(pf, rf.x)[0]
    rho_f = perron(full_matrix(pf.u(rf.x)))[0]
    print("  budget %-12s rho_reduced = %.10f   rho_full = %.10f   diff = %.2e   max|chi_red - chi_full| = %.2e (max chi %.2e)"
          % (lab, rho_r, rho_f, abs(rho_r - rho_f), np.abs(chi_r - chi_f).max(), chi_r.max()))
    print("         doses used / budget: reduced %s   full %s"
          % (np.array2string((chi_r * s_mat).sum(1) / b1, precision=10),
             np.array2string((chi_f * s_mat).sum(1) / b1, precision=10)))
    runs.append(("1 dose, " + lab, [chi_r, zero, zero], [chi_f, zero, zero], rho_r, rho_f))

print("\n=== 2. three doses, test data sigma2 = sigma3 = P, b2 = b3 = b1/4, daily; same start for both ===")
scales3 = [b1_week[:, None] / 7 / s_mat, b1_week[:, None] / 28 / p_mat, b1_week[:, None] / 28 / p_mat]
starts = [("uniform", np.full(3 * N, 1.0 / n))]
starts += [("random %d" % (r + 1), rng.dirichlet(np.ones(n), size=3 * m).ravel()) for r in range(3)]
for lab, x0 in starts:
    pr_, pf_ = Problem("reduced", scales3), Problem("full", scales3)
    rr, rf = pr_.solve(x0), pf_.solve(x0)
    cr, cf = chis(pr_, rr.x), chis(pf_, rf.x)
    rho_r = perron(red_matrix(pr_.u(rr.x)))[0]
    rho_f = perron(full_matrix(pf_.u(rf.x)))[0]
    print("  start %-9s rho_reduced = %.10f   rho_full = %.10f   diff = %.2e   max|chi-hat_red - chi-hat_full| = %.2e"
          % (lab, rho_r, rho_f, abs(rho_r - rho_f), np.abs(rr.x - rf.x).max()))
    runs.append(("3 doses, start " + lab, cr, cf, rho_r, rho_f))

# ------------------------------------------------- reference: the H notebook
cases = [("chi = 0", [zero, zero, zero]), ("OptimalChi1 of H", [OPT_H, zero, zero])]
for r in range(3):
    sc = 10.0 ** rng.uniform(-4, -1)
    cases.append(("random chi1 (%.1e)" % sc, [sc * rng.random((m, n)), zero, zero]))
for r in range(4):
    sc = 10.0 ** rng.uniform(-4, -1)
    cases.append(("random chi1,2,3 (%.1e)" % sc,
                  [sc * rng.random((m, n)), sc * rng.random((m, n)), sc * rng.random((m, n))]))
for lab, cr, cf, _, _ in runs:
    cases.append((lab + " | reduced min", cr))
    cases.append((lab + " | full min", cf))

fin, fout = os.path.join(S, "cases.json"), os.path.join(S, "ref_h.json")
with open(fin, "w") as fh:
    json.dump([{"name": nm, "chi1": c[0].tolist(), "chi2": c[1].tolist(), "chi3": c[2].tolist()}
               for nm, c in cases], fh)
with open(fout, "w") as fh:
    fh.write("")
res = subprocess.run(["wolframscript", "-file", os.path.join(D, "ref_h.wls"),
                      os.path.join(D, "eigenvalues_H_ZF_JG.nb"), fin, fout], capture_output=True, text=True)
print("\n" + res.stdout.strip())
if os.path.getsize(fout) == 0:
    sys.stderr.write(res.stderr)
    raise SystemExit("no reference produced")
with open(fout) as fh:
    ref = json.load(fh)

print("\n=== 3. entries and eigenvalues against eigenvalues_H_ZF_JG.nb ===")
print("  entries : max |diag(d) P A1 P^T - NGM_nb| / max|NGM_nb|,  d from (fractions),(yz-def) of the tex")
print("  eigs    : the 14 largest eigenvalues of A1 diag(g) (Arnoldi), distance to the nearest of the 16 the kernel")
print("            returns for NGM_nb (Arnoldi), over rho;  rho : |rho(A1 diag(g)) - Re[Eigenvalues[NGM,1]]| / rho")
hdr = "  %-36s %10s %10s %10s %16s %16s" % ("case", "entries", "eigs", "rho", "rho notebook", "rho reduced")
print(hdr)
print("  " + "-" * (len(hdr) - 2))
worst = 0.0
rho_nb = {}
for (nm, c), r in zip(cases, ref):
    u = [a.ravel() for a in c]
    ngm_nb = np.array(r["NGM"])
    y, z = yz(*u)
    d = np.empty(2 * N); d[0::2] = y; d[1::2] = z
    recs = (sps.diags(d) @ P @ A1 @ P.T).tocsr()
    e_ent = abs(recs - sps.csr_matrix(ngm_nb)).max() / np.abs(ngm_nb).max()
    mu = np.array([complex(a, b) for a, b in r["ev"]])
    lam = spsl.eigs(red_matrix(u), k=N - 2, which="LM", return_eigenvectors=False, tol=0)
    rho_r = perron(red_matrix(u))[0]
    e_eig = max(np.min(np.abs(l - mu)) for l in lam) / r["rho1"]
    e_rho = abs(rho_r - r["rho1"]) / r["rho1"]
    worst = max(worst, e_ent, e_eig, e_rho)
    rho_nb[nm] = r["rho1"]
    print("  %-36s %10.2e %10.2e %10.2e %16.12f %16.12f" % (nm, e_ent, e_eig, e_rho, r["rho1"], rho_r))
print("  worst over all cases: %.2e" % worst)

print("\n=== 4. the two minimizers, rho from the H notebook ===")
for lab, cr, cf, rho_r, rho_f in runs:
    a, b = rho_nb[lab + " | reduced min"], rho_nb[lab + " | full min"]
    print("  %-26s notebook rho at reduced min = %.10f   at full min = %.10f   diff = %.2e"
          % (lab, a, b, abs(a - b)))
