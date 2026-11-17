"""Solver of the ODE system of the age- and location-stratified SEIR model
(notes/ODE.tex, equations (3.1); numerical method in Section 7). All data and their sources:
notes/DATA.tex; all data are published or public.

Inputs (each built by its script from files in MathBio/data/Downloaded_from_source/):
  data/census/N_51x16.csv          census_N_51x16.py      populations N
  data/prem2017/U_16x16.csv        prem_U_16x16.py        contact matrix U
  data/census/d_51x51.csv, L_51.csv  geo_51.py            distances d, activity factors L
  data/glasser2023/table2.csv, beta0.csv  glasser2023_inputs.py   p, d, IFR, compliance, beta0
  data/oxcgrt/s_51_daily.csv       oxcgrt_s_51.py         stringency factors s_i(t)
  data/model/C_816x816.csv.gz      build_C_51x16.py       contact matrix C (from U, L, d, b)
  data/cdc/variant_days.csv        cdc_variants.py        days of the variants of Table S4
  data/cdc/state_t251_mock.csv     sero_initial_51x16.py  initial state (MOCK)
  data/mock/doses_51x16x16_mock.xlsx  mock_doses_51x16x16.py  weekly doses (MOCK)

The state vector has the order: compartment (S, P, Es, Ep, V, Ia, Ips, Ipm, Is, Im, Ih, R, M),
then location i (FIPS order), then age j. The force of infection uses the contact matrix C
read from data/model/C_816x816.csv.gz; the system is integrated by DOP853, interval by interval
between the variant days (and the weeks of vaccination).

Usage:
    python3 -u seir7_ode.py verify   | tee intermediate/seir7_ode_verify.output
        tolerances, group sums, signs, spectral radius on [251, 349]
    python3 -u seir7_ode.py solve    | tee intermediate/seir7_ode_solve.output
        state at t = 349 (start of vaccination) -> data/state_t349_51x16.csv
    python3 -u seir7_ode.py simulate [actual|optimal] [nrand] [resume] | tee -a intermediate/seir7_ode_simulate_<strategy>.output
        (after each week the state is saved to intermediate/simulate_<strategy>_checkpoint.npz;
        "resume" continues an interrupted run from the last completed week)
        from t = 251 through the weeks of the dose file. actual: the rates (6.1) of ODE.tex from
        the doses by location and age; optimal: each week, the doses of each location (all ages)
        allocated among its age groups by the alternating method (alternating.py, notes/
        stratified_seir_ngm.tex) with sigma = S, V, P of the solution at the start of the week,
        from the uniform and left-eigenvector starts and nrand random ones (default 0)
        -> data/simulate_weekly_51x16_<strategy>.csv (national totals at the start of each week),
           and for optimal data/simulate_weekly_51x16_optimal_rho.csv (rho per week)
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.sparse.linalg import LinearOperator, eigs

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CENSUS_N = os.path.join(DATA, "census", "N_51x16.csv")
PREM_U = os.path.join(DATA, "prem2017", "U_16x16.csv")
C_FILE = os.path.join(DATA, "model", "C_816x816.csv.gz")
DOSES = os.path.join(DATA, "mock", "doses_51x16x16_mock.xlsx")
INITIAL = os.path.join(DATA, "cdc", "state_t251_mock.csv")
STATE349 = os.path.join(DATA, "state_t349_51x16.csv")
WEEKLY = os.path.join(DATA, "simulate_weekly_51x16.csv")

# Constants (per day): supplement of Feng, Day, Glasser 2026, Table S1 (xi, gamma, omega, b_S, b_I,
# gamma_{s->h}), Section 2 (eta); deaths 30% outside and 70% in hospital (Glasser et al. 2023, Sec. 3).
XI = 1 / 2
GAMMA_A, GAMMA_S, GAMMA_M, GAMMA_H = 1 / 7, 1 / 5, 1 / 5, 1 / 10
GAMMA_S_TO_H = 1 / 2
OMEGA = 1 / 365
ETA = (0.5, 1.25, 1.0, 1.0, 0.75, 0.1)          # a, ps, pm, s, m, h
B_SI = 0.31
THETA_SPLIT = (0.3, 0.7)
T_START, T_VACC = 251.0, 349.0                  # 7 September and 14 December 2020

# Table S4 of the supplement: factor of beta, epsilon_1,2,3, factor of the compliance, k, factor of
# the hospitalization rates and of the death fractions; None = unchanged from the previous variant.
VARIANTS = {"Wuhan": (1.0, 0.9, 1.0, 1 / 5, 1.0), "Alpha": (1.25, 0.8, 0.8, None, 1.0),
            "Delta": (1.5, 0.7, 0.6, 1 / 4, 1.0), "BA.1": (2.0, 0.6, 0.4, None, 0.67),
            "BA.5": (1.0, 0.5, 0.2, 1 / 3, 0.67)}

NAMES = ["S", "P", "Es", "Ep", "V", "Ia", "Ips", "Ipm", "Is", "Im", "Ih", "R", "M"]
(S, P, ES, EP, V, IA, IPS, IPM, IS, IM, IH, R, M) = range(13)
INFECTIOUS = [IA, IPS, IPM, IS, IM, IH]


def decay_rates(U, d):
    """b_q with sum_l U_ql = (1/m) sum_{r,k} exp(-b_q d_rk) (Feng, Hill, Curns, Glasser 2017,
    Section 2.2.2.2). The right side decreases from m (b = 0) to 1 (b -> infinity), the
    diagonal d_rr = 0 contributing 1; a root exists when 1 < sum_l U_ql < m."""
    m = d.shape[0]
    A = U.sum(axis=1)
    if np.any(A <= 1) or np.any(A >= m):
        raise ValueError(f"marginal contact rates {A.min():.3g} - {A.max():.3g} outside (1, {m})")
    g = lambda b, a: np.exp(-b * d).sum() / m - a
    return np.array([brentq(g, 0.0, 1.0, args=(a,), xtol=1e-15, rtol=1e-14) for a in A])


class Model:
    """Parameters of ODE.tex, Sections 2 and 4, with the data of DATA.tex."""

    def __init__(self, initial="mock"):
        pop = pd.read_csv(CENSUS_N)
        self.locs = list(dict.fromkeys(pop.state))
        t2 = pd.read_csv(os.path.join(DATA, "glasser2023", "table2.csv"))
        self.ages = list(t2.age)
        m, n = len(self.locs), len(self.ages)
        self.m, self.n = m, n
        if list(pop.state) != list(np.repeat(self.locs, n)) or list(pop.age) != self.ages * m:
            raise ValueError("N_51x16.csv not in the order (location, age) of the model")
        self.N = pop.N.to_numpy(float).reshape(m, n)
        Uf = pd.read_csv(PREM_U, index_col=0)
        D = pd.read_csv(os.path.join(DATA, "census", "d_51x51.csv"), index_col=0)
        Lf = pd.read_csv(os.path.join(DATA, "census", "L_51.csv"))
        sf = pd.read_csv(os.path.join(DATA, "oxcgrt", "s_51_daily.csv"), index_col=0)
        bf = pd.read_csv(os.path.join(DATA, "glasser2023", "beta0.csv"))
        if (list(Uf.index) != self.ages or list(D.index) != self.locs or list(Lf.state) != self.locs
                or list(sf.columns) != self.locs or list(bf.age) != self.ages):
            raise ValueError("locations or ages of an input file differ from the model's")
        self.U = Uf.to_numpy(float)                                          # U[j, q]
        self.d = D.to_numpy(float)                                           # d[i, p], miles
        self.L = Lf.L.to_numpy(float)
        self.s_t = sf.index.to_numpy(float)
        self.s_val = sf.to_numpy(float)                                      # s_i(t), daily
        self.b = decay_rates(self.U, self.d)
        self.p = t2.p.to_numpy(float)
        self.q = 1 - 0.5 * (1 - self.p)                                     # paper, Table 2
        self.delta_s0 = GAMMA_S_TO_H * t2.d.to_numpy(float)
        self.delta_m0 = 0.5 * self.delta_s0                                 # paper, Table 2
        sfr = t2.IFR_percent.to_numpy(float) / 100 * (1 - self.p)          # supplement, Section 2
        self.die_s0, self.die_h0 = THETA_SPLIT[0] * sfr, THETA_SPLIT[1] * sfr   # 1 - theta
        self.c0 = t2.compliance.to_numpy(float)
        self.beta0 = bf.beta0.to_numpy(float)
        self.xi, self.omega, self.omega_v = XI, OMEGA, OMEGA
        self.gamma_a, self.gamma_s, self.gamma_m, self.gamma_h = GAMMA_A, GAMMA_S, GAMMA_M, GAMMA_H
        self.eta = np.array(ETA)
        self.bS = self.bI = B_SI
        # variants: switch days from cdc_variants.py, factors of Table S4
        vd = pd.read_csv(os.path.join(DATA, "cdc", "variant_days.csv"))
        self.var_t, self.var_par, prev = [], [], None
        for name, t in zip(vd.variant, vd.t):
            f = VARIANTS[name]
            par = tuple(prev[i] if f[i] is None else f[i] for i in range(5)) if prev else f
            self.var_t.append(float(t))
            self.var_par.append(par)
            prev = par
        self.breaks = [t for t in self.var_t if np.isfinite(t)]
        self.k = self.var_par[0][3]                                          # Wuhan, for sero_initial
        self.delta_s, self.delta_m = self.delta_s0, self.delta_m0
        self.a = np.outer(self.L, self.U.sum(axis=0))                        # a_ij = L_i sum_l U_lj
        # the contact matrix C, formed by build_C_51x16.py from U, L, d, b and read from its file
        Cf = pd.read_csv(C_FILE, index_col=0, float_precision="round_trip")
        labels = [f"{s} {g}" for s in self.locs for g in self.ages]
        if list(Cf.index) != labels or list(Cf.columns) != labels:
            raise ValueError(f"rows or columns of {C_FILE} not in the order of the model")
        self.C = Cf.to_numpy(float)                                          # C[(i,j), (p,q)]
        self.chi = np.zeros((3, m, n))                                       # chi_1, chi_2, chi_3
        self.x251 = None
        if initial == "mock":
            st = pd.read_csv(INITIAL)
            self.x251 = np.stack([st[c].to_numpy(float).reshape(m, n) for c in NAMES]).ravel()

    def variant(self, t):
        """(beta factor, eps, compliance factor, k, severity factor) at time t."""
        i = int(np.searchsorted(self.var_t, t, side="left")) - 1
        return self.var_par[max(i, 0)]

    def ngm_context(self, t):
        """The model of the weekly problem of notes/stratified_seir_ngm.tex at time t, as the
        arguments of alternating.set_model: C, d_c = s_i a_ij psi_S,j beta_j, psi_I q_s, q_m/q_s,
        omega, omega_v, eps_1, eps_2, eps_3, with the parameters (variant, s_i) at t."""
        bf, e, cf, _, sev = self.variant(t)
        c = cf * self.c0
        psiS, psiI = 1.0 - c * self.bS, 1.0 - c * self.bI
        ds, dm = sev * self.delta_s0, sev * self.delta_m0
        ea, eps_, epm, es, em, eh = self.eta
        p, q, ga, gs, gm, gh, xi = self.p, self.q, self.gamma_a, self.gamma_s, self.gamma_m, self.gamma_h, self.xi
        Qs = ea * p / ga + eps_ * (1 - p) / xi + es * (1 - p) / (gs + ds) + eh * (1 - p) * ds / (gs + ds) / gh
        Qm = ea * q / ga + epm * (1 - q) / xi + em * (1 - q) / (gm + dm) + eh * (1 - q) * dm / (gm + dm) / gh
        dc = self.s(t)[:, None] * self.a * (psiS * bf * self.beta0)[None, :]
        return (self.m, self.n, self.C, dc.ravel(), np.tile(psiI * Qs, self.m),
                np.tile(Qm / Qs, self.m), self.omega, self.omega_v, e, e, e)

    def s(self, t):
        """s_i(t) = 1 - SI_i(t)/100, linear between the daily values, constant after the last."""
        return np.array([np.interp(t, self.s_t, self.s_val[:, i]) for i in range(self.m)])

    def Lam(self, t, x):
        """lambda_ij of (2.1), linear in x."""
        X = x.reshape(13, self.m, self.n)
        bf, _, cf, _, _ = self.variant(t)
        c = cf * self.c0
        psiS, psiI = 1.0 - c * self.bS, 1.0 - c * self.bI
        J = np.tensordot(self.eta, X[INFECTIOUS], axes=1)
        y = psiI[None, :] * J / self.N
        return (psiS[None, :] * self.s(t)[:, None] * self.a * (bf * self.beta0)[None, :]
                * (self.C @ y.ravel()).reshape(self.m, self.n))

    def linear(self, t, x):
        """A(t) x: all terms of (3.1) except those with lambda."""
        X = x.reshape(13, self.m, self.n)
        _, e, _, k, sev = self.variant(t)
        c1, c2, c3 = self.chi
        xi, om, omv = self.xi, self.omega, self.omega_v
        ga, gs, gm, gh = self.gamma_a, self.gamma_s, self.gamma_m, self.gamma_h
        p, q = self.p, self.q
        ds, dm = sev * self.delta_s0, sev * self.delta_m0
        ths, thh = 1 - sev * self.die_s0, 1 - sev * self.die_h0
        F = np.empty_like(X)
        F[S] = om * X[P] + omv * X[V] - c1 * X[S]
        F[V] = c1 * X[S] - (omv + e * c2) * X[V]
        F[P] = om * X[R] - (om + e * c3) * X[P]
        F[ES] = -k * X[ES]
        F[EP] = -k * X[EP]
        F[IA] = k * p * X[ES] + k * q * X[EP] - ga * X[IA]
        F[IPS] = k * (1 - p) * X[ES] - xi * X[IPS]
        F[IPM] = k * (1 - q) * X[EP] - xi * X[IPM]
        F[IS] = xi * X[IPS] - (gs + ds) * X[IS]
        F[IM] = xi * X[IPM] - (gm + dm) * X[IM]
        F[IH] = ds * X[IS] + dm * X[IM] - gh * X[IH]
        F[R] = (ga * X[IA] + ths * gs * X[IS] + gm * X[IM] + thh * gh * X[IH]
                + e * c2 * X[V] + e * c3 * X[P] - om * X[R])
        F[M] = (1 - ths) * gs * X[IS] + (1 - thh) * gh * X[IH]
        return F

    def infection(self, t, lam, x):
        """Phi~(t; u, x) of Section 7.2, with lam = Lambda u."""
        X = x.reshape(13, self.m, self.n)
        e = self.variant(t)[1]
        F = np.zeros_like(X)
        F[S] = -lam * X[S]
        F[V] = -(1 - e) * lam * X[V]
        F[P] = -lam * X[P]
        F[ES] = lam * X[S]
        F[EP] = lam * ((1 - e) * X[V] + X[P])
        return F

    def f(self, t, x):
        return (self.linear(t, x) + self.infection(t, self.Lam(t, x), x)).ravel()

    def jvp(self, t, x, v):
        """Proposition 2: f_x(t, x) v = A v + Phi~(v, x) + Phi~(x, v)."""
        return (self.linear(t, v) + self.infection(t, self.Lam(t, v), x)
                + self.infection(t, self.Lam(t, x), v)).ravel()

    def solve(self, t0, t1, x0, rtol=1e-10, atol=1e-8, t_out=None, keep_steps=False):
        """Integrate from t0 to t1, restarting at the variant days in (t0, t1).
        Returns (states at t_out (default [t1]), stats)."""
        t_out = [t1] if t_out is None else list(t_out)
        knots = [t0] + [b for b in self.breaks if t0 < b < t1] + [t1]
        x, out, stats = np.asarray(x0, float), {}, {"nfev": 0, "steps": []}
        for a, b in zip(knots[:-1], knots[1:]):
            te = [s for s in t_out if a < s <= b]
            sol = solve_ivp(self.f, (a, b), x, method="DOP853", rtol=rtol, atol=atol,
                            t_eval=None if keep_steps else (te or [b]))
            if sol.status != 0:
                raise RuntimeError(f"solve_ivp on [{a}, {b}]: {sol.message}")
            stats["nfev"] += sol.nfev
            x = sol.y[:, -1].copy()
            if keep_steps:
                stats["steps"].extend(np.diff(sol.t))
                if b in te:
                    out[b] = x
            else:
                for s, y in zip(sol.t, sol.y.T):
                    out[s] = y.copy()
        return [out.get(s) for s in t_out], stats


def group_sums(mod, x):
    return x.reshape(13, mod.m, mod.n).sum(axis=0)


def do_verify(mod):
    t0, t1 = T_START, T_VACC
    res = {}
    for rtol in (1e-6, 1e-8, 1e-10, 1e-12):
        tic = time.time()
        (x,), st = mod.solve(t0, t1, mod.x251, rtol=rtol, atol=rtol, keep_steps=True)
        h = np.array(st["steps"])
        res[rtol] = x
        print(f"rtol = atol = {rtol:.0e}: {len(h)} steps, h in [{h.min():.3g}, {h.max():.3g}],"
              f" nfev {st['nfev']}, {time.time() - tic:.1f} s")
    xb = res[1e-12]
    for rtol, x in res.items():
        print(f"   |x({t1:g}; {rtol:.0e}) - x({t1:g}; 1e-12)|_inf / |x|_inf = "
              f"{np.max(np.abs(x - xb)) / np.max(np.abs(xb)):.2e}")
    g0, g1 = group_sums(mod, mod.x251), group_sums(mod, xb)
    print(f"group sums: |sum(t0) - N| / N <= {np.max(np.abs(g0 - mod.N) / mod.N):.2e};"
          f" change on [{t0:g}, {t1:g}] <= {np.max(np.abs(g1 - g0) / mod.N):.2e} relative to N")
    X = xb.reshape(13, mod.m, mod.n)
    print(f"min over groups at {t1:g}: " + "  ".join(f"{nm} {X[c].min():.3g}" for c, nm in enumerate(NAMES)))
    ts = [t0, (t0 + t1) / 2, t1]
    xs, _ = mod.solve(t0, t1, mod.x251, rtol=1e-10, atol=1e-10, t_out=ts[1:])
    for t, x in zip(ts, [mod.x251] + xs):
        nn = x.size
        e = np.zeros(nn)
        norm1 = 0.0
        for c in range(nn):
            e[c] = 1.0
            norm1 = max(norm1, np.abs(mod.jvp(t, x, e)).sum())
            e[c] = 0.0
        op = LinearOperator((nn, nn), matvec=lambda v, t=t, x=x: mod.jvp(t, x, v), dtype=float)
        try:
            mu = eigs(op, k=1, which="LM", ncv=60, maxiter=2000, tol=1e-6, return_eigenvectors=False)
            est = f"Arnoldi |mu|max = {np.abs(mu).max():.4g} (mu = {mu[0].real:.4g}{mu[0].imag:+.3g}i)"
        except Exception as err:            # ArpackNoConvergence
            est = f"Arnoldi not converged ({type(err).__name__})"
        print(f"t = {t:g}: ||f_x||_1 = {norm1:.4g} (bound on rho); {est}")


def frame(mod, x):
    X = x.reshape(13, mod.m, mod.n)
    df = pd.DataFrame({"state": np.repeat(mod.locs, mod.n), "age": np.tile(mod.ages, mod.m)})
    for c, nm in enumerate(NAMES):
        df[nm] = X[c].ravel()
    return df


def do_solve(mod):
    tic = time.time()
    (x,), st = mod.solve(T_START, T_VACC, mod.x251)
    print(f"solved [{T_START:g}, {T_VACC:g}]: nfev {st['nfev']}, {time.time() - tic:.1f} s;"
          f" min compartment {x.min():.3g}")
    frame(mod, x).to_csv(STATE349, index=False, float_format="%.17g")
    print(f"wrote {STATE349}")


SIG_FLOOR = 1e-6      # persons: group sizes below this are raised to it (optimal strategy)
ELIG_MIN = 1.0        # persons: only groups with sigma_{k,p} >= ELIG_MIN may receive dose k
B_FLOOR = 1e-9        # persons per day: the optimizer needs b_{k,i} > 0
B_CAP = 0.99          # weekly doses of a location at most this fraction of its eligible people
CHI_MAX = -np.log(1 - B_CAP) / 7.0   # 0.658 per day: at most 99% of a group in a week, as in (6.1)


def cap_rates(chi, sig, elig, m, n):
    """chi_{k,p} <= CHI_MAX: the doses above the cap, sigma (chi - CHI_MAX), go to the other
    eligible groups of the same location in proportion to their doses sigma chi, repeated until
    no group exceeds the cap (possible since b_{k,i} <= B_CAP sum_eligible sigma / 7
    < CHI_MAX sum_eligible sigma). Returns the capped chi and the largest rate before capping."""
    out, top = [], 0.0
    for k in range(3):
        c = chi[k].reshape(m, n).copy()
        s = sig[k].reshape(m, n)
        el = elig[k].reshape(m, n)
        top = max(top, float(c.max()))
        for i in range(m):
            for _ in range(n):
                over = c[i] > CHI_MAX
                if not over.any():
                    break
                excess = (s[i, over] * (c[i, over] - CHI_MAX)).sum()
                c[i, over] = CHI_MAX
                free = ~over & (c[i] < CHI_MAX) & el[i]
                dose = s[i, free] * c[i, free]
                if dose.sum() > 0:
                    c[i, free] += excess * dose / dose.sum() / s[i, free]
                elif free.any():
                    c[i, free] += excess / s[i, free].sum()
        out.append(c.ravel())
    return out, top


def actual_rates(mod, X, Dw):
    """(6.1) of ODE.tex: chi = -ln(1 - min(D / X, 0.99)) / 7, for X = S, V, P."""
    chi = np.zeros((3, mod.m, mod.n))
    for k, comp in enumerate((S, V, P)):
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(X[comp] > 0, np.minimum(Dw[k] / X[comp], B_CAP), 0.0)
        chi[k] = -np.log1p(-frac) / 7.0
    return chi


def optimal_rates(mod, A, tau, X, Dw, nrand):
    """The weekly problem of notes/stratified_seir_ngm.tex at tau: minimize rho over chi with
    sum_{p in I_i} sigma_{k,p} chi_{k,p} = b_{k,i}; sigma_k = S, V, P at tau, b_{k,i} the doses
    of location i in the week (all ages) per day. Returns chi (3 x m x n) and a summary."""
    A.set_model(*mod.ngm_context(tau))
    elig = [(X[comp] >= ELIG_MIN).ravel() for comp in (S, V, P)]
    A.set_eligible(elig)
    A.set_cap(CHI_MAX)                            # chi <= CHI_MAX as a constraint of the problem
    sig = [np.maximum(X[comp], SIG_FLOOR).ravel() for comp in (S, V, P)]
    b = []
    for k in range(3):
        se = np.where(elig[k], sig[k], 0.0).reshape(mod.m, mod.n).sum(axis=1)   # eligible people
        bk = np.clip(Dw[k].sum(axis=1) / 7.0, B_FLOOR, np.maximum(B_CAP * se / 7.0, B_FLOOR))
        b.append(np.where(se > 0, bk, 0.0))       # no doses where no group is eligible
    ends = []
    for name, chi0 in A.starts(b, sig, nrand, seed=int(tau)):
        rho0 = A.rho_of(chi0)
        cm, rm, sw, _ = A.alternate(chi0, b, sig, label=f"[t={tau:g} {name}]")
        ends.append((rm, name, cm, sw, rho0))
    rm, name, chi, sw, _ = min(ends, key=lambda e: e[0])
    rho_uni = next(e[4] for e in ends if e[1] == "uniform")
    chi, top = cap_rates(chi, sig, elig, mod.m, mod.n)
    rho_cap = A.rho_of(chi)
    return (np.stack([c.reshape(mod.m, mod.n) for c in chi]),
            dict(rho_uniform=rho_uni, rho_opt=rm, rho_capped=rho_cap, chi_max_before_cap=top,
                 start=name, sweeps=sw))


def checkpoint_path(strategy):
    return os.path.join(HERE, "intermediate", f"simulate_{strategy}_checkpoint.npz")


def do_simulate(mod, strategy="actual", nrand=0, resume=False):
    """Weekly vaccination from T_VACC. strategy "actual" (not run): the rates of DATA.tex, Remark
    2.1, from the doses of the file by location and age; "optimal": the doses of each location
    (all ages) allocated among its age groups by the optimization of notes/stratified_seir_ngm.tex.
    After every week the state, the week and the results so far are saved to
    intermediate/simulate_<strategy>_checkpoint.npz; with resume=True the run continues from it."""
    if strategy not in ("actual", "optimal"):
        raise ValueError(f"strategy {strategy!r}")
    A = None
    if strategy == "optimal":
        import alternating as A
    sheets = pd.read_excel(DOSES, sheet_name=None)
    weeks = [float(c) for c in sheets["Dose1"].columns[2:]]
    Dk = np.stack([sheets[f"Dose{k}"].iloc[:, 2:].to_numpy(float).reshape(mod.m, mod.n, -1)
                   for k in (1, 2, 3)])                                      # 3 x m x n x weeks
    tic = time.time()
    ckpt = checkpoint_path(strategy)
    if resume and os.path.exists(ckpt):
        z = np.load(ckpt, allow_pickle=True)
        w0, x = int(z["next_week"]), z["x"].copy()
        rows, opt = list(z["rows"]), list(z["opt"])
        if not np.isclose(float(z["t"]), weeks[w0]):
            raise ValueError(f"{ckpt}: t = {float(z['t'])} is not the start of week {w0 + 1}")
        print(f"resuming from {ckpt}: week {w0 + 1} (t = {weeks[w0]:g})", flush=True)
    else:
        (x,), _ = mod.solve(T_START, T_VACC, mod.x251)
        w0, rows, opt = 0, [], []
    for w in range(w0, len(weeks)):
        tau = weeks[w]
        X = x.reshape(13, mod.m, mod.n)
        rows.append({"t": tau, **{nm: X[c].sum() for c, nm in enumerate(NAMES)}})
        Dw = Dk[:, :, :, w]
        if strategy == "actual":
            mod.chi[:] = actual_rates(mod, X, Dw)
        else:
            chi, info = optimal_rates(mod, A, tau, X, Dw, nrand)
            mod.chi[:] = chi
            opt.append({"t": tau, **info})
            print(f"week {w + 1} (t = {tau:g}): rho uniform {info['rho_uniform']:.6f}, optimal "
                  f"{info['rho_opt']:.6f} ({info['start']}, {info['sweeps']} sweeps), capped "
                  f"{info['rho_capped']:.6f} (max chi before cap {info['chi_max_before_cap']:.3g});"
                  f" {time.time() - tic:.0f} s", flush=True)
        (x,), _ = mod.solve(tau, tau + 7.0, x)
        if w + 1 < len(weeks):                 # the state at the start of the next week
            tmp = ckpt + ".tmp.npz"
            np.savez(tmp, next_week=w + 1, t=weeks[w + 1], x=x,
                     rows=np.array(rows, dtype=object), opt=np.array(opt, dtype=object))
            os.replace(tmp, ckpt)
    mod.chi[:] = 0.0
    X = x.reshape(13, mod.m, mod.n)
    rows.append({"t": weeks[-1] + 7.0, **{nm: X[c].sum() for c, nm in enumerate(NAMES)}})
    out = pd.DataFrame(rows)
    path = WEEKLY.replace(".csv", f"_{strategy}.csv")
    out.to_csv(path, index=False, float_format="%.6g")
    if opt:
        pd.DataFrame(opt).to_csv(path.replace(".csv", "_rho.csv"), index=False, float_format="%.10g")
    print(f"{strategy}: simulated [{T_START:g}, {weeks[-1] + 7:g}] with {len(weeks)} weeks of doses in "
          f"{time.time() - tic:.1f} s; at the end: V {out.V.iloc[-1] / 1e6:.1f} M, R {out.R.iloc[-1] / 1e6:.1f} M,"
          f" M (dead) {out.M.iloc[-1] / 1e3:.0f} K, hospitalized now {out.Ih.iloc[-1] / 1e3:.1f} K;"
          f" min compartment {x.min():.3g}")
    print(f"wrote {path}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "solve"
    mod = Model()
    print(f"N total {mod.N.sum():,.0f}; variant days " +
          ", ".join(f"{t:g}" for t in mod.breaks))
    if mode == "simulate":
        args = [a for a in sys.argv[2:] if a != "resume"]
        resume = "resume" in sys.argv[2:]
        strategy = args[0] if args else "optimal"   # actual: reference only, not run
        nrand = int(args[1]) if len(args) > 1 else 0
        do_simulate(mod, strategy, nrand, resume)
    else:
        {"verify": do_verify, "solve": do_solve}[mode](mod)
