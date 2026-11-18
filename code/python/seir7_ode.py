"""Solver of the ODE system of the age- and location-stratified SEIR model
(notes/DATA.tex, Section 2.1, equations (2)-(3); the computation in Section 2.2). All data and their sources:
notes/DATA.tex; all data are published or public.

Inputs (each built by its script from files in MathBio/data/Downloaded_from_source/):
  data/census/N_51x16.csv          census_N_51x16.py      populations N
  data/prem2017/U_16x16.csv        prem_U_16x16.py        contact matrix U
  data/census/d_51x51.csv, L_51.csv  geo_51.py            distances d, activity factors L
  ../../weekly_simulation/input/parameters.xlsx
                                   (edited by hand)       the parameter sets, one column each
                                   (2021: the values of Unknown 2020/21; feng2026: from
                                   glasser2023/table2.csv, beta0.csv, cdc/variant_days.csv and
                                   Tables S1, S4), then edited by hand; the set is chosen by
                                   params=NAME (default 2021)
  data/oxcgrt/s_51_daily.csv       oxcgrt_s_51.py         stringency factors s_i(t)
  data/model/C_816x816.csv.gz      build_C_51x16.py       contact matrix C (from U, L, d, b)
  data/cdc/state_t251_mock.csv     sero_initial_51x16.py  initial state (MOCK); format: read_state
  data/mock/doses_51x16x16_mock.xlsx  doses_51x16x16.py  weekly doses (MOCK): sheet 1
                                   by state (read for optimal), sheet 2 by state and age (actual)

The state vector has the order: compartment (S, P, Es, Ep, V, Ia, Ips, Ipm, Is, Im, Ih, R, M),
then location i (FIPS order), then age j. The force of infection uses the contact matrix C
read from data/model/C_816x816.csv.gz; the system is integrated by DOP853, interval by interval
between the switch days of the parameters (and the weeks of vaccination).

Every mode takes params=NAME, a set (column) of the parameter file (default 2021), and
paramfile=FILE, the parameter file (default weekly_simulation/input/parameters.xlsx); t_vacc,
the start of vaccination, is a parameter of the set (2021: 347, feng2026: 349).

Usage:
    python3 -u seir7_ode.py verify   | tee intermediate/seir7_ode_verify.output
        tolerances, group sums, signs, spectral radius on [251, t_vacc]
    python3 -u seir7_ode.py solve    | tee intermediate/seir7_ode_solve.output
        state at t_vacc (start of vaccination) -> data/state_t<t_vacc>_51x16__<params>.csv
    python3 -u seir7_ode.py simulate [actual|optimal] [resume] [initial=FILE] [doses=FILE] [params=NAME]
            | tee -a intermediate/seir7_ode_simulate_<strategy>__<tag>.output
        initial= the state at t = 251 (default data/cdc/state_t251_mock.csv), doses= the weekly
        doses (default data/mock/doses_51x16x16_mock.xlsx); a FILE is taken as given, else
        relative to this directory. tag = <initial>__<doses>__<params>, the two file names
        without directory and extension and the set, e.g.
        state_t251_mock__doses_51x16x16_mock__2021. out=DIR: the results and the checkpoint go
        to DIR (default: data/ and intermediate/). The number of weeks is that of the dose file:
        week w starts at t = t_vacc + 7 (w - 1), and the run ends at t_vacc + 7 (number of weeks).
        (after each week the state is saved to intermediate/simulate_<strategy>__<tag>_checkpoint.npz;
        "resume" continues an interrupted run with the same inputs from the last completed week)
        from t = 251 through the weeks of the dose file. actual: the rates of DATA.tex, Remark 2.1, from
        the doses by location and age; optimal: each week, the doses of each location (all ages)
        allocated among its age groups by the alternating method (alternating.py, notes/
        stratified_seir_ngm.tex) with sigma = S, V, P of the solution at the start of the week,
        from the uniform and left-eigenvector starts (and NRAND random ones, a constant below,
        0: random starts only to test for other local minima)
        -> data/simulate_weekly_51x16_<strategy>__<tag>.csv (national totals at the start of each
           week), and for optimal data/simulate_weekly_51x16_optimal__<tag>_rho.csv (rho per week)
    Any mode takes method=DOP853 (default; explicit RK of order 8, scipy) or method=SDIRK4
    (implicit RK of order 4 with step control, matrix-free Newton-GMRES with jvp; sdirk.py).
"""
import os
import resource
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
STATE_VACC = os.path.join(DATA, "state_t{t:g}_51x16__{params}.csv")    # the state at t_vacc
WEEKLY = os.path.join(DATA, "simulate_weekly_51x16.csv")


def input_path(p):
    """An input file given on the command line: as given (absolute, or relative to the current
    directory), else relative to this directory."""
    for q in (p, os.path.join(HERE, p)):
        if os.path.isfile(q):
            return os.path.abspath(q)
    raise FileNotFoundError(p)


def run_tag(initial, doses, params):
    """The part of the output names given by the inputs: <initial>__<doses>__<params>, the file
    names without directory and extension, and the name of the parameter set."""
    stem = lambda p: os.path.splitext(os.path.basename(p))[0]
    return f"{stem(initial)}__{stem(doses)}__{params}"

T_START = 251.0                                 # 7 September 2020, the initial state

# The parameters: one set (a column) of weekly_simulation/input/parameters.xlsx (edited by hand),
# chosen by params=NAME; the start of vaccination t_vacc is one of them.
PARAMS_FILE = os.path.normpath(os.path.join(HERE, "..", "..", "weekly_simulation", "input", "parameters.xlsx"))
DEFAULT_PARAMS = "2021"
OVER_TIME = ["beta_factor", "eps", "compliance_factor", "k", "severity"]   # order of Model.variant
SCALARS = ["xi", "gamma_a", "gamma_s", "gamma_m", "gamma_h", "gamma_s_to_h", "delta_m_factor",
           "q_factor", "omega", "omega_v", "eta_a", "eta_ps", "eta_pm", "eta_s", "eta_m", "eta_h",
           "b_S", "b_I", "deaths_outside", "deaths_hospital", "t_vacc"]
BY_AGE = ["p", "d", "IFR_percent", "compliance", "beta0"]


def read_params(name, ages, path=PARAMS_FILE):
    """The set name of the parameter file: (scalars dict, by-age dict of arrays, over-time dict
    parameter -> [(from_t, value), ...] increasing in from_t, from_t = -inf first)."""
    xs = pd.read_excel(path, sheet_name=None)
    for df in xs.values():
        df.columns = [str(c) for c in df.columns]
    if name not in xs["scalars"].columns:
        sets = [c for c in xs["scalars"].columns if c not in ("parameter", "meaning")]
        raise ValueError(f"params={name}: not a set of {path}; sets: {', '.join(sets)}")
    sc = xs["scalars"].set_index("parameter")[name]
    miss = [p for p in SCALARS if p not in sc.index or pd.isna(sc[p])]
    if miss:
        raise ValueError(f"{path}, set {name}: scalars {miss} missing")
    scal = {p: float(sc[p]) for p in SCALARS}
    ba = xs["by_age"]
    byage = {}
    for p in BY_AGE:
        r = ba[ba.parameter == p]
        if list(r.age) != list(ages) or r[name].isna().any():
            raise ValueError(f"{path}, set {name}: {p} not given for the ages {ages}")
        byage[p] = r[name].to_numpy(float)
    ot = xs["over_time"]
    over = {}
    for p in OVER_TIME:
        r = ot[(ot.parameter == p) & ot[name].notna()]
        sw = sorted((-np.inf if str(f).strip() == "start" else float(f), float(v))
                    for f, v in zip(r.from_t, r[name]))
        if not sw or sw[0][0] != -np.inf or len({f for f, _ in sw}) != len(sw):
            raise ValueError(f"{path}, set {name}: {p} needs one 'start' value and distinct days")
        over[p] = sw
    return scal, byage, over

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


STATE_KEYS = ["variable", "location"]   # cells A1, B1 of the initial state file


def read_state(path, locs, ages):
    """The initial state file: CSV, 1 + 13 m rows and 2 + n columns. Row 1: "variable",
    "location", then the age groups 0-4, ..., 75+. Then 13 blocks of m rows, the compartments
    in the order NAMES (S, P, Es, Ep, V, Ia, Ips, Ipm, Is, Im, Ih, R, M), within a block the
    locations in FIPS order; column A the compartment, column B the location, then the n ages.
    The labels are checked. Returns the state vector (compartment, location, age)."""
    m, n = len(locs), len(ages)
    df = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    if df.shape != (1 + len(NAMES) * m, 2 + n):
        raise ValueError(f"{path}: {df.shape[0]} x {df.shape[1]}, expected {1 + len(NAMES) * m}"
                         f" x {2 + n} (a header row; columns variable, location, then {n} ages)")
    head = df.iloc[0].str.strip().tolist()
    if head != STATE_KEYS + list(ages):
        raise ValueError(f"{path}: row 1 is {head}, expected {STATE_KEYS + list(ages)}")
    want = [(nm, loc) for nm in NAMES for loc in locs]
    got = list(zip(df.iloc[1:, 0].str.strip(), df.iloc[1:, 1].str.strip()))
    for r, (g, w) in enumerate(zip(got, want)):
        if g != w:
            raise ValueError(f"{path}: row {r + 2} is {g[0]}, {g[1]}; expected {w[0]}, {w[1]}")
    X = df.iloc[1:, 2:].to_numpy(float)
    B = X.reshape(len(NAMES), m, n)
    print(f"{path}: {df.shape[0]} rows x {df.shape[1]} columns read (a header row, 2 label columns;"
          f" {len(NAMES)} blocks of {m} locations, {n} ages); total {B.sum():,.0f} people; negative entries"
          f" {int((X < 0).sum())}", flush=True)
    print("   " + ", ".join(f"{nm} {B[c].sum():,.0f}" for c, nm in enumerate(NAMES)), flush=True)
    return X.ravel()


def write_state(path, x, locs, ages):
    """The state vector x in the format of read_state."""
    m, n = len(locs), len(ages)
    X = np.asarray(x, float).reshape(len(NAMES) * m, n)
    with open(path, "w") as f:
        f.write(",".join(STATE_KEYS + list(ages)) + "\n")
        for r, row in enumerate(X):
            f.write(f"{NAMES[r // m]},{locs[r % m]}," + ",".join("%.17g" % v for v in row) + "\n")


class Model:
    """Parameters and data of DATA.tex, Section 1."""

    def __init__(self, initial=INITIAL, params=DEFAULT_PARAMS, params_file=PARAMS_FILE):
        """initial: the file of the state at T_START (default INITIAL; "mock" is the same), or
        "none" for no initial state. params: the name of a set of params_file (default
        PARAMS_FILE)."""
        pop = pd.read_csv(CENSUS_N)
        self.locs = list(dict.fromkeys(pop.state))
        self.ages = list(dict.fromkeys(pop.age))
        m, n = len(self.locs), len(self.ages)
        self.m, self.n = m, n
        if list(pop.state) != list(np.repeat(self.locs, n)) or list(pop.age) != self.ages * m:
            raise ValueError("N_51x16.csv not in the order (location, age) of the model")
        self.N = pop.N.to_numpy(float).reshape(m, n)
        Uf = pd.read_csv(PREM_U, index_col=0)
        D = pd.read_csv(os.path.join(DATA, "census", "d_51x51.csv"), index_col=0)
        Lf = pd.read_csv(os.path.join(DATA, "census", "L_51.csv"))
        sf = pd.read_csv(os.path.join(DATA, "oxcgrt", "s_51_daily.csv"), index_col=0)
        if (list(Uf.index) != self.ages or list(D.index) != self.locs or list(Lf.state) != self.locs
                or list(sf.columns) != self.locs):
            raise ValueError("locations or ages of an input file differ from the model's")
        self.params, self.params_file = str(params), os.path.abspath(params_file)
        sc, ba, over = read_params(self.params, self.ages, self.params_file)
        self.t_vacc = sc["t_vacc"]
        self.U = Uf.to_numpy(float)                                          # U[j, q]
        self.d = D.to_numpy(float)                                           # d[i, p], miles
        self.L = Lf.L.to_numpy(float)
        self.s_t = sf.index.to_numpy(float)
        self.s_val = sf.to_numpy(float)                                      # s_i(t), daily
        self.b = decay_rates(self.U, self.d)
        self.p = ba["p"]
        self.q = 1 - sc["q_factor"] * (1 - self.p)
        self.delta_s0 = sc["gamma_s_to_h"] * ba["d"]
        self.delta_m0 = sc["delta_m_factor"] * self.delta_s0
        sfr = ba["IFR_percent"] / 100 * (1 - self.p)                        # supplement, Section 2
        self.die_s0, self.die_h0 = sc["deaths_outside"] * sfr, sc["deaths_hospital"] * sfr  # 1 - theta
        self.c0 = ba["compliance"]
        self.beta0 = ba["beta0"]
        self.xi, self.omega, self.omega_v = sc["xi"], sc["omega"], sc["omega_v"]
        self.gamma_a, self.gamma_s, self.gamma_m, self.gamma_h = (sc[g] for g in
                                                                  ("gamma_a", "gamma_s", "gamma_m", "gamma_h"))
        self.eta = np.array([sc[f"eta_{x}"] for x in ("a", "ps", "pm", "s", "m", "h")])
        self.bS, self.bI = sc["b_S"], sc["b_I"]
        # the parameters that change over time: each its own switch days; var_t the union of
        # them, var_par[i] the values (beta factor, eps, compliance factor, k, severity) on
        # (var_t[i], var_t[i + 1]]
        self.var_t = sorted({f for sw in over.values() for f, _ in sw})
        self.var_par = [tuple(next(v for f, v in reversed(over[p]) if f <= t) for p in OVER_TIME)
                        for t in self.var_t]
        self.breaks = [t for t in self.var_t if np.isfinite(t)]
        self.k = self.var_par[0][3]                                          # at the start, for sero_initial
        self.delta_s, self.delta_m = self.delta_s0, self.delta_m0
        self.a = np.outer(self.L, self.U.sum(axis=0))                        # a_ij = L_i sum_l U_lj
        # the contact matrix C, formed by build_C_51x16.py from U, L, d, b and read from its file
        Cf = pd.read_csv(C_FILE, index_col=0, float_precision="round_trip")
        labels = [f"{s} {g}" for s in self.locs for g in self.ages]
        if list(Cf.index) != labels or list(Cf.columns) != labels:
            raise ValueError(f"rows or columns of {C_FILE} not in the order of the model")
        self.C = Cf.to_numpy(float)                                          # C[(i,j), (p,q)]
        self.chi = np.zeros((3, m, n))                                       # chi_1, chi_2, chi_3
        self.method = "DOP853"                                               # or "SDIRK4"
        self.x251, self.initial_path = None, None
        if initial not in (None, "none"):
            self.initial_path = INITIAL if initial == "mock" else input_path(initial)
            self.x251 = read_state(self.initial_path, self.locs, self.ages)
            g = self.x251.reshape(13, m, n).sum(axis=0)
            print(f"   sum of the 13 compartments vs the population N: max |sum - N| / N ="
                  f" {np.max(np.abs(g - self.N) / self.N):.2e}", flush=True)

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
        """Integrate from t0 to t1, restarting at the variant days in (t0, t1), by the method
        self.method: "DOP853" (explicit RK of order 8) or "SDIRK4" (implicit RK of order 4,
        matrix-free, sdirk.py). Returns (states at t_out (default [t1]), stats)."""
        t_out = [t1] if t_out is None else list(t_out)
        knots = [t0] + [b for b in self.breaks if t0 < b < t1] + [t1]
        x, out, stats = np.asarray(x0, float), {}, {"nfev": 0, "steps": []}
        if self.method == "SDIRK4":
            import sdirk
            h0 = None                     # after a variant day, start from the last accepted step
            for a, b in zip(knots[:-1], knots[1:]):
                te = [s for s in t_out if a < s <= b] or [b]
                if te[-1] != b:
                    te.append(b)
                ts, ys, st = sdirk.solve(self.f, self.jvp, a, b, x, rtol=rtol, atol=atol, t_eval=te,
                                         h0=h0)
                h0 = st["h_next"]
                stats["nfev"] += st["nfev"]
                stats["steps"].extend(st["steps"])
                x = ys[-1].copy()
                out.update({s: y for s, y in zip(ts, ys)})
            return [out.get(s) for s in t_out], stats
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
    t0, t1 = T_START, mod.t_vacc
    res = {}
    for rtol in (1e-6, 1e-8, 1e-10, 1e-12):
        tic = time.time()
        (x,), st = mod.solve(t0, t1, mod.x251, rtol=rtol, atol=rtol, keep_steps=True)
        h = np.array(st["steps"])
        res[rtol] = x
        print(f"rtol = atol = {rtol:.0e}: {len(h)} steps, h in [{h.min():.3g}, {h.max():.3g}],"
              f" nfev {st['nfev']}, {time.time() - tic:.1f} s", flush=True)
    xb = res[1e-12]
    for rtol, x in res.items():
        print(f"   |x({t1:g}; {rtol:.0e}) - x({t1:g}; 1e-12)|_inf / |x|_inf = "
              f"{np.max(np.abs(x - xb)) / np.max(np.abs(xb)):.2e}", flush=True)
    g0, g1 = group_sums(mod, mod.x251), group_sums(mod, xb)
    print(f"group sums: |sum(t0) - N| / N <= {np.max(np.abs(g0 - mod.N) / mod.N):.2e};"
          f" change on [{t0:g}, {t1:g}] <= {np.max(np.abs(g1 - g0) / mod.N):.2e} relative to N", flush=True)
    X = xb.reshape(13, mod.m, mod.n)
    print(f"min over groups at {t1:g}: " + "  ".join(f"{nm} {X[c].min():.3g}" for c, nm in enumerate(NAMES)), flush=True)
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
        print(f"t = {t:g}: ||f_x||_1 = {norm1:.4g} (bound on rho); {est}", flush=True)


def frame(mod, x):
    X = x.reshape(13, mod.m, mod.n)
    df = pd.DataFrame({"state": np.repeat(mod.locs, mod.n), "age": np.tile(mod.ages, mod.m)})
    for c, nm in enumerate(NAMES):
        df[nm] = X[c].ravel()
    return df


def do_solve(mod):
    tic = time.time()
    (x,), st = mod.solve(T_START, mod.t_vacc, mod.x251)
    print(f"solved [{T_START:g}, {mod.t_vacc:g}]: nfev {st['nfev']}, {time.time() - tic:.1f} s;"
          f" min compartment {x.min():.3g}", flush=True)
    path = STATE_VACC.format(t=mod.t_vacc, params=mod.params)
    frame(mod, x).to_csv(path, index=False, float_format="%.17g")
    print(f"wrote {path}", flush=True)


SIG_FLOOR = 1e-6      # persons: group sizes below this are raised to it (optimal strategy)
ELIG_MIN = 1.0        # persons: only groups with sigma_{k,p} >= ELIG_MIN may receive dose k
B_FLOOR = 1e-9        # persons per day: the optimizer needs b_{k,i} > 0
B_CAP = 0.99          # weekly doses of a location at most this fraction of its eligible people
CHI_MAX = -np.log(1 - B_CAP) / 7.0   # 0.658 per day: at most 99% of a group in a week, as in DATA.tex, Remark 2.1
NRAND = 0             # random starts of the weekly optimization besides uniform and left ev; > 0 only
                      # to test whether other starts reach a smaller rho (other local minima)


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
    """DATA.tex, Remark 2.1: chi =-ln(1 - min(D / X, 0.99)) / 7, for X = S, V, P."""
    chi = np.zeros((3, mod.m, mod.n))
    for k, comp in enumerate((S, V, P)):
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(X[comp] > 0, np.minimum(Dw[k] / X[comp], B_CAP), 0.0)
        chi[k] = -np.log1p(-frac) / 7.0
    return chi


def optimal_rates(mod, A, tau, X, Dw, nrand):
    """The weekly problem of notes/stratified_seir_ngm.tex at tau: minimize rho over chi with
    sum_{p in I_i} sigma_{k,p} chi_{k,p} = b_{k,i}; sigma_k = S, V, P at tau, b_{k,i} the doses
    of location i in the week (all ages) per day, from Dw (3 x m, sheet 1 of the dose file).
    Returns chi (3 x m x n) and a summary, with the wall time (s) and iterations of the steps for
    chi1, chi2, chi3, added up over all sweeps and starts."""
    A.reset_stats()
    A.set_model(*mod.ngm_context(tau))
    elig = [(X[comp] >= ELIG_MIN).ravel() for comp in (S, V, P)]
    A.set_eligible(elig)
    A.set_cap(CHI_MAX)                            # chi <= CHI_MAX as a constraint of the problem
    sig = [np.maximum(X[comp], SIG_FLOOR).ravel() for comp in (S, V, P)]
    b = []
    for k in range(3):
        se = np.where(elig[k], sig[k], 0.0).reshape(mod.m, mod.n).sum(axis=1)   # eligible people
        bk = np.clip(Dw[k] / 7.0, B_FLOOR, np.maximum(B_CAP * se / 7.0, B_FLOOR))
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
                 start=name, sweeps=sw,
                 **{f"t_chi{k + 1}": A.STATS["time"][k] for k in range(3)},
                 **{f"it_chi{k + 1}": A.STATS["iters"][k] for k in range(3)}))


def max_rss_mb():
    """Peak resident memory of this process so far, MB (ru_maxrss is in KB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def checkpoint_path(strategy, tag, outdir=None):
    """The checkpoint, in intermediate/ or in outdir."""
    d = outdir or os.path.join(HERE, "intermediate")
    return os.path.join(d, f"simulate_{strategy}__{tag}_checkpoint.npz")


def output_paths(strategy, tag, outdir=None):
    """The weekly totals and (optimal) the rho of each week, named by strategy and run_tag, in
    data/ or in outdir."""
    path = WEEKLY.replace(".csv", f"_{strategy}__{tag}.csv")
    if outdir:
        path = os.path.join(outdir, os.path.basename(path))
    return path, path.replace(".csv", "_rho.csv")


def do_simulate(mod, strategy="actual", nrand=0, resume=False, doses=DOSES, outdir=None):
    """Weekly vaccination from mod.t_vacc, with the initial state of mod and the dose file doses,
    for as many weeks as the dose file has. strategy "actual" (not run): the rates of DATA.tex,
    Remark 2.1, from the doses by location and age; "optimal": the doses of each location (all
    ages) allocated among its age groups by the optimization of notes/stratified_seir_ngm.tex.
    After every week the state, the week and the results so far are saved to the checkpoint
    simulate_<strategy>__<tag>_checkpoint.npz (in intermediate/, or outdir), tag =
    run_tag(initial, doses, params); with resume=True the run continues from it. The results go
    to data/, or outdir."""
    if strategy not in ("actual", "optimal"):
        raise ValueError(f"strategy {strategy!r}")
    tag = run_tag(mod.initial_path, doses, mod.params)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    print(f"initial state {mod.initial_path}\ndoses {doses}\nparameters {mod.params} of {mod.params_file}"
          f" (vaccination from t = {mod.t_vacc:g})\noutput tag {tag}\n"
          f"output directory {outdir or DATA}", flush=True)
    A = None
    if strategy == "optimal":
        import alternating as A
    from doses_51x16x16 import read_sheet          # reads a dose file, real or mock
    if strategy == "optimal":     # sheet 1: one row per location (all ages)
        keys, nw, val = read_sheet(doses, 0, 1)
        if [k[0] for k in keys] != mod.locs:
            raise ValueError(f"{doses}, sheet 1: states not in the order of the model")
        Dk = val.reshape(mod.m, nw, 3).transpose(2, 0, 1)                    # 3 x m x weeks
    else:                         # sheet 2: one row per location and age
        keys, nw, val = read_sheet(doses, 1, 2)
        if keys != [[s, a] for s in mod.locs for a in mod.ages]:
            raise ValueError(f"{doses}, sheet 2: (state, age) not in the order of the model")
        Dk = val.reshape(mod.m, mod.n, nw, 3).transpose(3, 0, 1, 2)         # 3 x m x n x weeks
    weeks = [mod.t_vacc + 7.0 * w for w in range(nw)]                        # Week w+1 starts at t
    tic = time.time()
    ckpt = checkpoint_path(strategy, tag, outdir)
    if resume and not os.path.exists(ckpt):
        raise FileNotFoundError(f"resume: no checkpoint {ckpt} for these inputs")
    if resume:
        z = np.load(ckpt, allow_pickle=True)
        w0, x = int(z["next_week"]), z["x"].copy()
        rows, opt = list(z["rows"]), list(z["opt"])
        if not np.isclose(float(z["t"]), weeks[w0]):
            raise ValueError(f"{ckpt}: t = {float(z['t'])} is not the start of week {w0 + 1}")
        print(f"resuming from {ckpt}: week {w0 + 1} (t = {weeks[w0]:g})", flush=True)
    else:
        (x,), _ = mod.solve(T_START, mod.t_vacc, mod.x251)
        w0, rows, opt = 0, [], []
    for w in range(w0, len(weeks)):
        tau = weeks[w]
        X = x.reshape(13, mod.m, mod.n)
        rows.append({"t": tau, **{nm: X[c].sum() for c, nm in enumerate(NAMES)}})
        Dw = Dk[..., w]                        # 3 x m (optimal) or 3 x m x n (actual)
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
        t_ode = time.time()
        (x,), _ = mod.solve(tau, tau + 7.0, x)
        t_ode = time.time() - t_ode
        if strategy == "optimal":
            opt[-1].update(t_ode=t_ode, max_rss_MB=max_rss_mb())
            print("   times: " + ", ".join(f"chi{k} {info[f't_chi{k}']:.1f} s ({info[f'it_chi{k}']} it)"
                                          for k in (1, 3, 2))
                  + f", ODE {t_ode:.1f} s; max memory {max_rss_mb():.0f} MB", flush=True)
        if w + 1 < len(weeks):                 # the state at the start of the next week
            tmp = ckpt + ".tmp.npz"
            np.savez(tmp, next_week=w + 1, t=weeks[w + 1], x=x,
                     rows=np.array(rows, dtype=object), opt=np.array(opt, dtype=object))
            os.replace(tmp, ckpt)
    mod.chi[:] = 0.0
    X = x.reshape(13, mod.m, mod.n)
    rows.append({"t": weeks[-1] + 7.0, **{nm: X[c].sum() for c, nm in enumerate(NAMES)}})
    out = pd.DataFrame(rows)
    path, path_rho = output_paths(strategy, tag, outdir)
    out.to_csv(path, index=False, float_format="%.6g")
    if opt:
        pd.DataFrame(opt).to_csv(path_rho, index=False, float_format="%.10g")
        print(f"wrote {path_rho}", flush=True)
    print(f"{strategy}: simulated [{T_START:g}, {weeks[-1] + 7:g}] with {len(weeks)} weeks of doses in "
          f"{time.time() - tic:.1f} s; at the end: V {out.V.iloc[-1] / 1e6:.1f} M, R {out.R.iloc[-1] / 1e6:.1f} M,"
          f" M (dead) {out.M.iloc[-1] / 1e3:.0f} K, hospitalized now {out.Ih.iloc[-1] / 1e3:.1f} K;"
          f" min compartment {x.min():.3g}", flush=True)
    if opt:
        tot = {key: sum(o.get(key, 0) for o in opt)
               for key in ("t_chi1", "t_chi3", "t_chi2", "it_chi1", "it_chi3", "it_chi2", "t_ode")}
        print(f"total times over {len(opt)} weeks: " + ", ".join(
            f"chi{k} {tot[f't_chi{k}']:.0f} s ({tot[f'it_chi{k}']} it)" for k in (1, 3, 2))
            + f", ODE {tot['t_ode']:.0f} s; max memory {max_rss_mb():.0f} MB", flush=True)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    opts = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)   # method=, initial=, doses=, params=, paramfile=
    argv = [a for a in sys.argv[1:] if "=" not in a]
    bad = set(opts) - {"method", "initial", "doses", "out", "params", "paramfile"}
    if bad:
        raise ValueError(f"unknown options {sorted(bad)}: method=, initial=, doses=, out=, params=,"
                         " paramfile=")
    mode = argv[0] if argv else "solve"
    mod = Model(initial=opts.get("initial", INITIAL), params=opts.get("params", DEFAULT_PARAMS),
                params_file=input_path(opts["paramfile"]) if "paramfile" in opts else PARAMS_FILE)
    doses = input_path(opts["doses"]) if "doses" in opts else DOSES
    mod.method = opts.get("method", "DOP853").upper()
    if mod.method not in ("DOP853", "SDIRK4"):
        raise ValueError(f"method {mod.method!r}: DOP853 or SDIRK4")
    print(f"N total {mod.N.sum():,.0f}; parameters {mod.params}; vaccination from t = {mod.t_vacc:g};"
          f" switch days " + ", ".join(f"{t:g}" for t in mod.breaks) + f"; ODE method {mod.method}", flush=True)
    if mode == "simulate":
        args = [a for a in argv[1:] if a != "resume"]
        resume = "resume" in argv[1:]
        strategy = args[0] if args else "optimal"   # actual: reference only, not run
        if len(args) > 1:
            raise ValueError(f"unexpected arguments {args[1:]}: simulate [actual|optimal] [resume]")
        do_simulate(mod, strategy, NRAND, resume, doses=doses,
                    outdir=os.path.abspath(opts["out"]) if "out" in opts else None)
    else:
        {"verify": do_verify, "solve": do_solve}[mode](mod)
