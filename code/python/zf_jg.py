"""Next generation matrix of the model of Feng, Day, Glasser (JTB 2026) and John's method.

NGM = diag(d(chi)) Ac, of order 2N, N = m n, as in notes/stratified_seir_ngm.tex.
The columns of Ac carry the two weights of that note, q_s (naive infected,
column a = 0) and q_m (partially immune infected, column a = 1),

    weights="paper"     (default)  q_s, q_m of Lemma 2.1,  rho(NGM) = 4.440677536267 at chi = 0;
    weights="notebook"             the single weight q0 of eigenvalues_H_ZF_JG.nb for both,
                                   which reproduces the notebook, rho(NGM) = 4.423068109190629.

The reduced form of Theorem 3.1 of the note,

    rho(NGM) = rho(A1 diag(g)),   A1 = diag(d_c) C diag(psi_I q_s),   g = y + r z,  r = q_m / q_s,

is given by a1_matrix, g_vector and ngm_reduced.

Index conventions, kept as in the notebook:

  i, p = 0 : m-1   locations
  j, q = 0 : n-1   age groups

  Each (i, p) block of NGM has size 2n x 2n.  Inside a block the rows and
  columns are the pairs (susceptible, vaccinated-or-boosted) of one age
  group, interleaved:  (Y_i0, Z_i0, Y_i1, Z_i1, ...).

  Global row index of (i, j, a) with a = 0 for Y and a = 1 for Z is
  i * 2n + 2 * j + a.
"""

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl

from zf_jg_params import default_params


def zero_chi(par):
    """The three vaccination rate matrices set to zero, shape (m, n) each."""
    shape = (par["m"], par["n"])
    return np.zeros(shape), np.zeros(shape), np.zeros(shape)


def dfe_fractions(chi1, chi2, chi3, par):
    """Disease free equilibrium sizes sdfe, vdfe, pdfe, shape (m, n) each.

    sdfe = mN / (1 + c1 (1 + c2 + c2 (1 + c3))),  c1 = chi1 / (omega_v + eps2 chi2),
    vdfe = c1 sdfe,  pdfe = c2 vdfe,  with c2 = eps2 chi2 / omega, c3 = eps3 chi3 / omega.
    """
    om, omv = par["omega"], par["omega_v"]
    e2, e3 = par["eps2"], par["eps3"]

    c1 = chi1 / (omv + e2 * chi2)
    c2 = (e2 * chi2) / om
    c3 = (e3 * chi3) / om

    sdfe = par["mN"] / (1.0 + c1 * (1.0 + c2 + c2 * (1.0 + c3)))
    vdfe = c1 * sdfe
    pdfe = c2 * vdfe
    return sdfe, vdfe, pdfe


def compute_yz(chi1, chi2, chi3, par):
    """Y and Z of the notebook, shape (m, n) each.

    Y_ij = psi_S[j] s[i] a_ij beta[j] sdfe_ij / N_ij
    Z_ij = psi_S[j] s[i] a_ij beta[j] ((1 - eps1) vdfe_ij + pdfe_ij) / N_ij
    """
    sdfe, vdfe, pdfe = dfe_fractions(chi1, chi2, chi3, par)
    pref = par["vS"][:, None] * par["ma"] * (par["vpsi_S"] * par["vbeta"])[None, :]
    y = pref * sdfe / par["mN"]
    z = pref * ((1.0 - par["eps1"]) * vdfe + pdfe) / par["mN"]
    return y, z


def q0_vector(par):
    """Q0[j], the age dependent factor collecting infectivity and sojourn times.

    Q0 = eta_a Pas Ta + eta_ps Pps Tp + eta_s Ps Ts, with Pas = p, Pps = Ps = 1 - p,
    Ta = 1 / gamma_a, Tp = 1 / xi, Ts = 1 / (gamma_s + delta_s).
    """
    p = par["vp"]
    ta = 1.0 / par["vgamma_a"]
    tp = 1.0 / par["xi"]
    ts = 1.0 / (par["vgamma_s"] + par["vdelta_s"])
    return (par["eta_a"] * p * ta
            + par["eta_ps"] * (1.0 - p) * tp
            + par["eta_s"] * (1.0 - p) * ts)


def q_vectors(par):
    """q_s[j], q_m[j] of Lemma 2.1 of notes/stratified_seir_ngm.tex.

    q_s = eta_a p/gamma_a + eta_ps (1-p)/xi + eta_s (1-p)/(gamma_s+delta_s)
          + eta_h (1-p) delta_s/(gamma_s+delta_s)/gamma_h,
    q_m = eta_a q/gamma_a + eta_pm (1-q)/xi + eta_m (1-q)/(gamma_m+delta_m)
          + eta_h (1-q) delta_m/(gamma_m+delta_m)/gamma_h.
    """
    p, q = par["vp"], par["vq"]
    ga, gs, gm, gh = par["vgamma_a"], par["vgamma_s"], par["vgamma_m"], par["vgamma_h"]
    ds, dm = par["vdelta_s"], par["vdelta_m"]
    qs = (par["eta_a"] * p / ga + par["eta_ps"] * (1.0 - p) / par["xi"]
          + par["eta_s"] * (1.0 - p) / (gs + ds) + par["eta_h"] * (1.0 - p) * ds / (gs + ds) / gh)
    qm = (par["eta_a"] * q / ga + par["eta_pm"] * (1.0 - q) / par["xi"]
          + par["eta_m"] * (1.0 - q) / (gm + dm) + par["eta_h"] * (1.0 - q) * dm / (gm + dm) / gh)
    return qs, qm


def column_weights(par, weights="paper"):
    """The two column weights per age group, (q_s, q_m) or (q0, q0)."""
    if weights == "paper":
        return q_vectors(par)
    if weights == "notebook":
        q0 = q0_vector(par)
        return q0, q0
    raise ValueError("weights must be 'paper' or 'notebook'")


def q_matrix(par, weights="paper"):
    """Q = diag(q_s[0], q_m[0], q_s[1], q_m[1], ...), size 2n."""
    qs, qm = column_weights(par, weights)
    return sps.diags(np.column_stack([qs, qm]).ravel())


def psi_i_matrix(p, par):
    """Psi_I of location p, size 2n: diag(NPII[p,0], NPII[p,0], NPII[p,1], ...).

    The notebook concatenates NPII[p] with itself instead; both are the
    identity for the baseline NPII = 1.
    """
    return sps.diags(np.repeat(par["NPII"][p], 2))


def cc_block(i, p, par):
    """ccMat[i, p], size 2n by 2n: each contact entry doubled in row and column."""
    return sps.csr_matrix(np.kron(par["mc"][i, :, p, :], np.ones((2, 2))))


def ac_matrix(par, weights="paper"):
    """The chi free factor, blocks ccMat[i, p] . Psi_I[p] . Q, size 2 m n."""
    m = par["m"]
    q = q_matrix(par, weights)
    blocks = [[cc_block(i, p, par) @ psi_i_matrix(p, par) @ q
               for p in range(m)] for i in range(m)]
    return sps.bmat(blocks, format="csr")


def omega_matrix(chi1, chi2, chi3, par):
    """Omega = diag(Y_i0, Z_i0, Y_i1, Z_i1, ...) over all locations, size 2 m n."""
    y, z = compute_yz(chi1, chi2, chi3, par)
    diag = np.empty(2 * par["m"] * par["n"])
    diag[0::2] = y.ravel()
    diag[1::2] = z.ravel()
    return sps.diags(diag)


def _chi_or_zero(chi1, chi2, chi3, par):
    z1, z2, z3 = zero_chi(par)
    return (z1 if chi1 is None else np.asarray(chi1, dtype=float),
            z2 if chi2 is None else np.asarray(chi2, dtype=float),
            z3 if chi3 is None else np.asarray(chi3, dtype=float))


def ngm_blocks(chi1=None, chi2=None, chi3=None, par=None, weights="paper"):
    """NGM assembled block by block, the way the notebook does it:

        mH[i, p] = Omega[i] . ccMat[i, p] . Psi_I[p] . Q

    with Omega[i] = diag(Y_i0, Z_i0, Y_i1, Z_i1, ...) of size 2n.  Omitted chi
    default to zero, which is the state of the notebook.
    """
    if par is None:
        par = default_params()
    m, n = par["m"], par["n"]
    chi1, chi2, chi3 = _chi_or_zero(chi1, chi2, chi3, par)
    y, z = compute_yz(chi1, chi2, chi3, par)
    q = q_matrix(par, weights)

    blocks = []
    for i in range(m):
        d = np.empty(2 * n)
        d[0::2] = y[i]
        d[1::2] = z[i]
        om_i = sps.diags(d)
        blocks.append([om_i @ cc_block(i, p, par) @ psi_i_matrix(p, par) @ q
                       for p in range(m)])
    return sps.bmat(blocks, format="csr")


def ngm_factored(chi1=None, chi2=None, chi3=None, par=None, weights="paper"):
    """NGM as the product D(chi) . Ac of the diagonal and the chi free factor.

    The same matrix as ngm_blocks: D(chi) is diagonal, so the left product only
    scales row k of Ac by d_k, which is the block product of the notebook.
    """
    if par is None:
        par = default_params()
    chi1, chi2, chi3 = _chi_or_zero(chi1, chi2, chi3, par)
    return (omega_matrix(chi1, chi2, chi3, par) @ ac_matrix(par, weights)).tocsr()


def ngm_reversed(chi1=None, chi2=None, chi3=None, par=None, weights="paper"):
    """The reversed product Ac . D(chi).

    A genuinely different matrix from D(chi) . Ac, with the same characteristic
    polynomial: for square A and B the products AB and BA have the same
    eigenvalues with the same multiplicities, singular or not.  So rho is the
    same, which is the F V^-1 versus V^-1 F argument of the write-up.
    """
    if par is None:
        par = default_params()
    chi1, chi2, chi3 = _chi_or_zero(chi1, chi2, chi3, par)
    return (ac_matrix(par, weights) @ omega_matrix(chi1, chi2, chi3, par)).tocsr()


# the factored form is the default
build_ngm = ngm_factored


# ---------------------------------------------------------------- reduced form
# Theorem 3.1 of notes/stratified_seir_ngm.tex:  rho(NGM) = rho(A1 diag(g)), order N.

def dc_vector(par):
    """d_c, flattened over p = i n + j: the chi free factor psi_S[j] s[i] a_ij beta[j]."""
    return (par["vS"][:, None] * par["ma"] * (par["vpsi_S"] * par["vbeta"])[None, :]).ravel()


def r_vector(par, weights="paper"):
    """r_p = q_m / q_s of age group j, flattened over p = i n + j."""
    qs, qm = column_weights(par, weights)
    return np.tile(qm / qs, par["m"])


def a1_matrix(par, weights="paper"):
    """A1 = diag(d_c) C diag(psi_I q_s), order N = m n, independent of chi."""
    m, n = par["m"], par["n"]
    qs, _ = column_weights(par, weights)
    c = par["mc"].reshape(m * n, m * n)                  # row (i, j), column (p, q)
    col = par["NPII"].ravel() * np.tile(qs, m)
    return (sps.diags(dc_vector(par)) @ sps.csr_matrix(c) @ sps.diags(col)).tocsr()


def g_vector(chi1=None, chi2=None, chi3=None, par=None, weights="paper"):
    """g = y + r z, flattened over p = i n + j, with y, z the fractions of the note.

    y and z are Y and Z of compute_yz without the factor d_c.
    """
    if par is None:
        par = default_params()
    chi1, chi2, chi3 = _chi_or_zero(chi1, chi2, chi3, par)
    sdfe, vdfe, pdfe = dfe_fractions(chi1, chi2, chi3, par)
    y = (sdfe / par["mN"]).ravel()
    z = (((1.0 - par["eps1"]) * vdfe + pdfe) / par["mN"]).ravel()
    return y + r_vector(par, weights) * z


def ngm_reduced(chi1=None, chi2=None, chi3=None, par=None, weights="paper"):
    """A1 diag(g), order N, with the non-zero eigenvalues of the NGM."""
    if par is None:
        par = default_params()
    return (a1_matrix(par, weights) @ sps.diags(g_vector(chi1, chi2, chi3, par, weights))).tocsr()


def spectrum(a, k=None, tol=0.0):
    """The k eigenvalues of largest modulus, by Arnoldi, in a fixed order.

    k defaults to the largest scipy allows, N - 2.  The ordering is by
    decreasing modulus and then by real and imaginary part, so that two nearly
    equal matrices give two element-wise comparable lists.
    """
    nn = a.shape[0]
    k = nn - 2 if k is None else min(k, nn - 2)
    lam = spsl.eigs(a, k=k, which="LM", return_eigenvectors=False, tol=tol)
    return lam[np.lexsort((lam.imag, lam.real, -np.abs(lam)))]


def dominant(a, transpose=False, tol=0.0):
    """Dominant eigenvalue and eigenvector by Arnoldi.

    Returns (rho, v) with v scaled to unit Euclidean norm and made non-negative,
    matching Abs[Eigenvectors[..., 1]] of the notebook.
    """
    mat = a.T.tocsr() if transpose else a
    lam, vec = spsl.eigs(mat, k=1, which="LM", tol=tol)
    rho = float(np.real(lam[0]))
    v = np.abs(np.real(vec[:, 0]))
    return rho, v / np.linalg.norm(v)


def john_method(ngm, par):
    """Reproduce the cells under "John's method".

    Returns (rho, l0) where l0 has shape (m, n): the susceptible components of
    the dominant left eigenvector, per location and age group.  These are the
    four curves of the figure in the notebook.
    """
    rho, _ = dominant(ngm)
    _, w = dominant(ngm, transpose=True)
    t = w.reshape(par["m"], 2 * par["n"])
    return rho, t[:, 0::2].copy()


if __name__ == "__main__":
    par = default_params()
    direct = ngm_blocks(par=par)
    fact = ngm_factored(par=par)
    rev = ngm_reversed(par=par)
    rho, l0 = john_method(direct, par)
    sf, sr = spectrum(fact), spectrum(rev)

    np.set_printoptions(precision=16, linewidth=np.inf)
    print("shape(NGM)            =", direct.shape)
    print("rho                   =", repr(rho))
    print("D(chi).Ac vs blocks   = %.3e  (entrywise, relative; same matrix)"
          % (abs(direct - fact).max() / abs(direct).max()))
    print("Ac.D(chi) vs D(chi).Ac= %.2f       (entrywise, relative; different"
          " matrices)" % (abs(fact - rev).max() / abs(fact).max()))
    print("  same spectrum       = %.3e  (max |dlambda| / rho, %d eigenvalues)"
          % (np.max(np.abs(sf - sr)) / rho, len(sf)))
    print("l0                    =")
    print(l0)

    # reduced form, and the notebook's single weight, at chi = 0 and at a random chi
    rng = np.random.default_rng(0)
    shape = (par["m"], par["n"])
    chis = [(None, None, None),
            (0.3 * rng.random(shape), 0.02 * rng.random(shape), 0.02 * rng.random(shape))]
    for label, chi in zip(("chi = 0", "random chi"), chis):
        r_ngm = dominant(ngm_factored(*chi, par=par))[0]
        r_red = dominant(ngm_reduced(*chi, par=par))[0]
        r_nb = dominant(ngm_factored(*chi, par=par, weights="notebook"))[0]
        print("%-10s  rho(NGM) = %.12f   rho(A1 diag(g)) = %.12f   (rel. diff %.1e)"
              "   single weight q0: %.12f"
              % (label, r_ngm, r_red, abs(r_red - r_ngm) / r_ngm, r_nb))
