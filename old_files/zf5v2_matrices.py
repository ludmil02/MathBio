"""Matrices and vectors of eigenvalues_H_ZF5_v2.nb, formed in Python.

Same construction and the same names as the notebook: the disease-free
equilibrium (sdfe, vdfe, pdfe and the mN-free sdfe1, vdfe1, pdfe1), the
transmission coefficients Y, Z, Y1, Z1, DYZ, the three factors of H
(ccMat, psiImat, Q), the diagonal blocks Omega, Omega1, DYZ1, and the
32x32 next-generation matrix NGM = Join[L1, L2, L3, L4].

The factorization the optimization rests on is

    NGM = diag(Y1, Z1 interleaved) . diag(DYZ duplicated) . Aconst

with Aconst = Join over i of [ccMat[i,p] . psiImat[p] . Q]_p, the notebook's
noChi1/Aconst2 -- free of chi1, chi2, chi3.  Only the leading diagonal
depends on the vaccination rates, which is what SLSQP varies.

Against the CSVs the optimizer reads, build()[key] gives

    'Aconst'  ccMat . psiImat . Q            ZF5_v2 Export["Aconst.csv", Aconst2]
    'mHz1'    DYZ1 . ccMat . psiImat . Q     Aconst.dat, new_nb_Data/Ac.csv
    'Omega1'  diag(Y1, Z1) per region        new_nb_Data/DYZ.csv

so the "Ac" that x_math_bio.py reads is mHz1, not Aconst2: the ma, psiS,
beta diagonal is already folded in, and get_constA() in y_math_bio.py folds
it into Aconst2 to get the same matrix.

All data are the literals of the notebook, x = 1 (daily rates), psiS = S =
psiI = 1.  Note that params_bio1() in functions_bio.py sets x = 7 instead,
which changes omega = omega_v = x/182.5 and hence D(chi).

Run as a script to reproduce the notebook's checks and its three
reproduction numbers.
"""

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as sps_l

M = 4  # regions
N = 4  # age groups
REGIONS = ("Northeast", "Midwest", "South", "West")
AGES = ("0-19", "20-39", "40-59", "60+")

# ---------------------------------------------------------------- data ----
# mN: population by region x age
MN = np.array([[12861646, 14862307, 14294150, 13831766],
               [17039947, 18046593, 16847952, 16382252],
               [31775045, 34108261, 31760738, 29018710],
               [19577717, 22475544, 19374280, 17227215]], dtype=float)

# ma: activity/mixing amplitudes
MA = np.array([[17.073281356320802, 17.011021886542494, 14.464215476578978, 3.6569593567943652],
               [13.676777723325813, 13.626903952047815, 11.586751010942738, 2.929455634349783],
               [15.06551341188455, 15.010575473629661, 12.763266047514328, 3.2269116338380703],
               [13.603540223639197, 13.553933520407654, 11.524705352915733, 2.913768751778259]])

VBETA = np.array([0.0794498883424254, 0.07378658226273424,
                  0.08093055958811815, 0.21470494228237624])

VP = np.array([0.9706574350846233, 0.8938517556772458,
               0.6548278086229462, 0.28889590656079006])

VQ = np.array([0.9853287175423117, 0.9469258778386229,
               0.8274139043114731, 0.6444479532803951])

VDELTA_S = np.array([0.0005227159753841133, 0.010632200553688746,
                     0.02953817543239884, 0.057703228431405976])

VDELTA_M = np.array([0.00026135798769205665, 0.005316100276844373,
                     0.01476908771619942, 0.028851614215702988])

# vη = {ηa, ηps, ηpm, ηs, ηm, ηh} -- the notebook's own comment warns that
# this order is not the natural one.
ETA_A, ETA_PS, ETA_PM, ETA_S, ETA_M, ETA_H = 0.5, 1.25, 1.0, 1.0, 0.75, 0.1

# mc[i,j,p,q]: contact tensor, region x age x region x age
MC = np.array([
    # mc[1,1,p,q]
    0.20197824889699842, 0.059415283664697484, 0.04892172067444938, 0.008030707230540838,
    0.1521469690453322, 0.044793777860070695, 0.03339288184769395, 0.002877214165782988,
    0.16553227488415234, 0.04874272018491846, 0.03561633471701904, 0.0026950721853331765,
    0.13265108985915125, 0.03912354820446772, 0.023571691514938764, 0.0005104650644532836,
    # mc[1,2,p,q]
    0.06103198838432115, 0.1621666401078013, 0.08879328829009539, 0.012414165389758761,
    0.0459744160482341, 0.12225905533493119, 0.06060832987601685, 0.004447704478623026,
    0.050019068553821766, 0.13303720313296225, 0.06464391343488654, 0.004166142642932759,
    0.04008332491079227, 0.10678286747259674, 0.04278279609376618, 0.0007890958484599316,
    # mc[1,3,p,q]
    0.061021070464599016, 0.1080249810376071, 0.14362458579558354, 0.02178381591253095,
    0.04596619175476153, 0.08144111591294141, 0.09803495784237114, 0.00780463064199165,
    0.05001012071860447, 0.08862082445675072, 0.10456257978596326, 0.007310558668234078,
    0.04007615446566739, 0.07113187537338422, 0.06920186746625413, 0.0013846697027553945,
    # mc[1,4,p,q]
    0.0530544005552305, 0.07868141298325652, 0.11156873991077054, 0.14315896066513709,
    0.039965027338719886, 0.05931870585316376, 0.0761543481785637, 0.05129050004687597,
    0.043480996911706815, 0.0645481408191205, 0.0812250577010754, 0.048043556052264826,
    0.03484397005077645, 0.05180983517672048, 0.05375657036651684, 0.009099777390100771,
    # mc[2,1,p,q]
    0.1864572151389774, 0.05489509996649521, 0.04092321913380201, 0.003526046848494684,
    0.15883787710504257, 0.04672482099647865, 0.038472569688521994, 0.006315434929836045,
    0.16645777750802396, 0.048999330689790524, 0.037222141625251635, 0.0036235420810433495,
    0.1384701016999805, 0.04080592480578916, 0.02714997342339332, 0.0011189243590791157,
    # mc[2,2,p,q]
    0.05620517348314829, 0.14946555074840534, 0.074095594637493, 0.005437458989376001,
    0.04787967272668808, 0.12721993598921966, 0.06965844790907054, 0.00973892858108688,
    0.05017659550199961, 0.13341285383071985, 0.06739442242768076, 0.0055878047877789255,
    0.0417400639737951, 0.11110427030131684, 0.049157751217494744, 0.001725474894726343,
    # mc[2,3,p,q]
    0.05598955512300673, 0.09920012240767083, 0.11941240869771409, 0.009506504255907459,
    0.04769599325596215, 0.08443573224488508, 0.11226150612140896, 0.01702691757031642,
    0.049984104409647206, 0.08854596503673143, 0.10861280423287652, 0.009769359198844548,
    0.04157993771515215, 0.07373978257008608, 0.07922259761545583, 0.0030167095443343977,
    # mc[2,4,p,q]
    0.0483897647994067, 0.07182325186748455, 0.09220789380625656, 0.06210267824107499,
    0.041221936671215216, 0.0611334816575884, 0.08668610865372699, 0.11123091673257021,
    0.04319946909344056, 0.06410938811695327, 0.08386865341655675, 0.06381981794941248,
    0.035936049178990924, 0.053389359283444786, 0.06117411873395686, 0.019707111797920892,
    # mc[3,1,p,q]
    0.18556130844105437, 0.05464047987511577, 0.03992583124100672, 0.0030211698679518018,
    0.1522624964435324, 0.04482073788669957, 0.034047890652404864, 0.003314531597667506,
    0.1762963105157434, 0.05186051149336481, 0.04270122602847936, 0.007009586741681536,
    0.1370960269820946, 0.04041057478883919, 0.02613515273683758, 0.0008961647075264086,
    # mc[3,2,p,q]
    0.05599820943381795, 0.14894010182362313, 0.07237126775468089, 0.004664151792412439,
    0.04594938048450658, 0.12217325469894243, 0.06171666147695607, 0.005117050403640263,
    0.053202242437334624, 0.1413624089704349, 0.07740206694021748, 0.010821561843342861,
    0.041372482744302365, 0.11015194481375509, 0.04737369461664491, 0.001383519765388032,
    # mc[3,3,p,q]
    0.055880628953818706, 0.09902370436004378, 0.11683680502094429, 0.008168718861595085,
    0.045852899717216664, 0.0812276083196121, 0.09963591584396908, 0.008961918052484497,
    0.05309053270114724, 0.09398563061336453, 0.12495857104461666, 0.018952705717133245,
    0.04128561217417128, 0.07323516960420862, 0.07648050521663968, 0.002423073799034595,
    # mc[3,4,p,q]
    0.04852266901536028, 0.07203257273247518, 0.09064319750041606, 0.05361426218857699,
    0.03981531915490311, 0.05908720181678404, 0.07729857039792182, 0.058820315929519154,
    0.046099952601390444, 0.06836773898448949, 0.09694414728763369, 0.12439347598050467,
    0.03584941923755752, 0.053273281535798744, 0.05933436418464847, 0.01590351145202035,
    # mc[4,1,p,q]
    0.17692165561991846, 0.05218052056270084, 0.03143843516862934, 0.0006808261012789027,
    0.15069897386425823, 0.044409666204609816, 0.029547700804121193, 0.001217741235651662,
    0.163113647303639, 0.04807955699770153, 0.031094993630267853, 0.0010662358147586787,
    0.1710191212768521, 0.05030813792198626, 0.04142302315613488, 0.0067997643374913255,
    # mc[4,2,p,q]
    0.05402593029206072, 0.1439262777551564, 0.057664386988987765, 0.0010635753744862712,
    0.046018404183184, 0.12249241449207234, 0.05419640146414941, 0.0019023353956358151,
    0.049809428405089215, 0.1326148455433863, 0.057034446418792614, 0.001665656110778397,
    0.05222349458769408, 0.13876180141215785, 0.07597812119836828, 0.010622480378000503,
    # mc[4,3,p,q]
    0.054942625759409065, 0.0975186382104811, 0.0948726831888986, 0.0018983205922129109,
    0.04679923039573194, 0.08299591734527946, 0.08916695892507336, 0.0033953799057967727,
    0.050654579557569544, 0.08985446817352293, 0.09383626963688473, 0.002972943310353709,
    0.05310960687708606, 0.09401939743322546, 0.12500346571129078, 0.018959514977183435,
    # mc[4,4,p,q]
    0.05103757114600094, 0.07588825685021303, 0.07873972973366654, 0.01332886394068318,
    0.04347296871757569, 0.06458678677837276, 0.07400425508103237, 0.02384031284122476,
    0.04705429841233032, 0.06992405846740708, 0.07787955670772401, 0.020874217479185664,
    0.0493348343305421, 0.07316517449633562, 0.10374681915848274, 0.13312229585922322,
]).reshape(M, N, M, N)

# The three dose rates.  The notebook enters the coverage matrices and then
# overwrites all three with zeros, so its baseline NGM is the unvaccinated one.
CHI1_COVERAGE = np.array([[0.25, 0.50, 0.16, 0.12],
                          [0.14, 0.27, 0.095, 0.07],
                          [0.17, 0.35, 0.12, 0.09],
                          [0.16, 0.31, 0.11, 0.08]])
CHI2_COVERAGE = np.array([[0.002, 0.007, 0.0006, 0.0003],
                          [0.003, 0.012, 0.001, 0.0007],
                          [0.006, 0.024, 0.002, 0.001],
                          [0.015, 0.054, 0.005, 0.003]])
CHI3_COVERAGE = np.zeros((M, N))

ZERO_CHI = (np.zeros((M, N)), np.zeros((M, N)), np.zeros((M, N)))

# The stored optimum (carried at 18 digits in the notebook).
OPTIMAL_CHI1 = np.array(
    [[0.00300457816105132022, 0.006063747923999046954,
      0.003545425576831823678, 2.236626067229857952e-14],
     [0.004398722516730560125, 0.004449697286182037123,
      0.005958627241133407706, 6.642807744610886198e-11],
     [0.003537310173540902242, 0.004151002856664137187,
      0.003181218010565318565, 2.718994501058789159e-14],
     [0.006054774009199111705, 0.003829806663471072622,
      0.004735452868049963744, 4.59675690490171039e-11]])

# The per-group profile read off the left Perron eigenvector (figure f11)
# and the per-region scaling that turns it into a dose rate.
L0 = np.array([[0.22355562536934626, 0.24791487446658117,
                0.23178889192271754, 0.034510966529755153],
               [0.1815431132174088, 0.20127892880934503,
                0.19137158756814973, 0.028787888791755627],
               [0.19936130666301938, 0.22105692608163208,
                0.20955038315149774, 0.03228720752923144],
               [0.1685920236035752, 0.18702861840401064,
                0.15976616670663327, 0.015097661560599262]])
SCALE = np.array([0.01718580823153429, 0.024499694107403585,
                  0.016472195434267065, 0.027251035887713583])
NEWCHI1 = SCALE[:, None] * L0

# Susceptibles at t = 347 and the dose budget per region, for the constraint.
SSTAR = np.array([[1.0600216151660305e7, 1.1947824997470027e7,
                   1.200765016794735e7, 1.2488303055892894e7],
                  [1.4987652030268373e7, 1.6016533803076502e7,
                   1.5145225873499423e7, 1.5295304221225757e7],
                  [2.714573794622607e7, 2.9291708309119027e7,
                   2.7921743967612088e7, 2.680293855830742e7],
                  [1.7516265188547775e7, 2.023022880905246e7,
                   1.768703243117786e7, 1.6290434385275263e7]])
C2 = np.array([[279, 62294, 60028, 24269],
               [898, 92762, 96407, 37373],
               [1180, 115330, 133603, 56325],
               [749, 107797, 119699, 39046]], dtype=float)
TOTAL_DOSES = C2.sum(axis=1)


# ---------------------------------------------------------- parameters ----
def params_zf5v2(x=1.0):
    """Cell 3 of the notebook.  x is the time unit: 1 for days, 7 for weeks."""
    par = {}
    par['m'], par['n'], par['ntotal'] = M, N, M * N
    par['x'] = x
    par['kappa'] = x / 5.0
    par['xi'] = x / 2.0
    par['vgamma_a'] = np.full(N, 1.0 / 7.0) * x
    par['vgamma_s'] = np.full(N, 1.0 / 5.0) * x
    par['vgamma_m'] = np.full(N, 1.0 / 5.0) * x
    par['vgamma_h'] = np.full(N, 1.0 / 10.0) * x
    par['omega'] = x / 182.5
    par['omega_v'] = x / 182.5
    par['bS'] = par['bI'] = 0.31
    par['eps'] = np.array([0.8, 0.8, 0.8])
    par['beta'] = VBETA
    par['vp'], par['vq'] = VP, VQ
    par['vdelta_s'], par['vdelta_m'] = VDELTA_S, VDELTA_M
    par['mN'], par['ma'], par['mc'] = MN, MA, MC
    # vpsiS = vS = vpsiI = 1 in this notebook
    par['psiS'] = np.ones(N)
    par['psiI'] = np.ones(N)
    par['S'] = np.ones(M)
    par['NPII'] = np.ones((M, N))
    return par


# ------------------------------------------- disease-free equilibrium ----
def dfe(chi1, chi2, chi3, par):
    """sdfe, vdfe, pdfe and the mN-free sdfe1, vdfe1, pdfe1 (cells 17, 19)."""
    eps1, eps2, eps3 = par['eps']
    omega, omega_v = par['omega'], par['omega_v']
    p1 = chi1 / (omega_v + eps2 * chi2)
    p2 = eps2 * chi2 / omega
    p3 = 1.0 + eps3 * chi3 / omega
    sdfe1 = 1.0 / (1.0 + p1 * (1.0 + p2 + p2 * p3))
    vdfe1 = p1 * sdfe1
    pdfe1 = p2 * vdfe1
    mN = par['mN']
    return {'sdfe': mN * sdfe1, 'vdfe': mN * vdfe1, 'pdfe': mN * pdfe1,
            'sdfe1': sdfe1, 'vdfe1': vdfe1, 'pdfe1': pdfe1}


def transmission(chi1, chi2, chi3, par):
    """Y, Z and their factorization Y = DYZ*Y1, Z = DYZ*Z1 (cell 19)."""
    eps1 = par['eps'][0]
    d = dfe(chi1, chi2, chi3, par)
    mN = par['mN']
    pref = par['S'][:, None] * par['ma'] * (par['psiS'] * par['beta'])[None, :]
    Y = pref * (d['sdfe'] / mN)
    Z = pref * (((1.0 - eps1) * d['vdfe'] + d['pdfe']) / mN)
    Y1 = d['sdfe1']
    Z1 = (1.0 - eps1) * d['vdfe1'] + d['pdfe1']
    return Y, Z, Y1, Z1, pref


# ------------------------------------------------- the factors of H ------
def interleave(A, B):
    """diag entries {A[j], B[j]} alternating, as Flatten[Table[{a,b},...]]."""
    out = np.empty(A.size + B.size, dtype=float)
    out[0::2] = np.asarray(A).ravel()
    out[1::2] = np.asarray(B).ravel()
    return out


def q_matrix(par):
    """Q0 and the 2n x 2n diagonal Q (cell 26)."""
    Pas = par['vp']
    Pps = 1.0 - par['vp']
    Ps = 1.0 - par['vp']
    Ta = 1.0 / par['vgamma_a']
    Tp = 1.0 / par['xi']
    Ts = 1.0 / (par['vgamma_s'] + par['vdelta_s'])
    Q0 = ETA_A * Pas * Ta + ETA_PS * Pps * Tp + ETA_S * Ps * Ts
    return Q0, np.diag(interleave(Q0, Q0))


def cc_matrices(par):
    """ccM (mn x mn) and ccMat[i,p] (2n x 2n), cells 14 and 16."""
    ccM = par['mc'].reshape(M * N, M * N)
    ccMat = np.empty((M, M, 2 * N, 2 * N))
    for i in range(M):
        for p in range(M):
            blk = ccM[i * N:(i + 1) * N, p * N:(p + 1) * N]
            ccMat[i, p] = np.kron(blk, np.ones((2, 2)))
    return ccM, ccMat


def psi_i_matrices(par):
    """psiImat[p] = DiagonalMatrix[Flatten[{NPII[p], NPII[p]}]] (cell 13).

    Note the concatenation: unlike Omega and Q, this diagonal is not
    interleaved.  With NPII = 1 it is the identity either way.
    """
    NPII = par['NPII']
    return np.array([np.diag(np.concatenate((NPII[p], NPII[p])))
                     for p in range(M)])


def diag_blocks(A, B=None):
    """Omega[i] = diag{A[i,j], B[i,j]} interleaved over j (cells 22, 23)."""
    B = A if B is None else B
    return np.array([np.diag(interleave(A[i], B[i])) for i in range(M)])


def assemble(blocks):
    """Join[L1,...,Lm] with Li = the row of 2n x 2n blocks (cells 51, 53)."""
    return np.block([[blocks[i, p] for p in range(M)] for i in range(M)])


# ------------------------------------------------ H, NGM and Aconst ------
def build(chi1, chi2, chi3, par):
    """Everything the notebook forms, for one choice of the dose rates."""
    Y, Z, Y1, Z1, DYZ = transmission(chi1, chi2, chi3, par)
    Q0, Q = q_matrix(par)
    ccM, ccMat = cc_matrices(par)
    psiImat = psi_i_matrices(par)
    Omega = diag_blocks(Y, Z)
    Omega1 = diag_blocks(Y1, Z1)
    DYZ1 = diag_blocks(DYZ)

    mH = np.empty((M, M, 2 * N, 2 * N))
    mHz = np.empty_like(mH)
    noChi = np.empty_like(mH)   # DYZ1 . ccMat . psiImat . Q
    noChi1 = np.empty_like(mH)  # ccMat . psiImat . Q
    for i in range(M):
        for p in range(M):
            base = ccMat[i, p] @ psiImat[p] @ Q
            noChi1[i, p] = base
            noChi[i, p] = DYZ1[i] @ base
            mH[i, p] = Omega[i] @ base
            mHz[i, p] = DYZ1[i] @ Omega1[i] @ base

    return {'Y': Y, 'Z': Z, 'Y1': Y1, 'Z1': Z1, 'DYZ': DYZ,
            'Q0': Q0, 'Q': Q, 'ccM': ccM, 'ccMat': ccMat, 'psiImat': psiImat,
            'Omega': Omega, 'Omega1': Omega1, 'DYZ1': DYZ1,
            'mH': mH, 'mHz': mHz, 'noChi': noChi, 'noChi1': noChi1,
            'NGM': assemble(mH), 'mHz1': assemble(noChi),
            'Aconst': assemble(noChi1)}


def aconst(par):
    """The chi-free factor alone (the notebook's Aconst2, exported as Ac)."""
    Q0, Q = q_matrix(par)
    ccM, ccMat = cc_matrices(par)
    psiImat = psi_i_matrices(par)
    noChi1 = np.empty((M, M, 2 * N, 2 * N))
    for i in range(M):
        for p in range(M):
            noChi1[i, p] = ccMat[i, p] @ psiImat[p] @ Q
    return assemble(noChi1)


def dyz_diagonal(par):
    """diag(DYZ duplicated), the part of D that does not depend on chi."""
    DYZ = par['S'][:, None] * par['ma'] * (par['psiS'] * par['beta'])[None, :]
    d = DYZ.ravel()
    return sps.diags([interleave(d, d)], [0], (2 * M * N, 2 * M * N))


def d_chi(chi1, chi2, chi3, par):
    """diag(Y1, Z1 interleaved): the only chi-dependent factor of NGM."""
    _, _, Y1, Z1, _ = transmission(chi1, chi2, chi3, par)
    return sps.diags([interleave(Y1.ravel(), Z1.ravel())], [0],
                     (2 * M * N, 2 * M * N))


def ngm_from_factors(chi1, chi2, chi3, par, Ac=None):
    """NGM = D(chi) . diag(DYZ) . Aconst, without re-forming the blocks."""
    Ac = aconst(par) if Ac is None else Ac
    return d_chi(chi1, chi2, chi3, par) @ (dyz_diagonal(par) @ Ac)


def perron(A, k=1):
    """Largest eigenvalue and its eigenvector by Arnoldi."""
    e, v = sps_l.eigs(sps.csr_matrix(A), k=max(k, 2), which='LM', tol=0)
    q = np.argsort(np.abs(e))[::-1]
    return np.real(e[q[0]]), np.real(v[:, q[0]])


def reproduction_number(chi1, chi2, chi3, par):
    return perron(build(chi1, chi2, chi3, par)['NGM'])[0]


def doses(chi1, par=None):
    """Total[Transpose[chi1*Sstar]]: doses per region, against TOTAL_DOSES."""
    return (chi1 * SSTAR).sum(axis=1)


# ------------------------------------------------------------ checks -----
# What the notebook prints, for comparison.
REFERENCE = {'R0_chi0': 4.423068109190629,
             'R0_optimal': 2.9247935891448495,
             'R0_newchi': 2.935928081173093}

if __name__ == '__main__':
    np.set_printoptions(precision=16, linewidth=np.inf)
    par = params_zf5v2()

    out = build(*ZERO_CHI, par)
    Y, Z, Y1, Z1, DYZ = (out[k] for k in ('Y', 'Z', 'Y1', 'Z1', 'DYZ'))
    print('chky, chkz           =', np.linalg.norm(Y1 * DYZ - Y),
          np.linalg.norm(Z1 * DYZ - Z))
    print('||mHz - mH||         =', np.linalg.norm(out['mHz'] - out['mH']))
    print('||D(chi) Ac - NGM||  =', np.linalg.norm(
        ngm_from_factors(*ZERO_CHI, par, out['Aconst']) - out['NGM']))

    for name, chi1 in (('R0_chi0', np.zeros((M, N))),
                       ('R0_optimal', OPTIMAL_CHI1),
                       ('R0_newchi', NEWCHI1)):
        zero = np.zeros((M, N))
        r0 = reproduction_number(chi1, zero, zero, par)
        print('%-12s = %.13f   (notebook %.13f)'
              % (name, r0, REFERENCE[name]))

    print('doses(newChi1)       =', doses(NEWCHI1))
    print('TotalDoses           =', TOTAL_DOSES)
