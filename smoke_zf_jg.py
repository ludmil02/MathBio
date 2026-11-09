"""Smoke test: zf_jg.py against eigenvalues_H_ZF_JG.nb for various chi.

Per case the notebook is re-evaluated in a Wolfram kernel with that chi
substituted, and three things are checked:

  NGM vs nb   the block assembly ngm_blocks(), the way the notebook builds
              NGM, against the notebook's own Y, Z and NGM, entrywise
  D.Ac-NGM    ngm_factored() = D(chi) . Ac against that block assembly,
              entrywise.  D(chi) is diagonal, so this is the same matrix
  rho, l0     John's method against the notebook
  Ac.D spec   the reversed product Ac . D(chi) against D(chi) . Ac through
              Arnoldi, the 30 eigenvalues of largest modulus, scaled by rho.
              AB and BA share a characteristic polynomial, so these agree
              although the matrices do not
  Ac.D-D.Ac   how far apart the two products are entrywise, relative, printed
              to show the spectral agreement above is not an identity.  It is
              of order one and is not part of the pass criterion

chi = 0 is one of the cases, the rest are structured or random draws.

Usage:
    python3 -u smoke_zf_jg.py [--seed S] [--random K] [--tol T]

Needs wolframscript on the path.  Exits non-zero if any case is off.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

from zf_jg import (compute_yz, john_method, ngm_blocks, ngm_factored,
                    ngm_reversed, spectrum)
from zf_jg_params import default_params

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK = os.path.join(HERE, "eigenvalues_H_ZF_JG.nb")
SETTINGS = os.path.join(HERE, "settings_zf_jg.wl")     # loaded by the notebook

# chi values that appear in eigenvalues_H_ZF5_v2.nb, kept as a non-random case
CHI1_ZF = np.array([[0.25, 0.50, 0.16, 0.12],
                    [0.14, 0.27, 0.095, 0.07],
                    [0.17, 0.35, 0.12, 0.09],
                    [0.16, 0.31, 0.11, 0.08]])
CHI2_ZF = np.array([[0.002, 0.007, 0.0006, 0.0003],
                    [0.003, 0.012, 0.001, 0.0007],
                    [0.006, 0.024, 0.002, 0.001],
                    [0.015, 0.054, 0.005, 0.003]])

WLS = r"""
settings = Import[%(setfile)s, "HeldExpressions"];
nb = Get[%(nbfile)s];
cs = Cases[nb, Cell[c_, "Input", ___] :> c, Infinity];
(* the notebook loads settings_zf_jg.wl, done below with chi substituted, and
   exports NGM.csv; keep the working directory clean *)
cs = DeleteCases[cs, _?(! FreeQ[#, s_String /; StringContainsQ[s, "Export" | "settings_zf_jg"]] &)];
cases = Import[%(infile)s, "RawJSON"];
out = Table[
   Module[{c = cases[[t]]},
    With[{x1 = c["chi1"], x2 = c["chi2"], x3 = c["chi3"]},
     Scan[ReleaseHold[# /. {HoldPattern[m\[Chi]1 = _] :> (m\[Chi]1 = x1),
                            HoldPattern[m\[Chi]2 = _] :> (m\[Chi]2 = x2),
                            HoldPattern[m\[Chi]3 = _] :> (m\[Chi]3 = x3)}] &, settings]];
    Do[ToExpression[cs[[i]]], {i, Length[cs]}];
    <|"name" -> c["name"], "Y" -> Y, "Z" -> Z, "NGM" -> NGM,
      "rho" -> Re[First[Eigenvalues[NGM, 1]]], "l0" -> l0|>],
   {t, Length[cases]}];
Export[%(outfile)s, out, "JSON", "Compact" -> True];
Print["cases: ", Length[out]];
"""


def make_cases(seed, n_random):
    """Structured cases followed by n_random random draws."""
    par = default_params()
    shape = (par["m"], par["n"])
    zero = np.zeros(shape)
    cases = [
        ("chi = 0", zero, zero, zero),
        ("ZF chi1,chi2; chi3 = 0", CHI1_ZF, CHI2_ZF, zero),
        ("ZF chi1,chi2; chi3 = chi2/2", CHI1_ZF, CHI2_ZF, 0.5 * CHI2_ZF),
        ("chi1 only", CHI1_ZF, zero, zero),
        ("chi2, chi3 only", zero, CHI2_ZF, CHI2_ZF),
        ("large chi1", 50.0 * np.ones(shape), zero, zero),
    ]
    rng = np.random.default_rng(seed)
    for r in range(n_random):
        scale = 10.0 ** rng.uniform(-3, 1)       # spread over four decades
        c1 = scale * rng.random(shape)
        c2 = scale * 0.05 * rng.random(shape)
        c3 = scale * 0.05 * rng.random(shape)
        cases.append(("random %d (scale %.3g)" % (r + 1, scale), c1, c2, c3))
    return [{"name": nm, "chi1": a.tolist(), "chi2": b.tolist(), "chi3": c.tolist()}
            for nm, a, b, c in cases]


def wolfram_reference(cases, nb=NOTEBOOK):
    """Evaluate the notebook once per case in a single kernel session."""
    with tempfile.TemporaryDirectory() as tmp:
        fin = os.path.join(tmp, "cases.json")
        fout = os.path.join(tmp, "ref.json")
        with open(fin, "w") as fh:
            json.dump(cases, fh)
        code = WLS % {"setfile": json.dumps(SETTINGS),
                      "nbfile": json.dumps(nb),
                      "infile": json.dumps(fin),
                      "outfile": json.dumps(fout)}
        res = subprocess.run(["wolframscript", "-code", code],
                             capture_output=True, text=True)
        if not os.path.exists(fout):
            sys.stderr.write(res.stdout + res.stderr)
            raise RuntimeError("wolframscript produced no output")
        with open(fout) as fh:
            return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--random", type=int, default=8, help="number of random chi")
    ap.add_argument("--tol", type=float, default=1e-11,
                    help="relative tolerance on NGM, rho and l0")
    args = ap.parse_args()

    par = default_params()
    cases = make_cases(args.seed, args.random)
    print("evaluating %d cases in the notebook ..." % len(cases))
    ref = wolfram_reference(cases)

    hdr = "%-28s %10s %10s %10s %10s %10s %8s %9s" % (
        "case", "NGM vs nb", "D.Ac-NGM", "rho vs nb", "l0 vs nb",
        "Ac.D spec", "Ac.D-D.Ac", "rho")
    print(hdr)
    print("-" * len(hdr))
    worst = 0.0
    bad = []
    for case, r in zip(cases, ref):
        chi = [np.array(case[k]) for k in ("chi1", "chi2", "chi3")]
        y, z = compute_yz(*chi, par)
        direct = ngm_blocks(*chi, par=par)          # as the notebook builds it
        fact = ngm_factored(*chi, par=par)          # as D(chi) . Ac
        rev = ngm_reversed(*chi, par=par)           # as Ac . D(chi)
        rho, l0 = john_method(direct, par)

        # the block assembly against the notebook
        e_yz = max(rel(y, np.array(r["Y"])), rel(z, np.array(r["Z"])))
        e_ngm = max(rel(np.asarray(direct.todense()), np.array(r["NGM"])), e_yz)
        # D(chi) . Ac against the block assembly; D(chi) is diagonal, so this
        # is the same matrix and the comparison is entrywise
        e_fac = abs(direct - fact).max() / abs(direct).max()
        # Ac . D(chi) is a different matrix, and must have the same spectrum
        e_spec = np.max(np.abs(spectrum(fact) - spectrum(rev))) / abs(rho)
        d_rev = abs(fact - rev).max() / abs(fact).max()   # how different, O(1)
        # John's method against the notebook
        e_rho = abs(rho - r["rho"]) / abs(r["rho"])
        e_l0 = rel(l0, np.array(r["l0"]))

        errs = (e_ngm, e_fac, e_spec, e_rho, e_l0)   # d_rev is informational
        worst = max(worst, *errs)
        if max(errs) > args.tol:
            bad.append(case["name"])
        print("%-28s %10.3e %10.3e %10.3e %10.3e %10.3e %8.2f %9.6f"
              % (case["name"], e_ngm, e_fac, e_rho, e_l0, e_spec, d_rev, rho))

    print("-" * len(hdr))
    print("worst relative error: %.3e   tolerance: %.1e" % (worst, args.tol))
    if bad:
        print("FAILED: " + ", ".join(bad))
        return 1
    print("OK: all %d cases reproduce the notebook" % len(cases))
    return 0


def rel(a, b):
    """Relative error in the max norm, scaled by the reference."""
    s = np.max(np.abs(b))
    return np.max(np.abs(a - b)) / (s if s > 0 else 1.0)


if __name__ == "__main__":
    sys.exit(main())
