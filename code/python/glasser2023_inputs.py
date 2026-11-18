"""Age-specific inputs from Glasser, Feng, Vo, Jones, Clarke 2023 (J. Theor. Biol. 556, 111296;
MathBio/data/Downloaded_from_source/papers/2023GlasserFengVoJonesClarke.pdf), written to
data/glasser2023/:

table2.csv  Table 2 (= Table S2 of the supplement of Feng, Day, Glasser 2026), typed from the
            article: Pr(asymp|inf) = p_j, Pr(hosp|inf) = d_j, IFR (percent), compliance c0_j.
beta0.csv   probabilities of infection on contact beta_j, 26 November 2020 (the values used for
            transmission modeling, p. 5), digitized from Fig. 4: the figure is the raster image 0
            of page 5 (pdfimages); the y axis is calibrated on its tick marks 0, 0.05, ..., 0.35,
            and beta_j is read at the horizontal bar of the "+" marker of the 26 Nov 2020 curve at
            the 16 age groups. Resolution 0.00057 per pixel; estimated accuracy +-0.003.

Usage:  python3 -u glasser2023_inputs.py
"""
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.normpath(os.path.join(HERE, "..", "..", "data", "Downloaded_from_source", "papers",
                                    "2023GlasserFengVoJonesClarke.pdf"))
DIR = os.path.join(HERE, "data", "glasser2023")
AGES = [f"{5 * k}-{5 * k + 4}" for k in range(15)] + ["75+"]

# Glasser et al. 2023, Table 2 (columns 4-7)
P_ASYMP = [0.981400373, 0.972064267, 0.969763501, 0.960291933, 0.941577238, 0.913456364,
           0.880594618, 0.839642887, 0.775783537, 0.708076043, 0.623043155, 0.521907521,
           0.421531351, 0.328351424, 0.2503706, 0.163464241]
P_HOSP = [1.0435e-05, 0.000143099, 0.00027027, 0.003637205, 0.006854725, 0.017709616,
          0.025217824, 0.035166793, 0.037368978, 0.051520404, 0.062486388, 0.083247801,
          0.090697794, 0.110144916, 0.118491978, 0.139755917]
IFR_PCT = [0.0008, 0.001312163, 0.002400084, 0.004390005, 0.00802978, 0.014687309,
           0.026864628, 0.049138219, 0.08987895, 0.164398026, 0.300701205, 0.550014019,
           1.006033295, 1.840140307, 3.365809369, 14.30030403]
COMPLIANCE = [0.706360069, 0.715211185, 0.71952257, 0.720175768, 0.71795013, 0.713574937,
              0.707771036, 0.701279533, 0.694876881, 0.689376822, 0.685620487, 0.684456022,
              0.686708224, 0.693136453, 0.704376296, 0.744380002]


def table2():
    df = pd.DataFrame({"age": AGES, "p": P_ASYMP, "d": P_HOSP, "IFR_percent": IFR_PCT,
                       "compliance": COMPLIANCE})
    df.to_csv(os.path.join(DIR, "table2.csv"), index=False, float_format="%.12g")
    return df


def beta0_fig4():
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdfimages", "-png", "-f", "5", "-l", "5", PDF, os.path.join(tmp, "p5")], check=True)
        A = np.asarray(Image.open(os.path.join(tmp, "p5-000.png")).convert("RGB")).astype(float)
    H, W, _ = A.shape
    gray = (np.abs(A[..., 0] - A[..., 1]) < 12) & (np.abs(A[..., 1] - A[..., 2]) < 12) & (A[..., 0] < 170)
    xa = int(np.argmax(gray[:, :W // 3].sum(0)))                  # y axis
    ya = int(np.argmax(gray.sum(1)))                              # x axis

    def ticks(v):
        t = np.where(v)[0]
        return [g.mean() for g in np.split(t, np.where(np.diff(t) > 2)[0] + 1)]

    ty = ticks((gray[:, xa - 8:xa - 1].sum(1) >= 5) | (gray[:, xa + 2:xa + 9].sum(1) >= 5))
    ty = [y for y in ty if y < ya - 3]                            # drop the x axis itself
    tx = ticks(((gray[ya + 2:ya + 9, :].sum(0) >= 5) | (gray[ya - 8:ya - 1, :].sum(0) >= 5))
               & (np.arange(W) > xa + 3))
    if len(ty) != 8 or len(tx) != 16:
        raise ValueError(f"found {len(ty)} y ticks and {len(tx)} x ticks, expected 8 and 16")
    cy = np.polyfit(ty, np.arange(0.35, -0.001, -0.05), 1)        # value = cy[0] * row + cy[1]
    blue = (np.sqrt(((A - np.array([94, 129, 181])) ** 2).sum(-1)) < 60) | \
           ((A[..., 2] - A[..., 0] > 40) & (A[..., 2] > 120))     # the 26 Nov 2020 curve
    beta = []
    for x in tx:
        x = int(round(x))
        rows = blue[:ya, x - 9:x + 10].sum(1)
        beta.append(np.polyval(cy, int(np.argmax(rows))))
    df = pd.DataFrame({"age": AGES, "beta0": np.round(beta, 4)})
    df.to_csv(os.path.join(DIR, "beta0.csv"), index=False)
    print(f"Fig. 4: axes at x = {xa}, y = {ya}; {abs(cy[0]):.6f} per pixel; beta0 = "
          + " ".join(f"{b:.4f}" for b in df.beta0), flush=True)
    return df


if __name__ == "__main__":
    os.makedirs(DIR, exist_ok=True)
    table2()
    beta0_fig4()
    print(f"wrote {DIR}/table2.csv, beta0.csv", flush=True)
