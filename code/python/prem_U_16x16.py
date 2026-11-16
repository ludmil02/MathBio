"""Contact matrix U (16 x 16, contacts per day) from Prem, Cook, Jit 2017
(PLoS Comput. Biol. 13, e1005697), supporting file S2 (pcbi.1005697.s002.zip),
MUestimates_all_locations_2.xlsx, sheet "United States of America": all locations,
row = age group of the individual, column = age group of the contact, 0-4, ..., 75+.

Balanced as in Glasser et al. 2023 (J. Theor. Biol. 556, 111296), p. 5: each entry
replaced by the geometric mean of it and its transposed entry, U = sqrt(M * M^T).

Usage:  python3 -u prem_U_16x16.py      (expects MathBio/data/Downloaded_from_source/prem2017/
        MUestimates_all_locations_2.xlsx, from pcbi.1005697.s002.zip)  ->  data/prem2017/U_16x16.csv
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "data", "prem2017")
XLSX = os.path.normpath(os.path.join(HERE, "..", "..", "data", "Downloaded_from_source", "prem2017",
                                     "MUestimates_all_locations_2.xlsx"))
SHEET = "United States of America"
OUT = os.path.join(DIR, "U_16x16.csv")
AGES = [f"{5 * k}-{5 * k + 4}" for k in range(15)] + ["75+"]


def build():
    M = pd.read_excel(XLSX, sheet_name=SHEET, header=None).to_numpy(float)
    if M.shape != (16, 16) or not np.all(M > 0):
        raise ValueError(f"unexpected matrix in {XLSX}, sheet {SHEET}: shape {M.shape}")
    U = np.sqrt(M * M.T)
    pd.DataFrame(U, index=AGES, columns=AGES).to_csv(OUT, index_label="age", float_format="%.17g")
    print(f"Prem et al. 2017, US, all locations: row sums {M.sum(1).min():.2f} - {M.sum(1).max():.2f},"
          f" max |M - M^T| / max M = {np.abs(M - M.T).max() / M.max():.3f}")
    print(f"balanced U: row sums {U.sum(1).min():.2f} - {U.sum(1).max():.2f}; wrote {OUT}")
    return U


if __name__ == "__main__":
    build()
