"""Population N of the 51 locations x 16 age groups from the US Census Bureau,
Vintage 2024 population estimates, resident population on July 1, 2020
(file sc-est2024-alldata6.csv, kept gzipped; column POPESTIMATE2020; both sexes, all races,
ORIGIN = 0).

Single years of age 0, ..., 84 and 85+ are summed to 0-4, ..., 70-74 and 75+.
The file is downloaded once to MathBio/data/Downloaded_from_source/census/; the result is data/census/N_51x16.csv
(columns state, age, N, in the order of the model: locations by FIPS code, then age).

Usage:  python3 -u census_N_51x16.py
"""
import os
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "data", "census")
URL = "https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/state/asrh/sc-est2024-alldata6.csv"
RAW = os.path.normpath(os.path.join(HERE, "..", "..", "data", "Downloaded_from_source", "census",
                                    "sc-est2024-alldata6.csv.gz"))


def fetch_gz(url, path_gz):
    """Download url and keep it gzipped as path_gz (the file as downloaded, compressed)."""
    import gzip
    import shutil
    os.makedirs(os.path.dirname(path_gz), exist_ok=True)
    tmp = path_gz[:-3]
    urllib.request.urlretrieve(url, tmp)
    with open(tmp, "rb") as fi, gzip.open(path_gz, "wb", compresslevel=9) as fo:
        shutil.copyfileobj(fi, fo)
    os.remove(tmp)
OUT = os.path.join(DIR, "N_51x16.csv")
COLUMN = "POPESTIMATE2020"

FIPS = {1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT", 10: "DE", 11: "DC", 12: "FL",
        13: "GA", 15: "HI", 16: "ID", 17: "IL", 18: "IN", 19: "IA", 20: "KS", 21: "KY", 22: "LA",
        23: "ME", 24: "MD", 25: "MA", 26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE",
        32: "NV", 33: "NH", 34: "NJ", 35: "NM", 36: "NY", 37: "NC", 38: "ND", 39: "OH", 40: "OK",
        41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD", 47: "TN", 48: "TX", 49: "UT", 50: "VT",
        51: "VA", 53: "WA", 54: "WV", 55: "WI", 56: "WY"}
AGES = [f"{5 * k}-{5 * k + 4}" for k in range(15)] + ["75+"]


def build():
    os.makedirs(DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RAW), exist_ok=True)
    if not os.path.exists(RAW):
        print(f"downloading {URL}")
        fetch_gz(URL, RAW)
    df = pd.read_csv(RAW, encoding="latin-1")
    df = df[(df.SEX == 0) & (df.ORIGIN == 0) & (df.STATE > 0)]
    by_age = df.groupby(["STATE", "AGE"])[COLUMN].sum().unstack()       # state x single year of age
    if list(by_age.index) != sorted(FIPS) or list(by_age.columns) != list(range(86)):
        raise ValueError("unexpected states or ages in the census file")
    A = by_age.to_numpy(float)
    N = np.column_stack([A[:, 5 * k:5 * k + 5].sum(axis=1) for k in range(15)] + [A[:, 75:].sum(axis=1)])
    out = pd.DataFrame({"state": np.repeat([FIPS[s] for s in by_age.index], 16),
                        "age": np.tile(AGES, 51), "N": N.ravel().astype(np.int64)})
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(out)} groups, total {out.N.sum():,}")
    return out


if __name__ == "__main__":
    build()
