"""Stringency factors s_i(t) = 1 - SI_i(t)/100 of the 51 locations (Glasser et al. 2023,
Section 4.1), with SI_i the daily Oxford COVID-19 Government Response Tracker stringency
index of state i (Hale et al. 2021): OxCGRT_compact_subnational_v1.csv of
https://github.com/OxCGRT/covid-policy-dataset, column StringencyIndex_Average,
CountryCode USA, Jurisdiction STATE_TOTAL; 1 January 2020 - 31 December 2022.
The model evaluates s_i(t) by linear interpolation between the daily values (day d at
t = d - 31 December 2019), and keeps the last value after 31 December 2022.

Usage:  python3 -u oxcgrt_s_51.py   ->  data/oxcgrt/s_51_daily.csv (column t, then one column per state)
"""
import os
import urllib.request

import pandas as pd

from census_N_51x16 import FIPS

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.normpath(os.path.join(HERE, "..", "..", "data", "Downloaded_from_source", "oxcgrt",
                                    "OxCGRT_compact_subnational_v1.csv.gz"))
URL = "https://raw.githubusercontent.com/OxCGRT/covid-policy-dataset/main/data/OxCGRT_compact_subnational_v1.csv"
OUT = os.path.join(HERE, "data", "oxcgrt", "s_51_daily.csv")
T0 = pd.Timestamp("2019-12-31")


def build():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if not os.path.exists(RAW):
        from census_N_51x16 import fetch_gz
        fetch_gz(URL, RAW)
    x = pd.read_csv(RAW, usecols=["CountryCode", "RegionCode", "Jurisdiction", "Date",
                                  "StringencyIndex_Average"], low_memory=False)
    x = x[(x.CountryCode == "USA") & (x.Jurisdiction == "STATE_TOTAL")]
    x["state"] = x.RegionCode.str[3:]
    x["t"] = (pd.to_datetime(x.Date.astype(str), format="%Y%m%d") - T0).dt.days
    si = x.pivot(index="t", columns="state", values="StringencyIndex_Average")
    states = [FIPS[k] for k in sorted(FIPS)]
    if sorted(si.columns) != sorted(states) or si.isna().any().any():
        raise ValueError("states missing or gaps in the OxCGRT data")
    s = 1 - si[states] / 100
    s.to_csv(OUT, float_format="%.6g")
    print(f"t = {s.index.min()} - {s.index.max()}; s at t = 251: {s.loc[251].min():.2f} - {s.loc[251].max():.2f};"
          f" at t = 349: {s.loc[349].min():.2f} - {s.loc[349].max():.2f}; wrote {OUT}", flush=True)


if __name__ == "__main__":
    build()
