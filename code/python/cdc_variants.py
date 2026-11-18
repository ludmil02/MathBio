"""Days at which the variants of Table S4 of the supplement of Feng, Day, Glasser 2026
(Wuhan, Alpha, Delta, BS1, BA5, the column names of Table S4) become dominant in the United
States, from the CDC data set
"SARS-CoV-2 Variant Proportions" (data.cdc.gov, id jr58-6ysp), USA, model type "weighted",
for each week the estimates of the latest creation date.

Lineages (names as in the CDC data) are grouped by family: Alpha = B.1.1.7, Q.*; Delta =
B.1.617.2, AY.*; BS1 = B.1.1.529, BA.1*; BA5 = BA.5*, BQ.*, BF.*, BE.*. Table S4 names the
columns only; the lineages assigned to each are our choice. The switch day of a family is the
week-ending date of the first week in which its share exceeds 1/2; t counts days from
31 December 2019.  ->  data/cdc/variant_days.csv

Usage:  python3 -u cdc_variants.py
"""
import os
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "data", "cdc")
RAW = os.path.normpath(os.path.join(HERE, "..", "..", "data", "Downloaded_from_source", "cdc", "variants_usa.csv.gz"))
URL = ("https://data.cdc.gov/resource/jr58-6ysp.csv?$where=usa_or_hhsregion='USA'%20AND%20"
       "modeltype='weighted'%20AND%20week_ending<'2023-06-01'&$limit=500000")
T0 = pd.Timestamp("2019-12-31")


def family(name):
    n = str(name)
    if n.startswith(("B.1.1.7", "Q.")):
        return "Alpha"
    if n.startswith(("B.1.617.2", "AY.")):
        return "Delta"
    if n.startswith(("BA.5", "BQ.", "BF.", "BE.")):
        return "BA5"
    if n.startswith(("B.1.1.529", "BA.1")):
        return "BS1"
    return "other"


def build():
    os.makedirs(DIR, exist_ok=True)
    if not os.path.exists(RAW):
        from census_N_51x16 import fetch_gz
        fetch_gz(URL, RAW)
    v = pd.read_csv(RAW, parse_dates=["week_ending", "creation_date"])
    v = v[v.creation_date == v.groupby("week_ending").creation_date.transform("max")]
    v["family"] = v.variant.map(family)
    share = v.groupby(["week_ending", "family"]).share.sum().unstack().fillna(0.0)
    rows = [{"variant": "Wuhan", "date": "", "t": float("-inf"), "max_share": float("nan")}]
    for f in ("Alpha", "Delta", "BS1", "BA5"):
        weeks = share.index[share[f] > 0.5]
        d = weeks.min()
        rows.append({"variant": f, "date": d.date().isoformat(), "t": float((d - T0).days),
                     "max_share": round(float(share[f].max()), 3)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(DIR, "variant_days.csv"), index=False)
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    build()
