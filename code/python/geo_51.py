"""Distances d_ip and activity factors L_i of the 51 locations (50 states and DC), from
public Census Bureau files in MathBio/data/Downloaded_from_source/census/ (downloaded by this script if
missing); results in data/census/:

d_51x51.csv  distances (miles) between the 2010 centers of population of the states
             (CenPop2010_Mean_ST.txt), as in Glasser et al. 2023, Fig. 6: geodesic distances
             on the WGS84 ellipsoid (Vincenty's inverse formula), 1 mile = 1609.344 m.
L_51.csv     L_i = ln(rho_i) / ln(rho), the densities in persons per square mile of land
             (Glasser et al. 2023, Section 2.4), with rho_i the population-weighted average
             of the densities of the counties of location i and rho the population-weighted
             average of the rho_i (supplement of Feng, Day, Glasser 2026, Section 1a).
             County populations: Vintage 2024 estimates, July 1, 2020 (co-est2024-alldata.csv,
             POPESTIMATE2020); county land areas: 2024 Gazetteer (ALAND_SQMI).

Usage:  python3 -u geo_51.py
"""
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from census_N_51x16 import FIPS

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "data", "census")                       # results
RAW = os.path.normpath(os.path.join(HERE, "..", "..", "data", "Downloaded_from_source", "census"))   # downloads
URLS = {   # kept gzipped
    "CenPop2010_Mean_ST.txt.gz": "https://www2.census.gov/geo/docs/reference/cenpop2010/CenPop2010_Mean_ST.txt",
    "co-est2024-alldata.csv.gz": "https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/counties/totals/co-est2024-alldata.csv",
}
GAZ_ZIP = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2024_Gazetteer/2024_Gaz_counties_national.zip"


def fetch():
    from census_N_51x16 import fetch_gz
    for f, u in URLS.items():
        p = os.path.join(RAW, f)
        if not os.path.exists(p):
            print(f"downloading {u}", flush=True)
            fetch_gz(u, p)
    zp = os.path.join(RAW, "2024_Gaz_counties_national.zip")    # read as is, not unpacked
    if not os.path.exists(zp):
        os.makedirs(RAW, exist_ok=True)
        print(f"downloading {GAZ_ZIP}", flush=True)
        urllib.request.urlretrieve(GAZ_ZIP, zp)


def vincenty_miles(lat1, lon1, lat2, lon2):
    """Geodesic distance on the WGS84 ellipsoid (Vincenty 1975, inverse problem), in miles."""
    a, f = 6378137.0, 1 / 298.257223563
    b = (1 - f) * a
    if lat1 == lat2 and lon1 == lon2:
        return 0.0
    U1, U2 = np.arctan((1 - f) * np.tan(np.radians(lat1))), np.arctan((1 - f) * np.tan(np.radians(lat2)))
    Lr = np.radians(lon2 - lon1)
    lam = Lr
    for _ in range(200):
        sl, cl = np.sin(lam), np.cos(lam)
        ss = np.hypot(np.cos(U2) * sl, np.cos(U1) * np.sin(U2) - np.sin(U1) * np.cos(U2) * cl)
        cs = np.sin(U1) * np.sin(U2) + np.cos(U1) * np.cos(U2) * cl
        sig = np.arctan2(ss, cs)
        sa = np.cos(U1) * np.cos(U2) * sl / ss
        c2a = 1 - sa ** 2
        c2m = cs - 2 * np.sin(U1) * np.sin(U2) / c2a
        C = f / 16 * c2a * (4 + f * (4 - 3 * c2a))
        lam_new = Lr + (1 - C) * f * sa * (sig + C * ss * (c2m + C * cs * (-1 + 2 * c2m ** 2)))
        if abs(lam_new - lam) < 1e-13:
            lam = lam_new
            break
        lam = lam_new
    else:
        raise RuntimeError("Vincenty iteration did not converge")
    u2 = c2a * (a ** 2 - b ** 2) / b ** 2
    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    ds = B * ss * (c2m + B / 4 * (cs * (-1 + 2 * c2m ** 2) - B / 6 * c2m * (-3 + 4 * ss ** 2) * (-3 + 4 * c2m ** 2)))
    return b * A * (sig - ds) / 1609.344


def distances():
    cp = pd.read_csv(os.path.join(RAW, "CenPop2010_Mean_ST.txt.gz"), dtype={"STATEFP": int})
    cp = cp[cp.STATEFP.isin(FIPS)].sort_values("STATEFP")
    if list(cp.STATEFP) != sorted(FIPS):
        raise ValueError("states missing in CenPop2010_Mean_ST.txt")
    la, lo = cp.LATITUDE.to_numpy(float), cp.LONGITUDE.to_numpy(float)
    m = len(la)
    D = np.array([[vincenty_miles(la[i], lo[i], la[p], lo[p]) for p in range(m)] for i in range(m)])
    abbr = [FIPS[s] for s in cp.STATEFP]
    pd.DataFrame(D, index=abbr, columns=abbr).to_csv(os.path.join(DIR, "d_51x51.csv"),
                                                     index_label="state", float_format="%.10g")
    return D


def activity():
    co = pd.read_csv(os.path.join(RAW, "co-est2024-alldata.csv.gz"), encoding="latin-1")
    co = co[co.SUMLEV == 50]
    co["GEOID"] = co.STATE * 1000 + co.COUNTY
    with zipfile.ZipFile(os.path.join(RAW, "2024_Gaz_counties_national.zip")) as z:
        with z.open("2024_Gaz_counties_national.txt") as f:
            gz = pd.read_csv(f, sep="\t")
    gz.columns = [c.strip() for c in gz.columns]
    x = co.merge(gz[["GEOID", "ALAND_SQMI"]], on="GEOID", how="left")
    if x.ALAND_SQMI.isna().any():
        raise ValueError(f"{x.ALAND_SQMI.isna().sum()} counties without land area")
    x = x[x.STATE.isin(FIPS)]
    P = x.POPESTIMATE2020.to_numpy(float)
    x["rho"] = P / x.ALAND_SQMI
    x["Prho"] = P * x.rho
    st = x.groupby("STATE")[["POPESTIMATE2020", "Prho"]].sum()
    rho_i = (st.Prho / st.POPESTIMATE2020).to_numpy()            # population-weighted county densities
    Pi = st.POPESTIMATE2020.to_numpy(float)
    rho = (Pi * rho_i).sum() / Pi.sum()                           # population-weighted state densities
    L = np.log(rho_i) / np.log(rho)
    pd.DataFrame({"state": [FIPS[s] for s in st.index], "rho": rho_i, "L": L}).to_csv(
        os.path.join(DIR, "L_51.csv"), index=False, float_format="%.10g")
    print(f"rho (US, per sq mi) = {rho:.1f}; L_i in [{L.min():.3f}, {L.max():.3f}]", flush=True)
    return L


if __name__ == "__main__":
    fetch()
    D = distances()
    print(f"distances: max {D.max():.1f} miles", flush=True)
    activity()
    print(f"wrote {DIR}/d_51x51.csv, L_51.csv", flush=True)
