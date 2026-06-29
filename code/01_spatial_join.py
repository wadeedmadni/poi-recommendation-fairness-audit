"""
01_spatial_join.py
-------------------
Goal of this first step: attach a NEIGHBOURHOOD INCOME to every Foursquare
check-in, by figuring out which census tract each check-in's GPS point falls in.

Pipeline:
  check-in (lat/lon)  ->  census tract (polygon it sits inside)  ->  tract income
This is the "spatial join". If most check-ins get an income, H1 is testable.
"""

import pandas as pd
import geopandas as gpd

DATA = "/sessions/focused-wizardly-heisenberg/mnt/Research_Paper/data"

# NYC's five boroughs as county FIPS codes (the 3 digits after the state code 36)
NYC_COUNTIES = {"005", "047", "061", "081", "085"}  # Bronx, Brooklyn, Manhattan, Queens, Staten Island

# ---------------------------------------------------------------------------
# 1) TRACT SHAPES (the map outlines). One row per census tract.
# ---------------------------------------------------------------------------
tracts = gpd.read_file(f"{DATA}/tracts_ny/tl_2022_36_tract.shp")
# Keep only NYC tracts (state 36, county in our set). GEOID = SS CCC TTTTTT.
tracts = tracts[tracts["GEOID"].str[2:5].isin(NYC_COUNTIES)].copy()
tracts = tracts[["GEOID", "geometry"]]

# ---------------------------------------------------------------------------
# 2) INCOME TABLE. One row per tract. skiprows=[1] drops the 2nd descriptive header.
# ---------------------------------------------------------------------------
inc = pd.read_csv(f"{DATA}/income_b19013/ACSDT5Y2022.B19013-Data.csv", skiprows=[1])
# Turn "1400000US36005000100" into the 11-digit "36005000100" so it matches GEOID.
inc["GEOID"] = inc["GEO_ID"].str.split("US").str[-1]
# B19013_001E is income; "-" / "250,000+" etc. -> coerce to a real number, else NaN.
inc["income"] = pd.to_numeric(inc["B19013_001E"].astype(str).str.replace(r"[^0-9]", "", regex=True),
                              errors="coerce")
inc = inc[["GEOID", "income"]]

# Attach income to each tract shape.
tracts = tracts.merge(inc, on="GEOID", how="left")

# ---------------------------------------------------------------------------
# 3) CHECK-INS. Tab-separated, no header, 8 columns (see dataset readme).
# ---------------------------------------------------------------------------
cols = ["user_id", "venue_id", "venue_cat_id", "venue_cat_name",
        "lat", "lon", "tz_offset", "utc_time"]
chk = pd.read_csv(f"{DATA}/dataset_tsmc2014/dataset_TSMC2014_NYC.txt",
                  sep="\t", header=None, names=cols, encoding="latin-1")

# Turn lat/lon into geographic points. EPSG:4326 = standard GPS lat/lon.
pts = gpd.GeoDataFrame(chk,
                       geometry=gpd.points_from_xy(chk["lon"], chk["lat"]),
                       crs="EPSG:4326")
# Put tract shapes in the same coordinate system before matching.
tracts = tracts.to_crs("EPSG:4326")

# ---------------------------------------------------------------------------
# 4) THE SPATIAL JOIN: for each point, which tract polygon contains it?
# ---------------------------------------------------------------------------
joined = gpd.sjoin(pts, tracts, how="left", predicate="within")

# ---------------------------------------------------------------------------
# 5) REPORT: did it work?
# ---------------------------------------------------------------------------
n = len(joined)
matched_tract = joined["GEOID"].notna().sum()
matched_income = joined["income"].notna().sum()
print(f"check-ins total            : {n:,}")
print(f"matched to a NYC tract     : {matched_tract:,} ({matched_tract/n:.1%})")
print(f"of those, have an income   : {matched_income:,} ({matched_income/n:.1%})")
print(f"unique users (all)         : {joined['user_id'].nunique():,}")
print(f"unique users w/ income     : {joined.loc[joined['income'].notna(),'user_id'].nunique():,}")
print(f"income range $             : {joined['income'].min():,.0f}  to  {joined['income'].max():,.0f}")
print(f"income median $            : {joined['income'].median():,.0f}")

# Save the joined result so later steps don't recompute it.
out = f"{DATA}/processed"
import os; os.makedirs(out, exist_ok=True)
keep = ["user_id", "venue_id", "venue_cat_name", "lat", "lon", "utc_time", "GEOID", "income"]
joined[keep].to_csv(f"{out}/checkins_with_income.csv", index=False)
print(f"\nsaved -> {out}/checkins_with_income.csv")
