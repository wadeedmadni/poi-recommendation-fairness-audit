"""14_austin_prep.py -- city #2 (Austin/Gowalla): spatial join + SES grouping.
Mirrors NYC steps 01-02 but for Gowalla check-ins + Travis County TX income."""
import pandas as pd, numpy as np, geopandas as gpd
from collections import Counter
DATA="/sessions/focused-wizardly-heisenberg/mnt/Research_Paper/data"
g=pd.read_csv(f"{DATA}/gowalla_austin.tsv", sep="\t", header=None,
              names=["user_id","time","lat","lon","venue_id"])
print("austin checkins:", len(g), "users:", g.user_id.nunique(), "venues:", g.venue_id.nunique())

# Travis County tracts (county FIPS 453) + income
tr=gpd.read_file(f"{DATA}/tracts_tx/tl_2022_48_tract.shp")
tr=tr[tr["GEOID"].str[2:5]=="453"][["GEOID","geometry"]].to_crs("EPSG:4326")
inc=pd.read_csv(f"{DATA}/income_tx_b19013/ACSDT5Y2022.B19013-Data.csv", skiprows=[1])
inc["GEOID"]=inc["GEO_ID"].str.split("US").str[-1]
inc["income"]=pd.to_numeric(inc["B19013_001E"].astype(str).str.replace(r"[^0-9]","",regex=True),errors="coerce")
tr=tr.merge(inc[["GEOID","income"]],on="GEOID",how="left")

# spatial join on UNIQUE venues (locations) only -> fast
vid=g.drop_duplicates("venue_id")[["venue_id","lat","lon"]]
pts=gpd.GeoDataFrame(vid, geometry=gpd.points_from_xy(vid.lon,vid.lat), crs="EPSG:4326")
vj=gpd.sjoin(pts, tr, how="left", predicate="within")[["venue_id","GEOID","income"]]
g=g.merge(vj,on="venue_id",how="left")
matched=g["income"].notna().mean()
print(f"checkins with income: {matched:.1%}")

# local hour (Central) for home inference
g["hour"]=pd.to_datetime(g["time"],utc=True,errors="coerce").dt.tz_convert("America/Chicago").dt.hour
geoid2income=g.dropna(subset=["income"]).drop_duplicates("GEOID").set_index("GEOID")["income"].to_dict()
def modal(sub):
    return sub.groupby("user_id")["GEOID"].agg(lambda s: s.dropna().mode().iloc[0] if not s.dropna().mode().empty else np.nan)
night=modal(g[(g.hour>=22)|(g.hour<6)]); allm=modal(g)
rows=[]
for u in g.user_id.unique():
    for series in (night,allm):
        x=series.get(u,np.nan)
        if pd.notna(x) and x in geoid2income:
            rows.append((u,x,geoid2income[x])); break
us=pd.DataFrame(rows,columns=["user_id","home_GEOID","home_income"])
us["ses_group"]=pd.qcut(us["home_income"],3,labels=["low","mid","high"])
print("users placed:",len(us),"of",g.user_id.nunique())
print(us["ses_group"].value_counts().sort_index().to_string())

g[["user_id","venue_id","GEOID","income"]].to_csv(f"{DATA}/processed/austin_checkins.csv",index=False)
us.to_csv(f"{DATA}/processed/austin_users_ses.csv",index=False)
print("saved austin processed files")
