"""
02_user_home_ses.py
--------------------
Turn "check-ins with income" into "users labelled low / mid / high income".

For each user we need ONE income number = the income of the neighbourhood they
live in. We infer their home tract in priority order:
  A) tracts of their "Home (private)" check-ins   (most reliable)
  B) else: tracts of their night-time check-ins (22:00-06:00 local)  ~ where they sleep
  C) else: their single most-visited tract
We take the most common tract in whichever method fires first AND has an income.
Then we split users into 3 equal income groups (tertiles): low / mid / high.
"""

import pandas as pd
import numpy as np

DATA = "/sessions/focused-wizardly-heisenberg/mnt/Research_Paper/data"

# Reuse the joined file from step 01 (each check-in already has a tract + income).
# dtype GEOID=str keeps the tract code as text (e.g. "36005000100"), not a number.
df = pd.read_csv(f"{DATA}/processed/checkins_with_income.csv", dtype={"GEOID": str})

# Local NYC time -> hour of day (handles daylight saving correctly).
t = pd.to_datetime(df["utc_time"], format="%a %b %d %H:%M:%S %z %Y")
df["hour"] = t.dt.tz_convert("America/New_York").dt.hour

# Lookup: tract -> income (only tracts that actually have an income).
geoid2income = (df.dropna(subset=["income"])
                  .drop_duplicates("GEOID").set_index("GEOID")["income"].to_dict())

def modal_geoid_by_user(subset):
    """Most common tract per user within a slice of check-ins."""
    if subset.empty:
        return pd.Series(dtype=object)
    return subset.groupby("user_id")["GEOID"].agg(
        lambda s: s.dropna().mode().iloc[0] if not s.dropna().mode().empty else np.nan)

A = modal_geoid_by_user(df[df["venue_cat_name"] == "Home (private)"])        # home check-ins
B = modal_geoid_by_user(df[(df["hour"] >= 22) | (df["hour"] < 6)])           # night check-ins
C = modal_geoid_by_user(df)                                                  # all check-ins

rows = []
for uid in df["user_id"].unique():
    for method, series in (("home", A), ("night", B), ("frequent", C)):
        g = series.get(uid, np.nan)
        if pd.notna(g) and g in geoid2income:             # valid tract WITH income
            rows.append((uid, g, geoid2income[g], method))
            break

users = pd.DataFrame(rows, columns=["user_id", "home_GEOID", "home_income", "home_method"])

# Split into 3 equal-sized income groups.
users["ses_group"] = pd.qcut(users["home_income"], 3, labels=["low", "mid", "high"])

# ---------------------------------------------------------------- report
print(f"users placed into an SES group : {len(users):,} of {df['user_id'].nunique():,}")
print("\nhome inferred via:")
print(users["home_method"].value_counts().to_string())
print("\nSES group sizes:")
print(users["ses_group"].value_counts().sort_index().to_string())
cut_lo = users.loc[users.ses_group=="low","home_income"].max()
cut_hi = users.loc[users.ses_group=="mid","home_income"].max()
print(f"\nincome cutoffs : low <= ${cut_lo:,.0f} < mid <= ${cut_hi:,.0f} < high")
print("median home income per group $:")
print(users.groupby("ses_group", observed=True)["home_income"].median().to_string())


users.to_csv(f"{DATA}/processed/users_ses.csv", index=False)
print(f"saved -> {DATA}/processed/users_ses.csv  ({len(users)} users)")
