"""09_provider_exposure.py
PROVIDER-side fairness: when a recommender hands out its top-10 slots, do venues
in LOW-income neighbourhoods get their fair share of exposure -- or do popular
(often high-income-area) venues hog it? We compare each venue group's share of
RECOMMENDATIONS to its share of actual DEMAND (real visits).
exposure_ratio = rec_share / demand_share :  <1 under-exposed, >1 over-exposed."""
import pandas as pd, numpy as np
from collections import Counter
import heapq
DATA="/sessions/focused-wizardly-heisenberg/mnt/Research_Paper/data"
MIN_CHECKINS, TOPP, K = 10, 50, 10
chk=pd.read_csv(f"{DATA}/processed/checkins_with_income.csv",
                dtype={"GEOID":str}, usecols=["user_id","venue_id","income"])
# venue -> its neighbourhood income (a venue sits in one tract)
ven=chk.dropna(subset=["income"]).drop_duplicates("venue_id").set_index("venue_id")["income"]
# split venues into low/mid/high by their neighbourhood income (tertiles across venues)
vgrp=pd.qcut(ven, 3, labels=["low","mid","high"]).to_dict()
inc_of=ven.to_dict()

seqs=chk.groupby("user_id",sort=False)["venue_id"].apply(list).to_dict()
users=[u for u,s in seqs.items() if len(s)>=MIN_CHECKINS]

pop=Counter()
for u in users: pop.update(seqs[u][:int(0.8*len(seqs[u]))])
tot=sum(pop.values()); pop_norm={v:c/tot for v,c in pop.items()}
top_pop=[v for v,_ in pop.most_common(TOPP)]

def topk(L, cnt, ulen):
    cands=set(top_pop)|set(cnt)
    scored=[((1-L)*pop_norm.get(v,0)+L*(cnt.get(v,0)/ulen), v) for v in cands]
    return [v for _,v in heapq.nlargest(K, scored)]

def venue_group_share(counter):
    """share of counts falling in each venue income group (ignoring unknown)."""
    g=Counter()
    for v,c in counter.items():
        if v in vgrp: g[vgrp[v]]+=c
    s=sum(g.values()); return {k: g[k]/s for k in ["low","mid","high"]}

# DEMAND = real test-target visits by venue group
demand=Counter()
rec={0.0:Counter(), 0.5:Counter()}
for u in users:
    seq=seqs[u]; n=len(seq); start=max(int(0.8*n), n-10)
    cnt=Counter(seq[:start]); ulen=start
    for i in range(start,n):
        demand[seq[i]]+=1
        for L in (0.0,0.5):
            for v in topk(L,cnt,ulen): rec[L][v]+=1
        cnt[seq[i]]+=1; ulen+=1

dem=venue_group_share(demand)
sup=Counter(vgrp.values()); sup={k:sup[k]/sum(sup.values()) for k in ["low","mid","high"]}
print(f"venues with income: {len(vgrp):,}")
print(f"{'group':<6}{'venue_supply%':>14}{'demand%':>10}")
for g in ["low","mid","high"]:
    print(f"{g:<6}{sup[g]*100:>13.1f}{dem[g]*100:>10.1f}")
for L in (0.0,0.5):
    rs=venue_group_share(rec[L])
    tag="popularity" if L==0 else "personalized (L=0.5)"
    print(f"\n--- {tag} : recommendation exposure ---")
    print(f"{'group':<6}{'rec%':>8}{'demand%':>10}{'exposure_ratio':>16}")
    for g in ["low","mid","high"]:
        print(f"{g:<6}{rs[g]*100:>8.1f}{dem[g]*100:>10.1f}{(rs[g]/dem[g]):>16.2f}")
