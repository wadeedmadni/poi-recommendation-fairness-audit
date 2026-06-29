"""04_mobility_diversity.py
Does low-SES mobility look more REPETITIVE? That would explain why personalization
predicts low-SES users better. Per user we compute:
  unique_ratio = (# distinct places) / (# check-ins)   -> lower = more repetitive
  norm_entropy = spread of their visits across places   -> lower = more concentrated
Then compare low vs mid vs high, with a permutation test (low vs high)."""
import pandas as pd, numpy as np
from collections import Counter
DATA = "/sessions/focused-wizardly-heisenberg/mnt/Research_Paper/data"
MIN_CHECKINS = 5
chk = pd.read_csv(f"{DATA}/processed/checkins_with_income.csv", usecols=["user_id","venue_id"])
ses = pd.read_csv(f"{DATA}/processed/users_ses.csv", usecols=["user_id","ses_group"])
chk = chk.merge(ses, on="user_id", how="inner")
seqs  = chk.groupby("user_id")["venue_id"].apply(list).to_dict()
group = ses.set_index("user_id")["ses_group"].to_dict()

rows=[]
for u,seq in seqs.items():
    if len(seq) < MIN_CHECKINS: continue
    N=len(seq); c=Counter(seq); U=len(c)
    p=np.array(list(c.values()))/N
    ent=-(p*np.log2(p)).sum()
    norm_ent= ent/np.log2(U) if U>1 else 0.0
    rows.append((group[u], N, U, U/N, norm_ent))
d=pd.DataFrame(rows, columns=["ses","checkins","unique","unique_ratio","norm_entropy"])

print(f"{'group':<6}{'users':>7}{'med_checkins':>14}{'mean_unique':>13}{'unique_ratio':>14}{'norm_entropy':>14}")
for g in ["low","mid","high"]:
    s=d[d.ses==g]
    print(f"{g:<6}{len(s):>7}{s.checkins.median():>14.0f}{s.unique.mean():>13.1f}{s.unique_ratio.mean():>14.3f}{s.norm_entropy.mean():>14.3f}")

def perm_p(col, n=5000, seed=0):
    rng=np.random.default_rng(seed)
    a=d.loc[d.ses=="low",col].values; b=d.loc[d.ses=="high",col].values
    obs=a.mean()-b.mean(); pool=np.concatenate([a,b]); na=len(a)
    cnt=0
    for _ in range(n):
        rng.shuffle(pool); 
        if abs(pool[:na].mean()-pool[na:].mean())>=abs(obs): cnt+=1
    return obs, cnt/n

for col in ["unique_ratio","norm_entropy"]:
    obs,p=perm_p(col)
    print(f"\n{col}: low-high diff = {obs:+.3f}  (permutation p = {p:.4f})")
