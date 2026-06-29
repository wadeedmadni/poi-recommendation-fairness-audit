"""05_significance.py
Are the per-group accuracy gaps real or noise? For each model/metric we test the
gap (high - low) with: a permutation p-value (shuffle group labels 5000x) and a
95% bootstrap confidence interval. Significant = p<0.05 AND CI excludes 0."""
import pandas as pd, numpy as np
from collections import Counter, defaultdict
DATA="/sessions/focused-wizardly-heisenberg/mnt/Research_Paper/data"
MIN_CHECKINS, CAP = 5, 50
chk=pd.read_csv(f"{DATA}/processed/checkins_with_income.csv", usecols=["user_id","venue_id"])
ses=pd.read_csv(f"{DATA}/processed/users_ses.csv", usecols=["user_id","ses_group"])
chk=chk.merge(ses,on="user_id",how="inner").sort_index(kind="stable")
seqs=chk.groupby("user_id",sort=False)["venue_id"].apply(list).to_dict()
group=ses.set_index("user_id")["ses_group"].to_dict()
users=[u for u,s in seqs.items() if len(s)>=MIN_CHECKINS]
pop=Counter(); trans=defaultdict(Counter)
for u in users:
    h=seqs[u][:-1]; pop.update(h)
    for a,b in zip(h[:-1],h[1:]): trans[a][b]+=1
pop_rank=[v for v,_ in pop.most_common()]
def toplist(model,u):
    h=seqs[u][:-1]
    if model=="popular": return pop_rank[:CAP]
    base=trans[h[-1]].most_common() if model=="markov" else Counter(h).most_common()
    r=[v for v,_ in base]; seen=set(r); r+=[v for v in pop_rank if v not in seen]; return r[:CAP]

rng=np.random.default_rng(0)
def perm_p(a,b,n=5000):
    obs=b.mean()-a.mean(); pool=np.concatenate([a,b]); na=len(a); c=0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(pool[na:].mean()-pool[:na].mean())>=abs(obs): c+=1
    return obs, c/n
def boot_ci(a,b,n=2000):
    diffs=np.empty(n)
    for i in range(n):
        diffs[i]=rng.choice(b,len(b),True).mean()-rng.choice(a,len(a),True).mean()
    return np.percentile(diffs,[2.5,97.5])

for model in ["popular","markov","userhist"]:
    by={'low':{'h5':[],'h10':[],'rr':[]},'high':{'h5':[],'h10':[],'rr':[]}}
    for u in users:
        g=group[u]
        if g=='mid': continue
        tgt=seqs[u][-1]; lst=toplist(model,u)
        by[g]['h5'].append(1.0 if tgt in lst[:5] else 0.0)
        by[g]['h10'].append(1.0 if tgt in lst[:10] else 0.0)
        rank=lst.index(tgt)+1 if tgt in lst else None
        by[g]['rr'].append(1.0/rank if rank else 0.0)
    print(f"\n=== {model} ===")
    for m in ['h5','h10','rr']:
        a=np.array(by['low'][m]); b=np.array(by['high'][m])
        obs,p=perm_p(a,b); lo,hi=boot_ci(a,b)
        sig="**SIG**" if (p<0.05 and (lo>0 or hi<0)) else "ns"
        print(f"  {m:<4} gap(high-low)={obs:+.3f}  p={p:.4f}  95%CI[{lo:+.3f},{hi:+.3f}]  {sig}")
print("\n(low n=346, high n=344)")
