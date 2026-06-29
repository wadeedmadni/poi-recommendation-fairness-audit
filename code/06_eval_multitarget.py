"""06_eval_multitarget.py
Higher-power evaluation. Per user: first 80% of check-ins = train, last 20% = test.
Global popularity & transitions are learned from TRAIN only (no leakage).
We predict the next POI at EVERY test step, average per user -> one stable score
per user, then audit by SES group with permutation p + bootstrap CI (user-level)."""
import pandas as pd, numpy as np
from collections import Counter, defaultdict
DATA="/sessions/focused-wizardly-heisenberg/mnt/Research_Paper/data"
MIN_CHECKINS, CAP = 10, 50
chk=pd.read_csv(f"{DATA}/processed/checkins_with_income.csv", usecols=["user_id","venue_id"])
ses=pd.read_csv(f"{DATA}/processed/users_ses.csv", usecols=["user_id","ses_group"])
chk=chk.merge(ses,on="user_id",how="inner").sort_index(kind="stable")
seqs=chk.groupby("user_id",sort=False)["venue_id"].apply(list).to_dict()
group=ses.set_index("user_id")["ses_group"].to_dict()
users=[u for u,s in seqs.items() if len(s)>=MIN_CHECKINS]

# train portion = first 80% per user ; build global stats from train only
split={u:int(0.8*len(seqs[u])) for u in users}
pop=Counter(); trans=defaultdict(Counter)
for u in users:
    tr=seqs[u][:split[u]]; pop.update(tr)
    for a,b in zip(tr[:-1],tr[1:]): trans[a][b]+=1
pop_rank=[v for v,_ in pop.most_common()]

def ranked_list(model, context):
    if model=="popular":
        return pop_rank[:CAP]
    if model=="markov":
        base=[v for v,_ in trans[context[-1]].most_common()]
    else:  # userhist
        base=[v for v,_ in Counter(context).most_common()]
    out=base[:CAP]; seen=set(out)
    if len(out)<CAP:                       # cheap backfill: only need up to CAP items
        for v in pop_rank:
            if v not in seen:
                out.append(v)
                if len(out)>=CAP: break
    return out

def per_user_scores(model):
    res={}  # uid -> (h5,h10,rr) averaged over test steps
    for u in users:
        seq=seqs[u]; s=split[u]
        h5=h10=rr=0; k=0
        for i in range(s, len(seq)):
            ctx=seq[:i]; tgt=seq[i]; lst=ranked_list(model, ctx)
            h5 += tgt in lst[:5]; h10 += tgt in lst[:10]
            if tgt in lst: rr += 1.0/(lst.index(tgt)+1)
            k+=1
        if k: res[u]=(h5/k, h10/k, rr/k)
    return res

rng=np.random.default_rng(0)
def perm_p(a,b,n=5000):
    obs=b.mean()-a.mean(); pool=np.concatenate([a,b]); na=len(a); c=0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(pool[na:].mean()-pool[:na].mean())>=abs(obs): c+=1
    return obs,c/n
def boot_ci(a,b,n=2000):
    d=np.empty(n)
    for i in range(n): d[i]=rng.choice(b,len(b),True).mean()-rng.choice(a,len(a),True).mean()
    return np.percentile(d,[2.5,97.5])

print(f"users={len(users)} (>= {MIN_CHECKINS} check-ins); test steps total ~",
      sum(len(seqs[u])-split[u] for u in users))
for model in ["popular","markov","userhist"]:
    sc=per_user_scores(model)
    arr={g:np.array([sc[u][j] for u in sc if group[u]==g]) for g in ["low","mid","high"]
         for j in [0]}  # placeholder
    # collect per metric
    g_vals={g:{m:np.array([sc[u][mi] for u in sc if group[u]==g]) for mi,m in enumerate(["h5","h10","rr"])} for g in ["low","mid","high"]}
    print(f"\n=== {model} ===  ({'group':<5}{'HR@5':>8}{'HR@10':>8}{'MRR':>8})")
    for g in ["low","mid","high"]:
        print(f"            {g:<5}{g_vals[g]['h5'].mean():>8.3f}{g_vals[g]['h10'].mean():>8.3f}{g_vals[g]['rr'].mean():>8.3f}")
    for m in ["h5","h10","rr"]:
        a=g_vals['low'][m]; b=g_vals['high'][m]
        obs,p=perm_p(a,b); lo,hi=boot_ci(a,b)
        sig="**SIG**" if (p<0.05 and (lo>0 or hi<0)) else "ns"
        print(f"   gap {m:<4} high-low={obs:+.3f}  p={p:.4f}  CI[{lo:+.3f},{hi:+.3f}]  {sig}")
