"""07_mitigation.py
Mitigation: score(v) = (1-L)*popularity(v) + L*user_history_freq(v).
L=0 is the unfair popularity model; L=1 is pure personalization. We sweep L and
report overall accuracy AND the SES gap (high-low) with significance. Goal: show
the gap closes as we add personalization -- and what it costs in accuracy.
Note: the fix uses only the user's OWN history, never their income (no sensitive
attribute at inference)."""
import pandas as pd, numpy as np
from collections import Counter, defaultdict
DATA="/sessions/focused-wizardly-heisenberg/mnt/Research_Paper/data"
MIN_CHECKINS, TOPP = 10, 50
LAMBDAS=[0.0,0.25,0.5,0.75,1.0]
chk=pd.read_csv(f"{DATA}/processed/checkins_with_income.csv", usecols=["user_id","venue_id"])
ses=pd.read_csv(f"{DATA}/processed/users_ses.csv", usecols=["user_id","ses_group"])
chk=chk.merge(ses,on="user_id",how="inner").sort_index(kind="stable")
seqs=chk.groupby("user_id",sort=False)["venue_id"].apply(list).to_dict()
group=ses.set_index("user_id")["ses_group"].to_dict()
users=[u for u,s in seqs.items() if len(s)>=MIN_CHECKINS]

pop=Counter()
for u in users: pop.update(seqs[u][:int(0.8*len(seqs[u]))])   # train = first 80%
tot=sum(pop.values()); pop_norm={v:c/tot for v,c in pop.items()}
top_pop=[v for v,_ in pop.most_common(TOPP)]

# per-user, per-lambda mean metrics
scores={L:{ 'h5':{}, 'h10':{}, 'rr':{} } for L in LAMBDAS}
for u in users:
    seq=seqs[u]; n=len(seq); split=int(0.8*n); start=max(split, n-10)  # last <=10 test steps
    cnt=Counter(seq[:start]); ulen=start
    acc={L:[0,0,0] for L in LAMBDAS}; k=0
    for i in range(start,n):
        tgt=seq[i]; cands=set(top_pop)|set(cnt)
        for L in LAMBDAS:
            if tgt not in cands:
                continue
            st=(1-L)*pop_norm.get(tgt,0)+L*(cnt.get(tgt,0)/ulen)
            higher=0
            for v in cands:
                s=(1-L)*pop_norm.get(v,0)+L*(cnt.get(v,0)/ulen)
                if s>st: higher+=1
            rank=higher+1
            a=acc[L]; a[0]+=rank<=5; a[1]+=rank<=10; a[2]+=1.0/rank
        cnt[tgt]+=1; ulen+=1; k+=1
    for L in LAMBDAS:
        scores[L]['h5'][u]=acc[L][0]/k; scores[L]['h10'][u]=acc[L][1]/k; scores[L]['rr'][u]=acc[L][2]/k

rng=np.random.default_rng(0)
def perm_p(a,b,n=5000):
    obs=b.mean()-a.mean(); pool=np.concatenate([a,b]); na=len(a); c=0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(pool[na:].mean()-pool[:na].mean())>=abs(obs): c+=1
    return obs,c/n

print(f"users={len(users)}  (lambda 0=popularity ... 1=personalization)")
print(f"{'lambda':>7}{'overall_HR@10':>15}{'gap_HR@10':>12}{'p':>8}{'gap_MRR':>10}{'p':>8}")
for L in LAMBDAS:
    arr=lambda g,m: np.array([scores[L][m][u] for u in users if group[u]==g])
    overall=np.mean([scores[L]['h10'][u] for u in users])
    gh,ph=perm_p(arr('low','h10'),arr('high','h10'))
    gr,pr=perm_p(arr('low','rr'),arr('high','rr'))
    print(f"{L:>7.2f}{overall:>15.3f}{gh:>+12.3f}{ph:>8.4f}{gr:>+10.3f}{pr:>8.4f}")
