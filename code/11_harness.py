"""11_harness.py  -- unified evaluation harness (run per split).
Model axis = blend lambda in {0,0.5,1.0} (0=popularity, 1=personalization).
Saves per (user, history-length, lambda): hit5, hit10, rr  -> results/ranks_<split>.csv
Usage: python3 11_harness.py PU    |    python3 11_harness.py LUO
"""
import sys, pandas as pd, numpy as np
from collections import Counter
DATA="/sessions/focused-wizardly-heisenberg/mnt/Research_Paper/data"
split_mode=sys.argv[1]; TOPP=50; LAMS=[0.0,0.5,1.0]
chk=pd.read_csv(f"{DATA}/processed/checkins_with_income.csv", usecols=["user_id","venue_id"]).sort_index(kind="stable")
ses=pd.read_csv(f"{DATA}/processed/users_ses.csv", usecols=["user_id","ses_group"])
grp=ses.set_index("user_id")["ses_group"].to_dict()
seqs=chk.groupby("user_id",sort=False)["venue_id"].apply(list).to_dict()
users=[u for u in seqs if len(seqs[u])>=10 and u in grp]

if split_mode=="PU":            # per-user temporal: global stats from everyone's first 80%
    pop=Counter()
    for u in users: pop.update(seqs[u][:int(0.8*len(seqs[u]))])
    eval_users=users
    def positions(u): n=len(seqs[u]); return range(int(0.8*n), n)      # test region
else:                            # leave-users-out: stats from train half, test the other half
    rng=np.random.default_rng(0); us=list(users); rng.shuffle(us)
    train_u=us[:len(us)//2]; eval_users=us[len(us)//2:]
    pop=Counter()
    for u in train_u: pop.update(seqs[u])
    def positions(u):
        n=len(seqs[u]); early=list(range(1,min(n,31))); recent=list(range(max(31,n-15),n))
        return sorted(set(early+recent))                                 # cold-start + recent
tot=sum(pop.values()); pop_norm={v:c/tot for v,c in pop.items()}
top_pop=[v for v,_ in pop.most_common(TOPP)]; top_set=set(top_pop)

rows=[]
for u in eval_users:
    seq=seqs[u]; g=grp[u]; cnt=Counter(seq[:1]) if False else Counter()
    # we need cnt = seq[:i]; rebuild incrementally
    pos=set(positions(u)); cnt=Counter(); 
    for i in range(len(seq)):
        if i in pos and i>=1:
            tgt=seq[i]; ulen=i; cands=list(top_set|set(cnt))
            if tgt in cands:
                pn=np.array([pop_norm.get(v,0.0) for v in cands])
                fr=np.array([cnt.get(v,0)/ulen for v in cands])
                ti=cands.index(tgt)
                for lam in LAMS:
                    s=(1-lam)*pn+lam*fr; st=s[ti]; rank=int((s>st).sum())+1
                    rows.append((u,g,i,lam,1.0 if rank<=5 else 0.0,1.0 if rank<=10 else 0.0,
                                 1.0/rank if rank<=50 else 0.0))
            else:
                for lam in LAMS: rows.append((u,g,i,lam,0.0,0.0,0.0))
        cnt[seq[i]]+=1

df=pd.DataFrame(rows, columns=["user","ses","histlen","lam","hit5","hit10","rr"])
out=f"{DATA}/processed/ranks_{split_mode}.csv"; df.to_csv(out,index=False)
print(f"{split_mode}: eval_users={len(eval_users)} rows={len(df)} positions/user~{len(df)//max(1,len(eval_users))//3}")
print(f"saved -> {out}")
