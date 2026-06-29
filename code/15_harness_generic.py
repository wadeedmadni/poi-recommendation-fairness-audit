import sys, pandas as pd, numpy as np
from collections import Counter
CK,US,split_mode,OUT=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]
TOPP=50; LAMS=[0.0,0.5,1.0]
chk=pd.read_csv(CK, usecols=["user_id","venue_id"])
ses=pd.read_csv(US, usecols=["user_id","ses_group"])
grp=ses.set_index("user_id")["ses_group"].to_dict()
seqs=chk.groupby("user_id",sort=False)["venue_id"].apply(list).to_dict()
users=[u for u in seqs if len(seqs[u])>=10 and u in grp]
if split_mode=="PU":
    pop=Counter()
    for u in users: pop.update(seqs[u][:int(0.8*len(seqs[u]))])
    eval_users=users
    def positions(u): n=len(seqs[u]); return range(int(0.8*n),n)
else:
    rng=np.random.default_rng(0); us=list(users); rng.shuffle(us)
    train_u=us[:len(us)//2]; eval_users=us[len(us)//2:]
    pop=Counter()
    for u in train_u: pop.update(seqs[u])
    def positions(u):
        n=len(seqs[u]); return sorted(set(list(range(1,min(n,31)))+list(range(max(31,n-15),n))))
tot=sum(pop.values()); pop_norm={v:c/tot for v,c in pop.items()}
top_set=set(v for v,_ in pop.most_common(TOPP))
rows=[]
for u in eval_users:
    seq=seqs[u]; g=grp[u]; pos=set(positions(u)); cnt=Counter()
    for i in range(len(seq)):
        if i in pos and i>=1:
            tgt=seq[i]; ulen=i; cands=list(top_set|set(cnt))
            if tgt in cands:
                pn=np.array([pop_norm.get(v,0.0) for v in cands]); fr=np.array([cnt.get(v,0)/ulen for v in cands]); ti=cands.index(tgt)
                for lam in LAMS:
                    s=(1-lam)*pn+lam*fr; r=int((s>s[ti]).sum())+1
                    rows.append((u,g,i,lam,1.0 if r<=5 else 0.0,1.0 if r<=10 else 0.0,1.0/r if r<=50 else 0.0))
            else:
                for lam in LAMS: rows.append((u,g,i,lam,0.0,0.0,0.0))
        cnt[seq[i]]+=1
pd.DataFrame(rows,columns=["user","ses","histlen","lam","hit5","hit10","rr"]).to_csv(OUT,index=False)
print(f"{split_mode}: users={len(eval_users)} rows={len(rows)} -> {OUT}")
