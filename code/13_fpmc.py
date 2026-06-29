"""13_fpmc.py -- FPMC (Factorized Personalized Markov Chains, Rendle 2010), pure NumPy.
A genuine LEARNED sequential recommender: score(u, prev=l, item=i) =
  <VUI[u],VIU[i]> + <VLI[l],VIL[i]>, trained with BPR-SGD. We evaluate it on the
PU split (where the SES gap is significant) to see whether a learned model behaves
like popularity (+gap) or personalization (-gap)."""
import numpy as np, pandas as pd
from collections import Counter
DATA="/sessions/focused-wizardly-heisenberg/mnt/Research_Paper/data"
K, LR, REG, EPOCHS, B = 32, 0.05, 0.005, 12, 4096
rng=np.random.default_rng(0)
chk=pd.read_csv(f"{DATA}/processed/checkins_with_income.csv", usecols=["user_id","venue_id"]).sort_index(kind="stable")
ses=pd.read_csv(f"{DATA}/processed/users_ses.csv", usecols=["user_id","ses_group"])
grp=ses.set_index("user_id")["ses_group"].to_dict()
seqs=chk.groupby("user_id",sort=False)["venue_id"].apply(list).to_dict()
users=[u for u in seqs if len(seqs[u])>=10 and u in grp]

# index maps
items=sorted({v for u in users for v in seqs[u]}); it={v:i for i,v in enumerate(items)}
um={u:i for i,u in enumerate(users)}
NI=len(items); NU=len(users)
# training transitions from first 80% of each user
U=[];L=[];I=[]
pop=Counter()
for u in users:
    s=seqs[u]; cut=int(0.8*len(s)); tr=s[:cut]; pop.update(tr)
    for a,b in zip(tr[:-1],tr[1:]):
        U.append(um[u]); L.append(it[a]); I.append(it[b])
U=np.array(U);L=np.array(L);I=np.array(I)
top_pop=[it[v] for v,_ in pop.most_common(50)]; top_set=set(top_pop)

VUI=0.1*rng.standard_normal((NU,K)); VIU=0.1*rng.standard_normal((NI,K))
VLI=0.1*rng.standard_normal((NI,K)); VIL=0.1*rng.standard_normal((NI,K))
def sig(x): return 1/(1+np.exp(-x))
n=len(U)
for ep in range(EPOCHS):
    p=rng.permutation(n)
    for st in range(0,n,B):
        idx=p[st:st+B]; u=U[idx];l=L[idx];ip=I[idx]; ineg=rng.integers(0,NI,len(idx))
        sp=(VUI[u]*VIU[ip]).sum(1)+(VLI[l]*VIL[ip]).sum(1)
        sn=(VUI[u]*VIU[ineg]).sum(1)+(VLI[l]*VIL[ineg]).sum(1)
        d=sig(-(sp-sn))[:,None]
        np.add.at(VUI,u, LR*(d*(VIU[ip]-VIU[ineg])-REG*VUI[u]))
        np.add.at(VIU,ip,LR*(d*VUI[u]-REG*VIU[ip]))
        np.add.at(VIU,ineg,LR*(-d*VUI[u]-REG*VIU[ineg]))
        np.add.at(VLI,l, LR*(d*(VIL[ip]-VIL[ineg])-REG*VLI[l]))
        np.add.at(VIL,ip,LR*(d*VLI[l]-REG*VIL[ip]))
        np.add.at(VIL,ineg,LR*(-d*VLI[l]-REG*VIL[ineg]))

# evaluate on PU split test region
def perm_p(a,b,n=4000):
    obs=b.mean()-a.mean(); pool=np.concatenate([a,b]); na=len(a); c=0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(pool[na:].mean()-pool[:na].mean())>=abs(obs): c+=1
    return obs,c/n
peruser={}  # user -> list of (histlen,hit10,rr)
for u in users:
    s=seqs[u]; cut=int(0.8*len(s)); uu=um[u]
    hist=set(s[:cut])
    for i in range(cut,len(s)):
        tgt=s[i]; prev=s[i-1]
        if tgt not in it: continue
        cand=list(top_set|{it[v] for v in s[:i] if v in it}|{it[tgt]})
        cand=np.array(cand)
        sc=VIU[cand]@VUI[uu]+VIL[cand]@VLI[it[prev]]
        ti=np.where(cand==it[tgt])[0][0]; rank=int((sc>sc[ti]).sum())+1
        peruser.setdefault(u,[]).append((i,1.0 if rank<=10 else 0.0,1.0/rank if rank<=50 else 0.0))
        hist.add(tgt)

def gap(sel):
    lo=[];hi=[]
    for u,rows in peruser.items():
        if sel=="last10":
            rows=sorted(rows)[-10:]
        h10=np.mean([r[1] for r in rows])
        (lo if grp[u]=='low' else hi if grp[u]=='high' else []).append(h10) if grp[u] in('low','high') else None
    lo=np.array(lo);hi=np.array(hi); obs,p=perm_p(lo,hi)
    return lo.mean(),hi.mean(),obs,p
for sel in ["all","last10"]:
    lm,hm,g,p=gap(sel)
    print(f"FPMC PU/{sel:<6}: low={lm:.3f} high={hm:.3f} gap(high-low)={g:+.3f} p={p:.4f} {'SIG' if p<0.05 else 'ns'}")
print("compare: popularity(λ0) PU/all gap=+0.016*  ; personalization(λ1) PU/all gap=-0.036*")
