"""10_coldstart.py
Does the income gap live in COLD-START? We split users 50/50: 'train users' build
the global popularity model; 'test users' are evaluated with an expanding window
(predict check-in i from the first i-1) so we include their very first, history-poor
predictions. Model = realistic blend score = 0.5*popularity + 0.5*own-history.
We bin each prediction by how much history was available and measure the SES gap
per bin (user-level permutation test). No leakage: test users never train the model."""
import pandas as pd, numpy as np
from collections import Counter
DATA="/sessions/focused-wizardly-heisenberg/mnt/Research_Paper/data"
TOPP=50; L=0.5
chk=pd.read_csv(f"{DATA}/processed/checkins_with_income.csv", usecols=["user_id","venue_id"]).sort_index(kind="stable")
ses=pd.read_csv(f"{DATA}/processed/users_ses.csv", usecols=["user_id","ses_group"])
grp=ses.set_index("user_id")["ses_group"].to_dict()
seqs=chk.groupby("user_id",sort=False)["venue_id"].apply(list).to_dict()
allu=[u for u in seqs if len(seqs[u])>=10 and u in grp]

rng=np.random.default_rng(0)
allu=list(allu); rng.shuffle(allu)
train_u=set(allu[:len(allu)//2]); test_u=allu[len(allu)//2:]

pop=Counter()
for u in train_u: pop.update(seqs[u])
tot=sum(pop.values()); pop_norm={v:c/tot for v,c in pop.items()}
top_pop=[v for v,_ in pop.most_common(TOPP)]; top_set=set(top_pop)

def bin_of(h):
    return "1-2" if h<=2 else "3-5" if h<=5 else "6-10" if h<=10 else "11-25" if h<=25 else "26+"
BINS=["1-2","3-5","6-10","11-25","26+"]

# per (user, bin) -> list of hit@10
rec=defaultdict=__import__("collections").defaultdict
hits={b:{} for b in BINS}      # hits[bin][user] = [0/1,...]
for u in test_u:
    seq=seqs[u]; cnt=Counter()
    for i in range(len(seq)):
        if i>=1 and i<=120:
            tgt=seq[i]; ulen=i; cands=top_set|set(cnt)
            if tgt in cands:
                st=L*pop_norm.get(tgt,0)+(1-L)*(cnt.get(tgt,0)/ulen) if False else (1-L)*pop_norm.get(tgt,0)+L*(cnt.get(tgt,0)/ulen)
                higher=sum(1 for v in cands if ((1-L)*pop_norm.get(v,0)+L*(cnt.get(v,0)/ulen))>st)
                hit=1.0 if higher<10 else 0.0
            else:
                hit=0.0
            b=bin_of(i)
            hits[b].setdefault(u,[]).append(hit)
        cnt[seq[i]]+=1

def perm_p(a,b,n=4000):
    obs=b.mean()-a.mean(); pool=np.concatenate([a,b]); na=len(a); c=0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(pool[na:].mean()-pool[:na].mean())>=abs(obs): c+=1
    return obs,c/n

print(f"test users: {len(test_u)} (model trained on the other {len(train_u)})")
print(f"{'history':>8}{'low_HR@10':>11}{'high_HR@10':>12}{'gap(high-low)':>15}{'p':>9}")
for b in BINS:
    low=np.array([np.mean(v) for u,v in hits[b].items() if grp[u]=='low'])
    high=np.array([np.mean(v) for u,v in hits[b].items() if grp[u]=='high'])
    if len(low)<10 or len(high)<10:
        print(f"{b:>8}  (too few users)"); continue
    obs,p=perm_p(low,high)
    sig="**SIG**" if p<0.05 else ""
    print(f"{b:>8}{low.mean():>11.3f}{high.mean():>12.3f}{obs:>+15.3f}{p:>9.4f}  {sig}")
