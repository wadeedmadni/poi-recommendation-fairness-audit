"""16_compare_cities.py -- does the protocol-sensitivity pattern replicate in Austin?"""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
DATA="/sessions/focused-wizardly-heisenberg/mnt/Research_Paper/data/processed"
FIG="/sessions/focused-wizardly-heisenberg/mnt/Research_Paper/figures"
rng=np.random.default_rng(0)
def perm_p(a,b,n=4000):
    if len(a)<8 or len(b)<8: return np.nan,np.nan
    obs=b.mean()-a.mean(); pool=np.concatenate([a,b]); na=len(a); c=0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(pool[na:].mean()-pool[:na].mean())>=abs(obs): c+=1
    return obs,c/n
def subset(d,sel):
    if sel=="all": return d
    g=d.groupby(["split","user"])["histlen"]
    if sel=="last1": return d[d.histlen==g.transform("max")]
    thr=g.transform(lambda s:s.nlargest(min(10,len(s))).min()); return d[d.histlen>=thr]
def matrix(puf,luof):
    df=pd.concat([pd.read_csv(puf).assign(split="PU"),pd.read_csv(luof).assign(split="LUO")])
    out={}
    for split in ["PU","LUO"]:
        for sel in ["last1","last10","all"]:
            s=subset(df[df.split==split],sel)
            for lam in [0.0,0.5,1.0]:
                per=s[s.lam==lam].groupby(["user","ses"])["hit10"].mean().reset_index()
                lo=per[per.ses=="low"]["hit10"].values; hi=per[per.ses=="high"]["hit10"].values
                g,p=perm_p(lo,hi); out[(split,sel,lam)]=(g,p)
    return out
NYC=matrix(f"{DATA}/ranks_PU.csv",f"{DATA}/ranks_LUO.csv")
AUS=matrix(f"{DATA}/ranks_austin_PU.csv",f"{DATA}/ranks_austin_LUO.csv")
print("REPLICATION: SES gap in HR@10 (+ = low-income disadvantaged); * p<0.05")
print(f"{'cell':<16}{'NYC gap':>10}{'p':>8}{'AUS gap':>10}{'p':>8}")
for split in ["PU","LUO"]:
    for sel in ["last1","last10","all"]:
        for lam in [0.0,0.5,1.0]:
            n=NYC[(split,sel,lam)]; a=AUS[(split,sel,lam)]
            ns="*" if n[1]<0.05 else " "; as_="*" if a[1]<0.05 else " "
            print(f"{split}/{sel:<6}λ{lam:<3}{n[0]:>+9.3f}{ns}{n[1]:>7.3f}{a[0]:>+9.3f}{as_}{a[1]:>7.3f}")
