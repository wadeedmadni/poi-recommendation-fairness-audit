import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
DATA="/sessions/focused-wizardly-heisenberg/mnt/Research_Paper/data/processed"
FIG="/sessions/focused-wizardly-heisenberg/mnt/Research_Paper/figures"
rng=np.random.default_rng(0)
def perm_p(a,b,n=3000):
    if len(a)<8 or len(b)<8: return np.nan,1
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
def grid(puf,luof):
    df=pd.concat([pd.read_csv(puf).assign(split="PU"),pd.read_csv(luof).assign(split="LUO")])
    rows=["PU/all","PU/last10","PU/last1","LUO/all","LUO/last10","LUO/last1"]; lams=[0.0,0.5,1.0]
    G=np.zeros((6,3)); P=np.ones((6,3))
    for r,key in enumerate(rows):
        split,sel=key.split("/"); s=subset(df[df.split==split],sel)
        for c,lam in enumerate(lams):
            per=s[s.lam==lam].groupby(["user","ses"])["hit10"].mean().reset_index()
            lo=per[per.ses=="low"]["hit10"].values; hi=per[per.ses=="high"]["hit10"].values
            g,p=perm_p(lo,hi); G[r,c]=g; P[r,c]=p
    return rows,lams,G,P
rows,lams,Gn,Pn=grid(f"{DATA}/ranks_PU.csv",f"{DATA}/ranks_LUO.csv")
_,_,Ga,Pa=grid(f"{DATA}/ranks_austin_PU.csv",f"{DATA}/ranks_austin_LUO.csv")
fig,axes=plt.subplots(1,2,figsize=(11,4.6))
for ax,(G,P,title) in zip(axes,[(Gn,Pn,"NYC (Foursquare)"),(Ga,Pa,"Austin (Gowalla)")]):
    im=ax.imshow(G,cmap="RdBu_r",vmin=-0.04,vmax=0.04,aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels([f"λ={l}" for l in lams])
    ax.set_yticks(range(6)); ax.set_yticklabels(rows)
    for i in range(6):
        for j in range(3):
            ax.text(j,i,f"{G[i,j]:+.3f}{'*' if P[i,j]<0.05 else ''}",ha="center",va="center",fontsize=8)
    ax.set_title(title); ax.set_xlabel("popularity → personalization")
axes[0].set_ylabel("split / target")
fig.suptitle("SES gap in HR@10 across protocols — and across cities (+red=low-income disadvantaged; * p<0.05)")
fig.tight_layout(); fig.savefig(f"{FIG}/fig5_city_replication.png",dpi=150)
print("saved fig5_city_replication.png")
print("NYC significant cells:", int((Pn<0.05).sum()), "/ 18   Austin significant cells:", int((Pa<0.05).sum()), "/ 18")
