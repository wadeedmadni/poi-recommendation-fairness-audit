"""12_matrix.py -- build the protocol-sensitivity matrix from the harness output.
For each (split x target-selection x lambda): SES gap (high-low) in HR@10 and MRR,
with user-level permutation p. Saves matrix + heatmap."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
DATA="/sessions/focused-wizardly-heisenberg/mnt/Research_Paper/data"
FIG="/sessions/focused-wizardly-heisenberg/mnt/Research_Paper/figures"
rng=np.random.default_rng(0)
def perm_p(a,b,n=4000):
    if len(a)<8 or len(b)<8: return np.nan,np.nan
    obs=b.mean()-a.mean(); pool=np.concatenate([a,b]); na=len(a); c=0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(pool[na:].mean()-pool[:na].mean())>=abs(obs): c+=1
    return obs,c/n

def load(split):
    d=pd.read_csv(f"{DATA}/processed/ranks_{split}.csv"); d["split"]=split; return d
df=pd.concat([load("PU"),load("LUO")],ignore_index=True)

def subset(d, sel):
    if sel=="all": return d
    # per (split,user) choose last1 / last10 histlens
    g=d.groupby(["split","user"])["histlen"]
    if sel=="last1":
        keep=g.transform("max"); return d[d["histlen"]==keep]
    thr=g.transform(lambda s: s.nlargest(min(10,len(s))).min()); return d[d["histlen"]>=thr]

cells=[]
for split in ["PU","LUO"]:
    for sel in ["last1","last10","all"]:
        s=subset(df[df.split==split], sel)
        for lam in [0.0,0.5,1.0]:
            sl=s[s.lam==lam]
            per=sl.groupby(["user","ses"])[["hit10","rr"]].mean().reset_index()
            lo=per[per.ses=="low"]; hi=per[per.ses=="high"]
            gh,ph=perm_p(lo["hit10"].values,hi["hit10"].values)
            gr,pr=perm_p(lo["rr"].values,hi["rr"].values)
            cells.append((split,sel,lam,gh,ph,gr,pr))
M=pd.DataFrame(cells,columns=["split","target","lambda","gapHR10","pHR10","gapMRR","pMRR"])
M.to_csv(f"{DATA}/processed/sensitivity_matrix.csv",index=False)

print("PROTOCOL-SENSITIVITY MATRIX  (gap = high-income minus low-income; + = low-SES disadvantaged)")
print(f"{'split':<5}{'target':<8}{'lam':>4}{'gapHR10':>9}{'pHR10':>8}{'gapMRR':>9}{'pMRR':>8}  sig")
for _,r in M.iterrows():
    sig="HR@10*" if r.pHR10<0.05 else ""
    sig+=" MRR*" if r.pMRR<0.05 else ""
    print(f"{r.split:<5}{r.target:<8}{r['lambda']:>4.1f}{r.gapHR10:>+9.3f}{r.pHR10:>8.3f}{r.gapMRR:>+9.3f}{r.pMRR:>8.3f}  {sig}")

# heatmap of gapHR10
piv=M.pivot_table(index=["split","target"],columns="lambda",values="gapHR10")
pp =M.pivot_table(index=["split","target"],columns="lambda",values="pHR10")
fig,ax=plt.subplots(figsize=(6.5,5))
im=ax.imshow(piv.values,cmap="RdBu_r",vmin=-0.025,vmax=0.025,aspect="auto")
ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f"λ={c}" for c in piv.columns])
ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([f"{a}/{b}" for a,b in piv.index])
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        star="*" if pp.values[i,j]<0.05 else ""
        ax.text(j,i,f"{piv.values[i,j]:+.3f}{star}",ha="center",va="center",fontsize=9)
ax.set_title("SES gap in HR@10 across evaluation protocols\n(+red = low-income disadvantaged; * p<0.05)")
ax.set_xlabel("model: λ=0 popularity → λ=1 personalization"); ax.set_ylabel("split / target-selection")
fig.colorbar(im,label="gap (high − low)"); fig.tight_layout(); fig.savefig(f"{FIG}/fig4_sensitivity_matrix.png",dpi=150)
print(f"\nsaved matrix + heatmap")
