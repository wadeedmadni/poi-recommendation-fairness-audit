"""08_figures.py -- publication figures from validated results of scripts 04/06/07."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
FIG="/sessions/focused-wizardly-heisenberg/mnt/Research_Paper/figures"
plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
C={"low":"#C44E52","mid":"#8C8C8C","high":"#4C72B0"}

# ---- Fig 1: per-group HR@10 across the three model paradigms (the "flip") ----
models=["Popularity","Markov","User-history"]
hr={"low":[.020,.227,.442],"mid":[.031,.173,.373],"high":[.036,.199,.405]}
x=np.arange(len(models)); w=0.26
fig,ax=plt.subplots(figsize=(7,4.2))
for i,g in enumerate(["low","mid","high"]):
    ax.bar(x+(i-1)*w, hr[g], w, label=f"{g} income", color=C[g])
ax.set_xticks(x); ax.set_xticklabels(models)
ax.set_ylabel("HR@10 (higher = better)")
ax.set_title("Recommendation accuracy by income group\nPopularity favours high-income; personalization favours low-income")
ax.legend(title="home neighbourhood")
fig.tight_layout(); fig.savefig(f"{FIG}/fig1_audit_by_group.png",dpi=150); plt.close()

# ---- Fig 2: mitigation trade-off (accuracy + SES gap vs lambda) ----
lam=[0,.25,.5,.75,1.0]
acc=[.029,.405,.407,.407,.437]
gap=[.007,-.001,-.001,-.001,-.001]      # MRR gap high-low (>0 = unfair to low)
sig=[True,False,False,False,False]
fig,ax1=plt.subplots(figsize=(7,4.2))
ax1.plot(lam,acc,"o-",color="#4C72B0",label="overall HR@10")
ax1.set_xlabel(r"$\lambda$  (0 = popularity  →  1 = personalization)")
ax1.set_ylabel("overall HR@10",color="#4C72B0"); ax1.tick_params(axis="y",labelcolor="#4C72B0")
ax2=ax1.twinx(); ax2.spines["top"].set_visible(False)
ax2.axhline(0,color="#999",lw=1,ls="--")
ax2.plot(lam,gap,"s-",color="#C44E52",label="SES MRR gap (high−low)")
ax2.scatter([0],[.007],s=140,facecolors="none",edgecolors="#C44E52",lw=2,zorder=5)
ax2.annotate("significant\n(p=0.005)",(0,.007),xytext=(0.12,.006),color="#C44E52",fontsize=9)
ax2.set_ylabel("SES gap in MRR (>0 = unfair to low)",color="#C44E52"); ax2.tick_params(axis="y",labelcolor="#C44E52")
ax1.set_title("Mitigation: a little personalization removes the gap AND raises accuracy")
fig.tight_layout(); fig.savefig(f"{FIG}/fig2_mitigation_tradeoff.png",dpi=150); plt.close()

# ---- Fig 3: mechanism -- mobility repetitiveness by group ----
groups=["low","mid","high"]; ur=[.428,.499,.494]
fig,ax=plt.subplots(figsize=(5.5,4.2))
ax.bar(groups,ur,color=[C[g] for g in groups])
ax.set_ylabel("unique-venue ratio (lower = more repetitive)")
ax.set_title("Why personalization helps low-income users:\nlower-income mobility is more repetitive (p<0.0001)")
ax.set_ylim(0.38,0.52)
ax.plot([0,2],[0.508,0.508],color="k",lw=1); ax.text(1,0.509,"***",ha="center")
fig.tight_layout(); fig.savefig(f"{FIG}/fig3_mechanism_diversity.png",dpi=150); plt.close()
print("saved:", [f"{FIG}/fig1_audit_by_group.png", f"{FIG}/fig2_mitigation_tradeoff.png", f"{FIG}/fig3_mechanism_diversity.png"])
