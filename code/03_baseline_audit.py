"""03_baseline_audit.py -- Test H1 with simple baselines, audited per SES group.
Leave-one-out: HISTORY=all but last check-in, TARGET=last. HIT@k / MRR per group."""
import pandas as pd, numpy as np
from collections import Counter, defaultdict
DATA = "/sessions/focused-wizardly-heisenberg/mnt/Research_Paper/data"
MIN_CHECKINS, CAP = 5, 50
chk = pd.read_csv(f"{DATA}/processed/checkins_with_income.csv", usecols=["user_id","venue_id"])
ses = pd.read_csv(f"{DATA}/processed/users_ses.csv", usecols=["user_id","ses_group"])
chk = chk.merge(ses, on="user_id", how="inner").sort_index(kind="stable")  # file is time-ordered
seqs  = chk.groupby("user_id", sort=False)["venue_id"].apply(list).to_dict()
group = ses.set_index("user_id")["ses_group"].to_dict()
users = [u for u,s in seqs.items() if len(s) >= MIN_CHECKINS]
pop = Counter(); trans = defaultdict(Counter)
for u in users:
    hist = seqs[u][:-1]; pop.update(hist)
    for a,b in zip(hist[:-1], hist[1:]): trans[a][b] += 1
pop_rank = [v for v,_ in pop.most_common()]
def toplist(model,u):
    hist=seqs[u][:-1]
    if model=="popular": return pop_rank[:CAP]
    base = trans[hist[-1]].most_common() if model=="markov" else Counter(hist).most_common()
    ranked=[v for v,_ in base]; seen=set(ranked); ranked+=[v for v in pop_rank if v not in seen]
    return ranked[:CAP]
def evaluate(model):
    h5={"low":[],"mid":[],"high":[]}; h10={k:[] for k in h5}; rr={k:[] for k in h5}
    for u in users:
        g=group[u]; tgt=seqs[u][-1]; lst=toplist(model,u)
        h5[g].append(1.0 if tgt in lst[:5] else 0.0); h10[g].append(1.0 if tgt in lst[:10] else 0.0)
        rank=lst.index(tgt)+1 if tgt in lst else None; rr[g].append(1.0/rank if rank else 0.0)
    print(f"\n=== {model} ===\n{'group':<6}{'n':>6}{'HR@5':>9}{'HR@10':>9}{'MRR':>9}")
    for g in ["low","mid","high"]:
        print(f"{g:<6}{len(h5[g]):>6}{np.mean(h5[g]):>9.3f}{np.mean(h10[g]):>9.3f}{np.mean(rr[g]):>9.3f}")
    print(f"GAP(high-low): HR@5 {np.mean(h5['high'])-np.mean(h5['low']):+.3f} | "
          f"HR@10 {np.mean(h10['high'])-np.mean(h10['low']):+.3f} | MRR {np.mean(rr['high'])-np.mean(rr['low']):+.3f}")
print(f"users evaluated: {len(users)} (>= {MIN_CHECKINS} check-ins)")
print("group sizes:", {g: sum(1 for u in users if group[u]==g) for g in ['low','mid','high']})
for m in ["popular","markov","userhist"]: evaluate(m)
