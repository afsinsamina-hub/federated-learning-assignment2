"""Aggregate results/raw/*.jsonl into tables (markdown + csv) and figures."""
import os, glob, json, numpy as np, pandas as pd
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
rows = []
for f in glob.glob("results/raw/*.jsonl"):
    for r in map(json.loads, open(f)):
        t = r["test"]; rows.append(dict(dataset=r["dataset"], seed=r["seed"], group=r["group"], method=r["method"], attack=r["attack"],
            acc=t["accuracy"], prec=t["precision"], rec=t["recall"], f1=t["f1"], auc=t["auc"], loss=t["loss"],
            latency_ms=r["latency_ms"], throughput=r["throughput"], msgs=r.get("pbft_messages", 0), blocks=r.get("blocks", 0),
            eps=r.get("eps_dp", r.get("eps_ldp", r.get("eps_ldp_round", np.nan))), nfeat=r.get("n_features", np.nan),
            chain_ok=r.get("chain_valid", np.nan), failed=r.get("failed_rounds", 0)))
df = pd.DataFrame(rows); df.to_csv("results/all_runs.csv", index=False)
ms = lambda s: f"{s.mean():.3f} ± {s.std(ddof=1):.3f}"
out = []
def table(title, d, idx, col, val):
    p = d.pivot_table(index=idx, columns=col, values=val, aggfunc=ms if False else (lambda s: ms(s)))
    out.append(f"### {title}\n\n" + p.to_markdown() + "\n"); return p

main = df[df.group == "main"]; abl = df[df.group == "ablation"]
for met in ["auc", "acc", "f1"]:
    for atk in ["none", "label_flip", "noise", "stat_opt", "dyn_opt"]:
        p = main[main.attack == atk].pivot_table(index="method", columns="dataset", values=met, aggfunc=lambda s: ms(s))
        p.to_csv(f"results/table_{met}_{atk}.csv")
    # compact: attacks x methods averaged over datasets (per-seed mean over datasets then mean/std over seeds)
    g = main.groupby(["method", "attack", "seed"])[met].mean().reset_index()
    p = g.pivot_table(index="method", columns="attack", values=met, aggfunc=lambda s: ms(s)).reindex(columns=["none", "label_flip", "noise", "stat_opt", "dyn_opt"])
    out.append(f"### Main results, {met.upper()} (mean ± std over 10 seeds; datasets averaged)\n\n" + p.to_markdown() + "\n"); p.to_csv(f"results/table_main_{met}.csv")
    for ds in df.dataset.unique():
        p = main[main.dataset == ds].pivot_table(index="method", columns="attack", values=met, aggfunc=lambda s: ms(s)).reindex(columns=["none", "label_flip", "noise", "stat_opt", "dyn_opt"])
        out.append(f"#### {met.upper()} on {ds}\n\n" + p.to_markdown() + "\n")
for met in ["auc", "acc"]:
    g = abl.groupby(["method", "attack", "seed"])[met].mean().reset_index()
    p = g.pivot_table(index="method", columns="attack", values=met, aggfunc=lambda s: ms(s)).reindex(columns=["none", "label_flip", "dyn_opt"])
    out.append(f"### Ablations, {met.upper()} (mean ± std, 10 seeds, datasets averaged)\n\n" + p.to_markdown() + "\n"); p.to_csv(f"results/table_ablation_{met}.csv")
c = main[main.attack == "none"].groupby("method")[["latency_ms", "throughput", "msgs", "blocks"]].mean().round(2)
out.append("### Server-side latency per round (ms), throughput (updates/s), PBFT messages and blocks per run (clean)\n\n" + c.to_markdown() + "\n"); c.to_csv("results/table_latency.csv")
# significance: paired Wilcoxon over (dataset, seed) vs FedAvg, per attack, on AUC
sig = []
for atk in ["none", "label_flip", "noise", "stat_opt", "dyn_opt"]:
    w = main[main.attack == atk].pivot_table(index=["dataset", "seed"], columns="method", values="auc")
    for m in w.columns:
        if m == "FedAvg" or "FedAvg" not in w: continue
        d = (w[m] - w["FedAvg"]).dropna()
        try: pv = wilcoxon(d).pvalue if (d != 0).any() else 1.0
        except Exception: pv = np.nan
        sig.append(dict(attack=atk, method=m, mean_diff_auc=d.mean(), p_wilcoxon=pv, n=len(d)))
s = pd.DataFrame(sig); s.to_csv("results/table_significance.csv", index=False)
out.append("### Paired Wilcoxon test of AUC vs FedAvg (over dataset x seed pairs)\n\n" + s.round(4).to_markdown(index=False) + "\n")
cl = abl[abl.attack == "none"].groupby("method")[["latency_ms", "msgs", "blocks"]].mean().round(2)
out.append("### Ablations: aggregation-phase latency per round (ms), PBFT/PoS messages and blocks (clean)\n\n" + cl.to_markdown() + "\n"); cl.to_csv("results/table_ablation_latency.csv")
open("results/all_tables.md", "w", encoding="utf-8").write("\n".join(out))

# figure: AUC under each attack, per method (datasets averaged)
g = main.groupby(["method", "attack", "seed"])["auc"].mean().reset_index()
atks = ["none", "label_flip", "noise", "stat_opt", "dyn_opt"]; meths = list(g.method.unique())
fig, ax = plt.subplots(figsize=(9, 4)); w = 0.8 / len(meths)
for i, m in enumerate(meths):
    mu = [g[(g.method == m) & (g.attack == a)].auc.mean() for a in atks]; sd = [g[(g.method == m) & (g.attack == a)].auc.std() for a in atks]
    ax.bar(np.arange(len(atks)) + i * w, mu, w, yerr=sd, label=m, capsize=2)
ax.set_xticks(np.arange(len(atks)) + 0.4 - w / 2); ax.set_xticklabels(atks); ax.set_ylabel("Test AUC"); ax.legend(fontsize=7); ax.set_ylim(0, 1.02)
plt.tight_layout(); plt.savefig("results/figures/auc_attacks.png", dpi=200)
print("tables written;", len(df), "runs")
