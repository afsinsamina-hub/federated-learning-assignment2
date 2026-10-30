"""Collects every number used in the report from results/ into results/report_data.json (+ figures)."""
import os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
df = pd.read_csv("results/all_runs.csv"); main = df[df.group == "main"]; abl = df[df.group == "ablation"]
ms = lambda s, f=3: f"{s.mean():.{f}f} ± {s.std(ddof=1):.{f}f}"
ORDER = ["FedAvg", "PPFBXAIO", "PEFL", "BPPFL", "TabNet-BCFL", "TabNet-BCFL (no DP)"]
DS = {"heart": "Heart Disease", "breast_cancer": "Breast Cancer", "diabetes": "Pima Diabetes"}; AT = ["none", "label_flip", "noise", "stat_opt", "dyn_opt"]
D = {}
D["clean"] = [[DS[d], m] + [ms(main[(main.dataset == d) & (main.method == m) & (main.attack == "none")][k]) for k in ("acc", "prec", "rec", "f1", "auc")]
              for d in DS for m in ORDER]
for met in ("auc", "acc"):
    g = main.groupby(["method", "attack", "seed"])[met].mean().reset_index()
    D[f"attack_{met}"] = [[m] + [ms(g[(g.method == m) & (g.attack == a)][met]) for a in AT] for m in ORDER]
    for d in DS:
        gg = main[main.dataset == d]
        D[f"attack_{met}_{d}"] = [[m] + [ms(gg[(gg.method == m) & (gg.attack == a)][met]) for a in AT] for m in ORDER]
ABLN = sorted(abl.method.unique(), key=lambda s: (s.split(" / ")[0], s))
g = abl.groupby(["method", "attack", "seed"])["auc"].mean().reset_index()
lat = abl[abl.attack == "none"].groupby("method").latency_ms.mean()
D["ablation"] = [[m] + [ms(g[(g.method == m) & (g.attack == a)].auc) for a in ("none", "label_flip", "dyn_opt")] + [f"{lat[m]:.1f}"] for m in ABLN]
c = main[main.attack == "none"].groupby("method")[["latency_ms", "throughput", "msgs", "blocks"]].mean()
D["cost"] = [[m, f"{c.loc[m].latency_ms:.2f}", f"{c.loc[m].throughput:.1f}", f"{int(c.loc[m].msgs)}", f"{int(c.loc[m].blocks)}"] for m in ORDER]
s = pd.read_csv("results/table_significance.csv")
D["sig"] = [[m] + [f"{s[(s.method == m) & (s.attack == a)].mean_diff_auc.iloc[0]:+.3f} ({'<0.001' if s[(s.method == m) & (s.attack == a)].p_wilcoxon.iloc[0] < 0.001 else '%.3f' % s[(s.method == m) & (s.attack == a)].p_wilcoxon.iloc[0]})" for a in AT] for m in ORDER[1:]]
sw = pd.DataFrame(json.load(open("results/sweep_privacy.json")))
D["sweep"] = [[m, f"{e:g}"] + [ms(sw[(sw.method == m) & (sw.eps == e) & (sw.attack == a)].auc) for a in ("none", "dyn_opt")] for m in ("PEFL", "TabNet-BCFL") for e in sorted(sw[sw.method == m].eps.unique())]
tun = json.load(open("results/tuning.json"))
D["tuning"] = [[k, f"{v:g}", " / ".join(f"{float(a):g}:{b:.3f}" for a, b in tun["table"][k].items())] for k, v in tun["best"].items()]
D["n_runs"] = int(len(df)); D["n_main"] = int(len(main)); D["feat"] = {d: float(main[(main.dataset == d) & (main.method == "PPFBXAIO") & (main.attack == "none")].nfeat.mean()) for d in DS}
json.dump(D, open("results/report_data.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

# figures
os.makedirs("results/figures", exist_ok=True)
cols = dict(zip(ORDER, ["#7f7f7f", "#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#c5b0d5"]))
fig, ax = plt.subplots(figsize=(8, 4.0)); w = 0.8 / len(ORDER)
for i, m in enumerate(ORDER):
    mu = [main[(main.dataset == d) & (main.method == m) & (main.attack == "none")].auc.mean() for d in DS]
    sd = [main[(main.dataset == d) & (main.method == m) & (main.attack == "none")].auc.std() for d in DS]
    ax.bar(np.arange(3) + i * w, mu, w, yerr=sd, capsize=2, label=m, color=cols[m])
ax.set_xticks(np.arange(3) + 0.4 - w / 2); ax.set_xticklabels(list(DS.values())); ax.set_ylabel("Test AUC"); ax.set_ylim(0.3, 1.02); ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
ax.axhline(0.5, color="k", lw=.5, ls=":"); plt.tight_layout(); plt.savefig("results/figures/fig_clean_auc.png", dpi=200); plt.close()
g = main.groupby(["method", "attack", "seed"]).auc.mean().reset_index()
fig, ax = plt.subplots(figsize=(8, 4.0))
for i, m in enumerate(ORDER):
    mu = [g[(g.method == m) & (g.attack == a)].auc.mean() for a in AT]; sd = [g[(g.method == m) & (g.attack == a)].auc.std() for a in AT]
    ax.bar(np.arange(5) + i * w, mu, w, yerr=sd, capsize=2, label=m, color=cols[m])
ax.set_xticks(np.arange(5) + 0.4 - w / 2); ax.set_xticklabels(["no attack", "label flip", "extra noise", "STAT-OPT", "DYN-OPT"]); ax.set_ylabel("Test AUC (3 datasets averaged)")
ax.set_ylim(0.3, 1.02); ax.axhline(0.5, color="k", lw=.5, ls=":"); ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1)); plt.tight_layout(); plt.savefig("results/figures/fig_attacks_auc.png", dpi=200); plt.close()
fig, axs = plt.subplots(1, 2, figsize=(8, 3))
for ax, m in zip(axs, ("PEFL", "TabNet-BCFL")):
    for a, ls in (("none", "-"), ("dyn_opt", "--")):
        q = sw[(sw.method == m) & (sw.attack == a)].groupby("eps").auc.agg(["mean", "std"]); ax.errorbar(q.index, q["mean"], q["std"], ls=ls, marker="o", capsize=2, label={"none": "no attack", "dyn_opt": "DYN-OPT"}[a])
    ax.set_xscale("log"); ax.set_title(m + (" (local eps)" if m == "PEFL" else " (central eps)"), fontsize=9); ax.set_xlabel("privacy budget eps"); ax.set_ylabel("Test AUC"); ax.set_ylim(0.4, 1); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig("results/figures/fig_privacy_sweep.png", dpi=200); plt.close()
print("ok", D["n_runs"])
