"""Builds tables (CSV + Markdown) and figures from results/raw_results.json."""
import json, os, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.metrics import roc_curve

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # always work from the project folder, wherever the script is started
sys.stdout.reconfigure(errors="replace")   # avoid crashes on consoles with limited encodings (e.g. cp949)
R = json.load(open("results/raw_results.json", encoding="utf-8"))
os.makedirs("results/figures", exist_ok=True)
NAMES = {"fedavg": "FedAvg", "ppfbxaio": "PPFBXAIO", "pefl": "PEFL", "bppfl": "BPPFL", "tabnet_bcfl": "TabNet-BCFL"}
DS = {"heart": "UCI Heart Disease", "breast_cancer": "Breast Cancer Wisconsin", "diabetes": "Pima Diabetes"}
M = list(NAMES); COL = dict(zip(M, ["#7f7f7f", "#d62728", "#1f77b4", "#2ca02c", "#9467bd"]))
clean = pd.DataFrame([{k: v for k, v in r.items() if k not in ("curve_val", "curve_test", "y_true", "y_prob")} for r in R["clean"]])
att = pd.DataFrame([{k: v for k, v in r.items() if k != "curve_val"} for r in R["attack"]])
abl = pd.DataFrame([{k: v for k, v in r.items() if k != "curve_val"} for r in R["ablation"]])
ms = lambda s, f=3, pct=False: f"{(s.mean()*100 if pct else s.mean()):.{f}f} ± {(s.std(ddof=1)*100 if pct else s.std(ddof=1)):.{f}f}"
md = []

# ---- Table 4.1 clean performance
rows = []
for ds in DS:
    for m in M:
        g = clean[(clean.dataset == ds) & (clean.method == m)]
        rows.append({"Dataset": DS[ds], "Model": NAMES[m], "Accuracy (%)": ms(g.accuracy, 2, True), "Precision": ms(g.precision), "Recall": ms(g.recall),
                     "F1": ms(g.f1), "AUC": ms(g.auc), "Test loss": ms(g.loss), "#features": f"{g.n_features.mean():.1f}"})
T1 = pd.DataFrame(rows); T1.to_csv("results/table_clean_performance.csv", index=False)
md.append("### Table 4.1 – Clean performance (mean ± std, 10 seeds)\n" + T1.to_markdown(index=False))

# ---- significance vs FedAvg (paired, per seed)
rows = []
for ds in DS:
    base = clean[(clean.dataset == ds) & (clean.method == "fedavg")].sort_values("seed")
    for m in M[1:]:
        g = clean[(clean.dataset == ds) & (clean.method == m)].sort_values("seed")
        d_ = g.accuracy.values - base.accuracy.values
        p = wilcoxon(d_).pvalue if np.any(d_ != 0) else 1.0
        rows.append({"Dataset": DS[ds], "Model": NAMES[m], "Mean Δacc vs FedAvg (pp)": f"{d_.mean()*100:+.2f}", "Wilcoxon p": f"{p:.3f}"})
T = pd.DataFrame(rows); T.to_csv("results/table_significance.csv", index=False)
md.append("### Table 4.1b – Paired comparison with FedAvg (clean, accuracy)\n" + T.to_markdown(index=False))

# ---- latency / throughput
rows = []
for m in M:
    g = clean[clean.method == m]
    rows.append({"Model": NAMES[m], "Round latency (ms)": ms(g.latency_ms, 1), "Throughput (tx/s)": ms(g.throughput_tps, 0), "Ledger valid": "n/a (no ledger)" if m == "fedavg" else f"{int(g.ledger_valid.sum())}/{len(g)} runs"})
T2 = pd.DataFrame(rows); T2.to_csv("results/table_latency_throughput.csv", index=False)
md.append("### Table 4.3 – Measured cost (simulation on 1 CPU core; relative comparison only)\n" + T2.to_markdown(index=False))
md.append(f"\nPaillier aggregation matched plaintext aggregation to max abs error {clean[clean.method=='bppfl'].paillier_max_err.max():.2e}.")

# ---- attacks
ATT = ["none", "noise", "label_flip", "stat_opt", "dyn_opt"]
AN = {"none": "Clean", "noise": "Extra noise", "label_flip": "Label-flip (20% clients)", "stat_opt": "STAT-OPT", "dyn_opt": "DYN-OPT"}
rows = []
for ds in DS:
    for m in M:
        r = {"Dataset": DS[ds], "Model": NAMES[m]}
        c = clean[(clean.dataset == ds) & (clean.method == m)]; r[AN["none"]] = ms(c.accuracy, 1, True)
        for a in ATT[1:]:
            g = att[(att.dataset == ds) & (att.method == m) & (att.attack == a)]; r[AN[a]] = ms(g.accuracy, 1, True)
        rows.append(r)
T3 = pd.DataFrame(rows); T3.to_csv("results/table_attacks.csv", index=False)
md.append("### Table 4.2 – Accuracy (%) under poisoning attacks (mean ± std, 10 seeds, 20% malicious clients)\n" + T3.to_markdown(index=False))

# ---- ablation
rows = []
VN = {"full": "Full PPFBXAIO", "lgoa": "w/o LGOA feature selection", "ega": "w/o EGA (explainability weights)", "poc": "w/o PoC validation (ledger contribution check)"}
for ds in DS:
    for v in VN:
        r = {"Dataset": DS[ds], "Variant": VN[v]}
        for a in ["none", "dyn_opt", "label_flip"]:
            g = abl[(abl.dataset == ds) & (abl.variant == v) & (abl.attack == a)]; r[{"none": "Clean", "dyn_opt": "DYN-OPT", "label_flip": "Label-flip"}[a] + " acc (%)"] = ms(g.accuracy, 1, True)
        rows.append(r)
T4 = pd.DataFrame(rows); T4.to_csv("results/table_ablation.csv", index=False)
md.append("### Table 4.4 – Ablation of PPFBXAIO (mean ± std, 10 seeds)\n" + T4.to_markdown(index=False))
open("results/all_tables.md", "w", encoding="utf-8").write("\n\n".join(md))

# ================= figures
# convergence (validation accuracy)
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
for a_, ds in zip(ax, DS):
    for m in M:
        cur = np.array([r["curve_val"] for r in R["clean"] if r["dataset"] == ds and r["method"] == m])
        mu = cur.mean(0) * 100; a_.plot(range(1, len(mu) + 1), mu, label=NAMES[m], color=COL[m], lw=1.6)
    a_.set_title(DS[ds]); a_.set_xlabel("Global round"); a_.set_ylabel("Validation accuracy (%)"); a_.grid(alpha=.3)
ax[0].legend(fontsize=8); plt.tight_layout(); plt.savefig("results/figures/fig1_convergence.png", dpi=160); plt.close()

# ROC curves (seed 0) and confusion matrices (seed 0)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a_, ds in zip(ax, DS):
    for m in M:
        r = [x for x in R["clean"] if x["dataset"] == ds and x["method"] == m and x["seed"] == 0][0]
        fpr, tpr, _ = roc_curve(r["y_true"], r["y_prob"]); a_.plot(fpr, tpr, color=COL[m], label=f"{NAMES[m]} (AUC {r['auc']:.3f})")
    a_.plot([0, 1], [0, 1], "k--", lw=.7); a_.set_title(f"ROC – {DS[ds]} (seed 0)"); a_.set_xlabel("FPR"); a_.set_ylabel("TPR"); a_.legend(fontsize=7); a_.grid(alpha=.3)
plt.tight_layout(); plt.savefig("results/figures/fig2_roc.png", dpi=160); plt.close()

fig, ax = plt.subplots(3, 5, figsize=(14, 8))
for i, ds in enumerate(DS):
    for j, m in enumerate(M):
        r = [x for x in R["clean"] if x["dataset"] == ds and x["method"] == m and x["seed"] == 0][0]
        cm = np.array(r["confusion_matrix"]); a_ = ax[i, j]; a_.imshow(cm, cmap="Blues")
        for (p, q), v in np.ndenumerate(cm): a_.text(q, p, v, ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
        a_.set_xticks([0, 1]); a_.set_yticks([0, 1]); a_.set_title(f"{NAMES[m]}\n{DS[ds]}", fontsize=8)
        if j == 0: a_.set_ylabel("True")
        if i == 2: a_.set_xlabel("Predicted")
plt.tight_layout(); plt.savefig("results/figures/fig3_confusion_matrices.png", dpi=160); plt.close()

# latency / throughput
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
lat = [clean[clean.method == m].latency_ms.mean() for m in M]; tp = [clean[clean.method == m].throughput_tps.mean() for m in M]
ax[0].bar([NAMES[m] for m in M], lat, color=[COL[m] for m in M]); ax[0].set_yscale("log"); ax[0].set_ylabel("Round latency (ms, log)")
ax[1].bar([NAMES[m] for m in M], tp, color=[COL[m] for m in M]); ax[1].set_yscale("log"); ax[1].set_ylabel("Throughput (tx/s, log)")
for a_ in ax: a_.tick_params(axis="x", rotation=25); a_.grid(alpha=.3, axis="y")
plt.tight_layout(); plt.savefig("results/figures/fig4_latency_throughput.png", dpi=160); plt.close()

# attacks
fig, ax = plt.subplots(1, 3, figsize=(16, 3.8)); w_ = 0.16
for a_, ds in zip(ax, DS):
    for i, m in enumerate(M):
        vals = [clean[(clean.dataset == ds) & (clean.method == m)].accuracy.mean() * 100] + [att[(att.dataset == ds) & (att.method == m) & (att.attack == a)].accuracy.mean() * 100 for a in ATT[1:]]
        errs = [clean[(clean.dataset == ds) & (clean.method == m)].accuracy.std() * 100] + [att[(att.dataset == ds) & (att.method == m) & (att.attack == a)].accuracy.std() * 100 for a in ATT[1:]]
        a_.bar(np.arange(5) + (i - 2) * w_, vals, w_, yerr=errs, color=COL[m], label=NAMES[m], capsize=1.5)
    a_.set_xticks(range(5)); a_.set_xticklabels([AN[a].split(" (")[0] for a in ATT], fontsize=8); a_.set_ylim(0, 100); a_.set_title(DS[ds]); a_.set_ylabel("Accuracy (%)"); a_.grid(alpha=.3, axis="y")
ax[0].legend(fontsize=7, ncol=2); plt.tight_layout(); plt.savefig("results/figures/fig5_attacks.png", dpi=160); plt.close()

# ablation
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8)); V = list(VN)
for a_, a in zip(ax, ["none", "dyn_opt", "label_flip"]):
    for j, ds in enumerate(DS):
        vals = [abl[(abl.dataset == ds) & (abl.variant == v) & (abl.attack == a)].accuracy.mean() * 100 for v in V]
        a_.bar(np.arange(4) + (j - 1) * 0.27, vals, 0.27, label=DS[ds])
    a_.set_xticks(range(4)); a_.set_xticklabels(["Full", "w/o LGOA", "w/o EGA", "w/o PoC"]); a_.set_ylim(0, 118); a_.set_title({"none": "Clean", "dyn_opt": "DYN-OPT", "label_flip": "Label-flip"}[a]); a_.set_ylabel("Accuracy (%)"); a_.grid(alpha=.3, axis="y")
ax[0].legend(fontsize=7, ncol=3, loc="upper center"); plt.tight_layout(); plt.savefig("results/figures/fig6_ablation.png", dpi=160); plt.close()
print("\n\n".join(md))
