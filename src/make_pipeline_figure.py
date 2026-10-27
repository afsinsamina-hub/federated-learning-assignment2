"""Draws Figure 1 (experimental pipeline) -> results/figures/fig0_pipeline.png"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(11, 4.3)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 4.3)
def box(x, y, w, h, t, fc="#e8eef7", fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc=fc, ec="#333", lw=1)); ax.text(x + w/2, y + h/2, t, ha="center", va="center", fontsize=fs)
def arr(x1, y1, x2, y2): ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, lw=1.1, color="#333"))
box(0.1, 1.6, 1.5, 1.1, "3 datasets\n(Heart, Breast Cancer,\nPima)")
box(1.9, 1.6, 1.5, 1.1, "Preprocessing\nMin–Max (train stats)\n70/15/15 split")
box(3.7, 1.6, 1.5, 1.1, "Dirichlet non-IID\npartition (α = 0.5)\n10 clients")
arr(1.6, 2.15, 1.9, 2.15); arr(3.4, 2.15, 3.7, 2.15)
ax.add_patch(FancyBboxPatch((5.5, 0.35), 3.9, 3.7, boxstyle="round,pad=0.05", fc="#fbfbf3", ec="#888", lw=1, ls="--"))
ax.text(7.45, 3.8, "One communication round (×100)", ha="center", fontsize=9, weight="bold")
box(5.7, 2.8, 1.7, 0.7, "1  Local training\n(4 clients, E = 5)", "#e3f1e3")
box(7.6, 2.8, 1.7, 0.7, "2  Client-side privacy\nclip + noise (per method)", "#e3f1e3")
box(7.6, 1.75, 1.7, 0.7, "3  Attack injection\n(20 % malicious)", "#f7dede")
box(5.7, 1.75, 1.7, 0.7, "4  Server / chain\naggregation or validation", "#e3f1e3")
box(5.7, 0.6, 1.7, 0.8, "5  Hash-chained ledger\n(SHA-256 block)", "#e3f1e3")
box(7.6, 0.6, 1.7, 0.8, "6  Global model\nupdate", "#e3f1e3")
arr(7.4, 3.15, 7.6, 3.15); arr(8.45, 2.8, 8.45, 2.45); arr(7.6, 2.1, 7.4, 2.1); arr(6.55, 1.75, 6.55, 1.4); arr(7.4, 1.0, 7.6, 1.0)
box(9.7, 1.6, 1.2, 1.1, "Evaluation\nAcc, P, R, F1,\nAUC, cost", "#fff0d6")
arr(5.2, 2.15, 5.5, 2.15); arr(9.4, 2.15, 9.7, 2.15)
plt.tight_layout(); plt.savefig("results/figures/fig0_pipeline.png", dpi=170); plt.close()
