"""Learning-rate selection, identical protocol for every method: clean runs, seeds 100-102, 3 datasets,
choose the lr with the best mean validation AUC (test data never used)."""
import os, sys, json, itertools, numpy as np
from multiprocessing import Pool
import fl, methods as M

GRID = {"FedAvg": (lambda lr: M.FedAvg(lr), [0.03, 0.1, 0.3, 1.0, 3.0]),
        "PPFBXAIO": (lambda lr: M.PPFBXAIO(lr), [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]),
        "PEFL": (lambda lr: M.PEFL(lr), [0.1, 0.3, 1.0, 3.0]),
        # BPPFL tuned with crypto/signature/chain switched off: the decrypted aggregate is identical up to 2^-20 (verified in tests), ~100x faster
        "BPPFL": (lambda lr: M.BPPFL(lr, crypto=False, sign=False, chain=False), [0.03, 0.1, 0.3, 1.0, 3.0]),
        "TabNet-BCFL": (lambda lr: M.TabNetBCFL(lr), [1e-3, 3e-3, 1e-2, 3e-2]),
        "TabNet-BCFL-noDP": (lambda lr: M.TabNetBCFL(lr, dp=False), [1e-3, 3e-3, 1e-2, 3e-2])}
SEEDS = [100, 101, 102]; DS = ["heart", "breast_cancer", "diabetes"]

def job(a):
    name, lr, ds, sd = a
    o = fl.run(GRID[name][0](lr), ds, sd, "none"); return name, lr, ds, sd, o["val"]["auc"]

if __name__ == "__main__":
    TUNING = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "tuning.json")
    old = json.load(open(TUNING))["table"] if os.path.exists(TUNING) else {}
    todo = sys.argv[1:] or list(GRID)
    jobs = [(n, lr, ds, sd) for n, (_, lrs) in GRID.items() if n in todo for lr in lrs for ds in DS for sd in SEEDS
            if str(lr) not in old.get(n, {})]
    with Pool(2) as p: res = p.map(job, jobs, chunksize=1)
    tab = {}
    for n, lr, ds, sd, v in res: tab.setdefault(n, {}).setdefault(str(lr), []).append(v)
    tab = {n: {k: float(np.mean(v)) for k, v in t.items()} for n, t in tab.items()}
    for n, t in old.items(): tab.setdefault(n, {}).update({k: v for k, v in t.items() if k not in tab.get(n, {})})
    best = {n: float(max(t, key=lambda k: t[k])) for n, t in tab.items()}
    json.dump(dict(table=tab, best=best),
              open(TUNING, "w"), indent=1)
    print(json.dumps(best), {n: {k: round(v, 3) for k, v in t.items()} for n, t in tab.items()})
