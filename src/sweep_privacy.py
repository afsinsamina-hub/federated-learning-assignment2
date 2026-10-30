"""Privacy-budget sensitivity (supplementary): PEFL local eps and TabNet-BCFL central eps, clean and DYN-OPT, 10 seeds."""
import os, sys, json
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); sys.path.insert(0, "src")
from multiprocessing import Pool
import config as C, fl, methods as M
CASES = [("PEFL", e, lambda e=e: M.PEFL(C.lr("PEFL", .3), eps=e)) for e in (1, 10, 100, 1000)] + \
        [("TabNet-BCFL", e, lambda e=e: M.TabNetBCFL(C.lr("TabNet-BCFL-noDP", 3e-2), target_eps=e)) for e in (6.855, 50, 500, 5000)]
def task(a):
    ds, sd = a; res = []
    for n, e, f in CASES:
        for at in ("none", "dyn_opt"):
            o = fl.run(f(), ds, sd, at)
            res.append(dict(dataset=ds, seed=sd, method=n, eps=e, attack=at, auc=o["test"]["auc"], acc=o["test"]["accuracy"]))
    return res
if __name__ == "__main__":
    with Pool(2) as p: R = p.map(task, [(d, s) for s in C.SEEDS for d in C.DATASETS], chunksize=1)
    json.dump([r for x in R for r in x], open("results/sweep_privacy.json", "w")); print("done")
