"""Runs every experiment and stores raw results in results/raw_results.json.
   usage: python src/run_experiments.py   (about 20-25 min on 1 CPU)"""
import json, time, sys, os, logging
sys.path.insert(0, os.path.dirname(__file__))
from common import prepare
from fl_sim import run_fl

SEEDS = list(range(10))
DATASETS = ["heart", "breast_cancer", "diabetes"]
METHODS = ["fedavg", "ppfbxaio", "pefl", "bppfl", "tabnet_bcfl"]
ATTACKS = ["noise", "label_flip", "stat_opt", "dyn_opt"]
ABLATIONS = ["full", "lgoa", "ega", "poc"]     # component removed

logging.basicConfig(filename="logs/experiments.log", encoding="utf-8", level=logging.INFO, format="%(asctime)s %(message)s")
log = lambda m: (logging.info(m), print(m, flush=True))
out = {"clean": [], "attack": [], "ablation": []}
cache = {}
t00 = time.time()
for ds in DATASETS:
    for sd in SEEDS:
        data = prepare(ds, sd)
        for m in METHODS:
            r = run_fl(m, data, sd, lgoa_cache=cache)               # full Paillier encryption in clean runs
            out["clean"].append(dict(dataset=ds, seed=sd, method=m, **r))
            log(f"clean {ds} s{sd} {m} acc={r['accuracy']:.3f} auc={r['auc']:.3f} lat={r['latency_ms']:.1f}ms")
            for a in ATTACKS:
                r = run_fl(m, data, sd, attack=a, lgoa_cache=cache, crypto=False)
                r.pop("y_true"); r.pop("y_prob"); r.pop("curve_test")
                out["attack"].append(dict(dataset=ds, seed=sd, method=m, attack=a, **r))
            log(f"attacks {ds} s{sd} {m} done")
        for ab in ABLATIONS:
            for a in ["none", "dyn_opt", "label_flip"]:
                r = run_fl("ppfbxaio", data, sd, attack=a, ablate=(None if ab == "full" else ab), lgoa_cache=cache)
                r.pop("y_true"); r.pop("y_prob"); r.pop("curve_test")
                out["ablation"].append(dict(dataset=ds, seed=sd, variant=ab, attack=a, **r))
        log(f"ablation {ds} s{sd} done  elapsed={time.time()-t00:.0f}s")
        json.dump(out, open("results/raw_results.json", "w", encoding="utf-8"))
log("ALL DONE")
