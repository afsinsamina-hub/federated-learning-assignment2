"""Run all experiments (resumable). Usage:  python src/run_experiments.py [method names ...]
One JSON line per (dataset, seed, group, method, attack) in results/raw/<dataset>_<seed>.jsonl; finished entries are skipped."""
import os, sys, json, time, traceback
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src")
from multiprocessing import Pool
import config as C, fl

def task(a):
    ds, sd, only = a
    path = f"results/raw/{ds}_{sd}.jsonl"; os.makedirs("results/raw", exist_ok=True)
    done = set()
    if os.path.exists(path): done = {(r["group"], r["method"], r["attack"]) for r in map(json.loads, open(path))}
    data = None
    jobs = [("main", n, f, at) for n, f in C.MAIN.items() for at in C.ATTACKS] + \
           [("ablation", n, f, at) for n, f in C.ABL.items() for at in C.ABL_ATTACKS]
    for g, n, f, at in jobs:
        if (g, n, at) in done or (only and n.split(" / ")[0] not in only): continue
        t = time.time()
        try:
            o = fl.run(f(), ds, sd, at)
        except Exception as e:
            print("FAIL", ds, sd, n, at, repr(e), flush=True); traceback.print_exc(); continue
        rec = dict(dataset=ds, seed=sd, group=g, method=n, attack=at, **json.loads(json.dumps(o, default=float)))
        with open(path, "a") as fh: fh.write(json.dumps(rec) + "\n")
        print(f"{ds} s{sd} {g:8s} {n:32s} {at:10s} auc={o['test']['auc']:.3f} acc={o['test']['accuracy']:.3f} {time.time()-t:.1f}s", flush=True)

if __name__ == "__main__":
    only = set(sys.argv[1:])
    with Pool(2) as p: p.map(task, [(ds, sd, only) for sd in C.SEEDS for ds in C.DATASETS], chunksize=1)
    print("ALL DONE")
