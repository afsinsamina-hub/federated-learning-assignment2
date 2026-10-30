"""
Federated-learning engine shared by ALL methods (identical data, partitions, malicious clients, attacks,
rounds, local epochs, batch size, client sampling and seeds). Only the method-specific parts differ
(model, optimiser, client/server privacy step, aggregation rule, ledger), and those are in methods.py.

A run is: for each round, sample m of K clients, every sampled client trains locally from the global state,
malicious clients replace their update according to the attack (applied AFTER any client-side privacy step,
i.e. an attacker is not bound by the protocol), the server aggregates, and the global model is evaluated.
Clean and attacked runs use the very same method configuration; the attack only changes what malicious clients send.
"""
import time
import numpy as np
import torch, torch.nn as nn
from common import prepare, evaluate
from models import get_vec, set_vec
from attacks import OptAttack, noise_attack

CFG = dict(rounds=50, local_epochs=5, batch=32, clients=10, per_round=4, n_mal=2, noise_mult=10.0)
ATTACKS = ["none", "label_flip", "noise", "stat_opt", "dyn_opt"]


def to_t(X, y=None):
    X = torch.tensor(X, dtype=torch.float32)
    return X if y is None else (X, torch.tensor(y, dtype=torch.float32))


def local_train(model, vec0, X, y, epochs, bs, make_opt, seed, loss_extra=None):
    set_vec(model, vec0); model.train()
    g = torch.Generator().manual_seed(seed)
    opt = make_opt(model.parameters())
    Xt, yt = to_t(X, y); n = len(Xt); bce = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            if len(b) < 2: continue                         # batch-norm needs >= 2 samples
            opt.zero_grad()
            loss = bce(model(Xt[b]), yt[b]) + model.reg
            loss.backward(); opt.step()
    return get_vec(model)


@torch.no_grad()
def predict(model, vec, X):
    set_vec(model, vec); model.eval()
    return torch.sigmoid(model(to_t(X))).numpy()


def metrics(model, vec, X, y):
    p = predict(model, vec, X)
    m = evaluate(y, p)
    pc = np.clip(p, 1e-7, 1 - 1e-7); m["loss"] = float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))
    return m


def run(method, dataset, seed, attack="none", cfg=None, data=None):
    """Run one federated experiment. `method` is an object from methods.py."""
    cfg = {**CFG, **(cfg or {})}
    assert attack in ATTACKS
    data = data or prepare(dataset, seed, cfg["clients"], 0.5)
    rng = np.random.default_rng(seed)
    K, m = cfg["clients"], cfg["per_round"]
    mal = set(np.random.default_rng(10_000 + seed).choice(K, cfg["n_mal"], replace=False).tolist()) if attack != "none" else set()
    atk = OptAttack(attack) if attack in ("stat_opt", "dyn_opt") else None
    method.setup(data, cfg, seed)                                # builds model, feature mask, ledger, accountant ...
    Xs = {k: method.transform(data["Xtr"][data["parts"][k]]) for k in range(K)}
    ys = {k: data["ytr"][data["parts"][k]] for k in range(K)}
    if attack == "label_flip":
        for k in mal: ys[k] = 1 - ys[k]
    gvec = get_vec(method.model)
    log = dict(sel_mal=0, sel_total=0, agg_time=0.0, round_time=0.0, n_updates=0)
    t_all = time.time()
    diverged = False
    for r in range(cfg["rounds"]):
        if not np.all(np.isfinite(gvec)): diverged = True; break
        sel = rng.choice(K, m, replace=False)
        ups, ns, ids, honest = [], [], [], []
        t0 = time.time()
        try:
          for k in sel:
            w = method.local(r, gvec, Xs[k], ys[k], cfg, seed * 100003 + r * 101 + int(k))
            u = method.client_step(w - gvec, rng)
            ups.append(u); ns.append(len(Xs[k])); ids.append(int(k))
        except RuntimeError:                                   # model destroyed (e.g. TabNet masks undefined for NaN/inf weights)
            diverged = True; break
        ups = np.array(ups)
        benign = np.array([u for u, k in zip(ups, ids) if k not in mal]) if mal else ups
        bnorm = np.mean(np.linalg.norm(benign, axis=1)) if len(benign) else 1.0
        for j, k in enumerate(ids):
            if k in mal:
                log["sel_mal"] += 1
                if atk is not None:
                    ups[j] = atk.craft(benign) if len(benign) else -0.5 * ups[j]
                elif attack == "noise":
                    ups[j] = noise_attack(ups[j], bnorm, cfg["noise_mult"], rng)
        log["sel_total"] += len(ids)
        ta = time.time()
        gvec = method.aggregate(r, gvec, ups, np.array(ns), ids, rng)
        log["agg_time"] += time.time() - ta; log["n_updates"] += len(ids)
    log["round_time"] = (time.time() - t_all) / cfg["rounds"]
    if not np.all(np.isfinite(gvec)): diverged = True
    if diverged:                                               # collapsed model: scored as a constant classifier (AUC 0.5)
        def _chance(y):
            m_ = evaluate(y, np.full(len(y), 0.5)); m_["loss"] = float(np.log(2)); return m_
        out = dict(test=_chance(data["yte"]), val=_chance(data["yva"]), latency_ms=1e3 * log["agg_time"] / cfg["rounds"],
                   throughput=log["n_updates"] / max(log["agg_time"], 1e-9), diverged=True, **method.report())
        out["mal_share"] = log["sel_mal"] / max(log["sel_total"], 1); return out
    try:
        te = metrics(method.model, gvec, method.transform(data["Xte"]), data["yte"])
        va = metrics(method.model, gvec, method.transform(data["Xva"]), data["yva"])
    except (RuntimeError, ValueError):                         # weights so extreme that the forward pass is undefined
        def _chance(y):
            m_ = evaluate(y, np.full(len(y), 0.5)); m_["loss"] = float(np.log(2)); return m_
        te, va = _chance(data["yte"]), _chance(data["yva"]); diverged = True
    out = dict(test=te, val=va, latency_ms=1e3 * log["agg_time"] / cfg["rounds"], diverged=diverged,
               throughput=log["n_updates"] / max(log["agg_time"], 1e-9), **method.report())
    out["mal_share"] = log["sel_mal"] / max(log["sel_total"], 1)
    return out
