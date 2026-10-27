"""
Federated-learning simulator (NumPy) for the five compared methods:
  fedavg, ppfbxaio, pefl, bppfl, tabnet_bcfl

Note: these are simplified re-implementations written for this assignment
(no official code was used). Every simplification is listed in README.md / the report.
"""
import time, hashlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from phe import paillier
from common import evaluate

# ----------------------------------------------------------------------------- model
def sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

class Net:
    """1-hidden-layer MLP. attn=True adds an input-dependent softmax feature-attention gate
    (TabNet-style sparse/attentive feature selection, one decision step)."""
    def __init__(self, d, h=16, attn=False):
        self.d, self.h, self.attn = d, h, attn
        self.shapes = [("W1", (d, h)), ("b1", (h,)), ("W2", (h,)), ("b2", ())]
        if attn: self.shapes += [("Wa", (d, d)), ("ba", (d,))]
        self.sizes = [int(np.prod(s)) if s else 1 for _, s in self.shapes]
        self.P = int(sum(self.sizes))

    def init(self, rng):
        w = []
        for (n, s), sz in zip(self.shapes, self.sizes):
            if n == "W1": a = rng.normal(0, np.sqrt(2.0 / self.d), s)
            elif n == "W2": a = rng.normal(0, np.sqrt(1.0 / self.h), s)
            else: a = np.zeros(s)
            w.append(np.asarray(a).ravel())
        return np.concatenate(w)

    def unpack(self, w):
        out, i = {}, 0
        for (n, s), sz in zip(self.shapes, self.sizes):
            out[n] = w[i:i + sz].reshape(s) if s else w[i]
            i += sz
        return out

    def forward(self, w, X):
        p = self.unpack(w); cache = {"X": X}
        if self.attn:
            z = X @ p["Wa"] + p["ba"]; z -= z.max(1, keepdims=True)
            a = np.exp(z); a /= a.sum(1, keepdims=True)
            xm = X * a * self.d; cache.update(a=a)
        else:
            xm = X
        z1 = xm @ p["W1"] + p["b1"]; hdn = np.maximum(z1, 0)
        out = sigmoid(hdn @ p["W2"] + p["b2"])
        cache.update(xm=xm, z1=z1, h=hdn)
        return out, cache

    def predict(self, w, X): return self.forward(w, X)[0]

    def grad(self, w, X, y):
        p = self.unpack(w); n = len(y)
        out, c = self.forward(w, X)
        dz2 = (out - y) / n
        g = {"W2": c["h"].T @ dz2, "b2": dz2.sum()}
        dh = np.outer(dz2, p["W2"]); dz1 = dh * (c["z1"] > 0)
        g["W1"] = c["xm"].T @ dz1; g["b1"] = dz1.sum(0)
        if self.attn:
            dxm = dz1 @ p["W1"].T
            da = dxm * c["X"] * self.d
            dz = c["a"] * (da - (c["a"] * da).sum(1, keepdims=True))
            g["Wa"] = c["X"].T @ dz; g["ba"] = dz.sum(0)
        return np.concatenate([np.asarray(g[n_]).ravel() for n_, _ in self.shapes])

def bce(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def local_train(net, w, X, y, epochs, bs, lr, rng):
    w = w.copy(); n = len(y)
    for _ in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, bs):
            b = idx[s:s + bs]
            w -= lr * net.grad(w, X[b], y[b])
    return w

# ----------------------------------------------------------------------------- ledger
class Ledger:
    """Hash-chained ledger: block_i = SHA256(block_{i-1} || round || hashes of updates || model hash)."""
    def __init__(self):
        self.blocks = []; self.tx = 0
    def add(self, rnd, update_arrays, model):
        prev = self.blocks[-1]["hash"] if self.blocks else "0" * 64
        uh = [hashlib.sha256(np.ascontiguousarray(u).tobytes()).hexdigest() for u in update_arrays]
        mh = hashlib.sha256(np.ascontiguousarray(model).tobytes()).hexdigest()
        payload = (prev + str(rnd) + "".join(uh) + mh).encode()
        self.blocks.append({"round": rnd, "prev": prev, "updates": uh, "model": mh,
                            "hash": hashlib.sha256(payload).hexdigest()})
        self.tx += len(update_arrays) + 1
    def verify(self):
        prev = "0" * 64
        for b in self.blocks:
            payload = (prev + str(b["round"]) + "".join(b["updates"]) + b["model"]).encode()
            if b["prev"] != prev or hashlib.sha256(payload).hexdigest() != b["hash"]: return False
            prev = b["hash"]
        return True

# ----------------------------------------------------------------------------- LGOA feature selection
def levy(d, rng, theta=1.5):
    from math import gamma, sin, pi
    sig = (gamma(1 + theta) * sin(pi * theta / 2) / (gamma((1 + theta) / 2) * theta * 2 ** ((theta - 1) / 2))) ** (1 / theta)
    u = rng.normal(0, sig, d); v = rng.normal(0, 1, d)
    return u / np.abs(v) ** (1 / theta)

def lgoa_select(Xtr, ytr, Xva, yva, rng, pop=10, iters=15):
    """Levy Grasshopper Optimisation for binary feature selection.
    fitness = 0.99*val_error(LogReg) + 0.01*(#selected/d). Only train/val data are used."""
    d = Xtr.shape[1]
    def fit(m):
        if m.sum() == 0: return 1.0
        clf = LogisticRegression(max_iter=300).fit(Xtr[:, m], ytr)
        return 0.99 * (1 - clf.score(Xva[:, m], yva)) + 0.01 * m.sum() / d
    pos = rng.random((pop, d)); fits = np.array([fit(p > 0.5) for p in pos])
    bi = fits.argmin(); best, bfit = pos[bi].copy(), fits[bi]
    cmax, cmin, f, l = 1.0, 1e-4, 0.5, 1.5
    for t in range(iters):
        c = cmax - t * (cmax - cmin) / iters
        new = np.empty_like(pos)
        for i in range(pop):
            diff = pos - pos[i]; dist = np.abs(diff)
            dn = np.linalg.norm(diff, axis=1, keepdims=True) + 1e-9
            s = f * np.exp(-dist / l) - np.exp(-dist)
            S = (c * 0.5 * s * diff / dn); S[i] = 0
            x = c * S.sum(0) + best
            x = x + 0.01 * levy(d, rng) * (x - best)          # Levy-flight step (f = 0.01)
            new[i] = np.clip(x, 0, 1)
        pos = new; fits = np.array([fit(p > 0.5) for p in pos])
        if fits.min() < bfit: bi = fits.argmin(); best, bfit = pos[bi].copy(), fits[bi]
    mask = best > 0.5
    if mask.sum() == 0: mask[:] = True
    return mask

# ----------------------------------------------------------------------------- attacks
def craft_malicious(attack, u_own, honest_us, rng, P):
    if attack == "noise":                      # extra Gaussian noise injection
        return u_own + rng.normal(0, 0.5, P)
    if attack == "stat_opt":                   # static optimisation: fixed-scale reversed update
        return -5.0 * u_own
    if attack == "dyn_opt":                    # dynamic optimisation: reversed mean direction,
        m = np.mean(honest_us, axis=0)         # norm adapted each round to the largest honest norm
        mx = max(np.linalg.norm(u) for u in honest_us)
        return -m / (np.linalg.norm(m) + 1e-12) * mx * 1.5
    return u_own                                # label_flip handled at data level

# ----------------------------------------------------------------------------- main FL loop
def run_fl(method, data, seed, attack="none", ablate=None, cfg=None, paillier_bits=512, lgoa_cache=None, crypto=True):
    cfg = {**dict(rounds=100, local_epochs=5, batch=32, lr=0.1, clients=10, per_round=4,
                  clip=1.0, noise_z=0.3, mal_frac=0.2), **(cfg or {})}
    rng = np.random.default_rng(1000 + seed)
    Xtr, ytr, Xva, yva, Xte, yte = (data[k] for k in ("Xtr", "ytr", "Xva", "yva", "Xte", "yte"))
    parts = data["parts"]; N = len(parts)

    # ---- method set-up
    mask = np.ones(Xtr.shape[1], bool)
    if method == "ppfbxaio" and ablate != "lgoa":
        key = (data["name"], seed)
        if lgoa_cache is not None and key in lgoa_cache: mask = lgoa_cache[key]
        else:
            mask = lgoa_select(Xtr, ytr, Xva, yva, np.random.default_rng(7 + seed))
            if lgoa_cache is not None: lgoa_cache[key] = mask
    Xtr_, Xva_, Xte_ = Xtr[:, mask], Xva[:, mask], Xte[:, mask]
    net = Net(Xtr_.shape[1], h=16, attn=(method == "tabnet_bcfl"))
    P = net.P
    w = net.init(rng)
    n_mal = int(round(cfg["mal_frac"] * N)) if attack != "none" else 0
    malicious = set(rng.permutation(N)[:n_mal].tolist())
    ledger = Ledger() if method != "fedavg" else None
    paillier_err = 0.0
    if method == "bppfl" and crypto: pk, sk = paillier.generate_paillier_keypair(n_length=paillier_bits)
    use_poc = method == "ppfbxaio" and ablate != "poc"
    use_ega = method == "ppfbxaio" and ablate not in ("ega",)
    C, z = cfg["clip"], cfg["noise_z"]
    sd = z * C / np.sqrt(P)                       # per-coordinate noise std (total noise norm ~ z*C)

    curve_val, curve_test, round_ms = [], [], []
    accepted_total = rejected_total = 0
    prev_val_loss = bce(yva, net.predict(w, Xva_))
    total_time = 0.0

    for rnd in range(cfg["rounds"]):
        sel = rng.choice(N, cfg["per_round"], replace=False)
        t0 = time.perf_counter()
        # ---- local training
        raw, ns, is_mal = [], [], []
        for k in sel:
            Xk, yk = Xtr_[parts[k]], ytr[parts[k]].copy()
            if attack == "label_flip" and k in malicious: yk = 1 - yk
            wk = local_train(net, w, Xk, yk, cfg["local_epochs"], cfg["batch"], cfg["lr"], rng)
            raw.append(wk - w); ns.append(len(yk)); is_mal.append(k in malicious)
        honest_us = [u for u, m in zip(raw, is_mal) if not m] or raw
        # ---- client-side privacy mechanism
        subs = []
        for u, m in zip(raw, is_mal):
            if method in ("ppfbxaio", "pefl", "tabnet_bcfl"):
                nrm = np.linalg.norm(u); u = u * min(1.0, C / (nrm + 1e-12))
                if method == "ppfbxaio": u = u + rng.laplace(0, sd / np.sqrt(2), P)   # LDP (Laplace)
                else: u = u + rng.normal(0, sd, P)                                      # Gaussian DP
            subs.append(u)
        # malicious clients craft their submission
        for i, m in enumerate(is_mal):
            if m and attack in ("noise", "stat_opt", "dyn_opt"):
                subs[i] = craft_malicious(attack, raw[i], honest_us, rng, P)
        ns = np.array(ns, float)

        # ---- server / chain side
        if method == "fedavg":
            wts = ns / ns.sum(); agg = sum(a * u for a, u in zip(wts, subs)); w = w + agg

        elif method == "bppfl":
            wts = ns / ns.sum(); plain = sum(a * u for a, u in zip(wts, subs))
            if crypto:
                enc = []
                for a, u in zip(wts, subs):                  # each client encrypts its weighted update
                    enc.append([pk.encrypt(float(x)) for x in (a * u)])
                summed = enc[0]
                for e in enc[1:]: summed = [c1 + c2 for c1, c2 in zip(summed, e)]   # homomorphic sum
                agg = np.array([sk.decrypt(c) for c in summed])
                paillier_err = max(paillier_err, float(np.abs(agg - plain).max()))
            else:
                agg = plain                                   # plaintext-equivalent (used only in attack runs)
            w = w + agg
            ledger.add(rnd, subs, w)

        elif method == "pefl":
            norms = np.array([np.linalg.norm(u) for u in subs]); med = np.median(norms)
            ok = norms <= 3.0 * med                          # parameter anomaly filter
            cand = [k for k in range(N) if k not in sel]
            comm = rng.choice(cand, 3, replace=False)        # validation committee (non-participants)
            def comm_loss(wv): return float(np.median([bce(ytr[parts[c]], net.predict(wv, Xtr_[parts[c]])) for c in comm]))
            base = comm_loss(w)
            for i, u in enumerate(subs):
                if ok[i] and comm_loss(w + u) > base * 1.1 + 0.02: ok[i] = False
            if ok.any():
                wts = ns[ok] / ns[ok].sum(); w = w + sum(a * u for a, u in zip(wts, [s for s, o in zip(subs, ok) if o]))
            accepted_total += int(ok.sum()); rejected_total += int((~ok).sum())
            ledger.add(rnd, subs, w)

        elif method == "tabnet_bcfl":
            wts = ns / ns.sum(); w_new = w + sum(a * u for a, u in zip(wts, subs))
            vl = bce(yva, net.predict(w_new, Xva_))
            if vl <= 1.2 * prev_val_loss: w, prev_val_loss = w_new, vl     # smart-contract deployment gate
            else: rejected_total += 1
            ledger.add(rnd, subs, w)

        elif method == "ppfbxaio":
            vl0 = bce(yva, net.predict(w, Xva_)); score = ns.copy()
            if use_poc:                                       # Proof-of-Contribution
                delta = np.array([max(0.0, vl0 - bce(yva, net.predict(w + u, Xva_))) for u in subs])
                score = delta
            if use_ega:                                       # Explainability-guided weights
                attr = []
                for u in subs:
                    A = net.unpack(u)["W1"]; attr.append(np.abs(A).sum(1))
                attr = np.array(attr); ref = np.median(attr, axis=0)
                cos = attr @ ref / (np.linalg.norm(attr, axis=1) * np.linalg.norm(ref) + 1e-12)
                score = score * np.clip(cos, 0, 1) if use_poc else ns * np.clip(cos, 0, 1)
            if score.sum() <= 0:
                score = np.zeros_like(ns) if use_poc else ns   # PoC: nobody improved -> skip round
            if score.sum() > 0:
                wts = score / score.sum(); w = w + sum(a * u for a, u in zip(wts, subs))
            ledger.add(rnd, subs, w)
        dt = time.perf_counter() - t0
        total_time += dt; round_ms.append(dt * 1000)
        curve_val.append(float(np.mean((net.predict(w, Xva_) >= 0.5) == yva)))
        curve_test.append(float(np.mean((net.predict(w, Xte_) >= 0.5) == yte)))

    prob = net.predict(w, Xte_)
    res = evaluate(yte, prob)
    res.update(loss=bce(yte, prob), latency_ms=float(np.mean(round_ms)),
               throughput_tps=float((ledger.tx if ledger else cfg["rounds"] * cfg["per_round"]) / total_time),
               ledger_valid=(ledger.verify() if ledger else None), n_features=int(mask.sum()),
               rejected=rejected_total, paillier_max_err=paillier_err, curve_val=curve_val, curve_test=curve_test,
               y_true=yte.tolist(), y_prob=[round(float(x), 5) for x in prob])
    return res
