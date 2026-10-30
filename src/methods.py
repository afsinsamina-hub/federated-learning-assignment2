"""
The compared methods. Each class implements the same small interface used by fl.run():
    setup / transform / local / client_step / aggregate / report
Every mechanism below is implemented as an algorithm (not as a re-labelled variant of FedAvg); where a paper leaves a
parameter unspecified the choice is documented in the docstring and in the README.
"""
import math
import numpy as np
import torch, torch.nn as nn
import shap
from models import build, get_vec, set_vec, entropy_lower_bound
from fl import local_train, to_t, predict
from lgoa import lgoa_select
from ledger import PermissionedChain, PoSChain, sha256_array
from privacy import RdpAccountant, clip_l2, gaussian_noise, rappor_p_from_epsilon, rappor_perturb, rappor_epsilon

sgd = lambda lr: (lambda p: torch.optim.SGD(p, lr=lr))
adam = lambda lr: (lambda p: torch.optim.Adam(p, lr=lr))


class Base:
    name = "base"
    def __init__(self, lr=0.1): self.lr = lr
    def setup(self, data, cfg, seed):
        self.data, self.cfg, self.seed = data, cfg, seed
        self.d = data["Xtr"].shape[1]
    def transform(self, X): return X
    def client_step(self, u, rng): return u
    def report(self): return {}


class FedAvg(Base):
    """McMahan et al. 2017: plain weighted averaging of full model updates (no privacy, no ledger)."""
    name = "FedAvg"
    def setup(self, data, cfg, seed):
        super().setup(data, cfg, seed); self.model = build("mlp", self.d, seed)
    def local(self, r, g, X, y, cfg, s): return local_train(self.model, g, X, y, cfg["local_epochs"], cfg["batch"], sgd(self.lr), s)
    def aggregate(self, r, g, ups, ns, ids, rng):
        w = ns / ns.sum(); return g + (w[:, None] * ups).sum(0)


# ------------------------------------------------------------------------------------------------ PPFBXAIO
class _W(nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x): return s.m(x).unsqueeze(-1)


def shap_importance(model, vec, bg, ref):
    """Mean |SHAP| per feature (expected-gradients SHAP, Lundberg & Lee 2017 / Erion et al.) of the model given by `vec`."""
    set_vec(model, vec); model.eval()
    sv = shap.GradientExplainer(_W(model), bg).shap_values(ref, nsamples=20)
    return np.abs(np.asarray(sv)).reshape(len(ref), -1).mean(0)


def _cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class PPFBXAIO(Base):
    """Bhardwaj & Sumangali 2025 [1]. Components (flags allow the ablations):
       lgoa  : Levy Grasshopper feature selection (Alg. 1, Eqs. 3-15) on the Min-Max scaled training data;
       edbn  : Entropy Deep Belief Network (stacked RBMs; CD-1 federated pre-training for `pre` rounds, then supervised);
       ega   : SHAP-based explainability-guided aggregation: weight_i ~ n_i * max(cos(phi_i, phi_ref), 0), where phi_i is the
               SHAP importance of the *submitted* model on a small trusted reference batch (server side) and phi_ref mixes the
               previous global model's SHAP importance with the debiased RAPPOR frequencies of the clients' top-k reports;
       ldp   : basic RAPPOR on the top-k SHAP indicator bits sent by clients (eps = 2 ln(p/q));
       chain : permissioned PBFT chain (4 validators) logging updates, aggregates and Proof-of-Contribution rewards
               (reward_i = max(dA_i,0) * n_i / N, Eq. 32), with the contract rejecting updates whose hash does not match.
       Paper does not give: RAPPOR parameters (we use p = 0.75 -> eps = 2.2), the reference definition (our choice above)."""
    name = "PPFBXAIO"
    def __init__(self, lr=0.1, lgoa=True, edbn=True, ega=True, ldp=True, chain=True, p_keep=0.75, pre=5, align="cos"):
        super().__init__(lr); self.f = dict(lgoa=lgoa, edbn=edbn, ega=ega, ldp=ldp, chain=chain); self.p, self.pre, self.align = p_keep, pre, align

    def setup(self, data, cfg, seed):
        super().setup(data, cfg, seed)
        if self.f["lgoa"]:
            self.mask, self.fit = lgoa_select(data["Xtr"], data["ytr"], rng=np.random.default_rng(seed))
        else:
            self.mask, self.fit = np.ones(self.d, bool), None
        dd = int(self.mask.sum())
        self.model = build("edbn" if self.f["edbn"] else "mlp", dd, seed)
        self.n_hid_ok = (dd >= entropy_lower_bound(self.transform(data["Xtr"]))) if self.f["edbn"] else None
        self.ref = to_t(self.transform(data["Xva"][:25])); self.bg = to_t(self.transform(data["Xva"][25:45]))
        self.k = max(1, math.ceil(dd / 2)); self.reports = []; self.eps_ldp = rappor_epsilon(self.p) if self.f["ldp"] else float("inf")
        self.chain = PermissionedChain(4) if self.f["chain"] else None
        if self.chain: self.chain.submit({"type": "init", "model_hash": sha256_array(get_vec(self.model))})
        self.phi_prev = None; self.rewards = {}
        self._rng_t = torch.Generator().manual_seed(seed)

    def transform(self, X): return X[:, self.mask]

    def local(self, r, g, X, y, cfg, s):
        self._g = g
        if self.f["edbn"] and r < self.pre:                       # federated unsupervised pre-training of the RBMs (CD-1)
            set_vec(self.model, g); Xt = to_t(X)
            for _ in range(cfg["local_epochs"]):
                for i in range(0, len(Xt), cfg["batch"]):
                    h = Xt[i:i + cfg["batch"]]
                    for rbm in self.model.rbms:
                        rbm.cd1_step(h, 0.05, self._rng_t); h = rbm.h_prob(h).detach()
            return get_vec(self.model)
        return local_train(self.model, g, X, y, cfg["local_epochs"], cfg["batch"], sgd(self.lr), s)

    def client_step(self, u, rng):
        if self.f["ega"]:                                         # client sends its (LDP-perturbed) top-k SHAP bits
            phi = shap_importance(self.model, self._g + u, self.bg, self.ref)
            bits = np.zeros(len(phi)); bits[np.argsort(-phi)[:self.k]] = 1
            self.reports.append(rappor_perturb(bits, self.p, rng) if self.f["ldp"] else bits)
        return u

    def aggregate(self, r, g, ups, ns, ids, rng):
        w = ns.astype(float)
        if self.f["ega"]:
            p = self.p if self.f["ldp"] else 1.0
            f = np.mean(self.reports, 0); f = np.clip((f - (1 - p)) / (2 * p - 1), 0, 1) if self.f["ldp"] else f
            self.reports = []
            phi_g = shap_importance(self.model, g, self.bg, self.ref)
            nz = lambda a: a / (a.sum() + 1e-12)
            ref = 0.5 * nz(f) + 0.5 * nz(phi_g)
            cen = (lambda v: v - v.mean()) if self.align == "pearson" else (lambda v: v)       # 'pearson': centred cosine (can be negative)
            a = np.array([_cos(cen(shap_importance(self.model, g + u, self.bg, self.ref)), cen(ref)) for u in ups])
            w = ns * np.maximum(a, 0) + 1e-9
        agg = g + (w[:, None] * ups).sum(0) / w.sum()
        if self.chain:
            va_acc = lambda v: float(((predict(self.model, v, self.transform(self.data["Xva"])) >= .5) == self.data["yva"]).mean())
            base = va_acc(g); rew = {}
            for u, k, n in zip(ups, ids, ns):
                h = sha256_array(u)
                self.chain.submit({"type": "update", "round": r, "client": k, "claimed_hash": h, "update_hash": sha256_array(u)})
                rew[str(k)] = max(va_acc(g + u) - base, 0.0) * n / ns.sum()          # Proof of Contribution, Eq. (32)
            self.chain.submit({"type": "aggregate", "round": r, "model_hash": sha256_array(agg)})
            self.chain.submit({"type": "reward", "round": r, "rewards": rew})
        return agg

    def report(self):
        o = dict(n_features=int(self.mask.sum()), lgoa_fitness=self.fit, entropy_bound_ok=self.n_hid_ok, eps_ldp=self.eps_ldp)
        if self.chain: o.update(blocks=len(self.chain.chain), pbft_messages=self.chain.messages, chain_valid=self.chain.verify())
        return o


# ------------------------------------------------------------------------------------------------ TabNet-BCFL
def calibrate_noise(q, T, delta, target_eps):
    """Smallest noise multiplier z whose RDP-accounted (eps, delta) after T rounds is <= target_eps."""
    lo, hi = 0.3, 50.0
    for _ in range(40):
        mid = (lo + hi) / 2; a = RdpAccountant(q, mid); a.step(T)
        lo, hi = (lo, mid) if a.epsilon(delta) <= target_eps else (mid, hi)
    return hi


class TabNetBCFL(Base):
    """Alsalhi/Otoum-style TabNet-BCFL [4]: the original TabNet architecture (pytorch-tabnet), FedAvg aggregation, Adam,
       BCE + TabNet sparsity regulariser, blockchain with hash-verified updates and smart-contract bookkeeping.
       Differential privacy (the paper reports eps = 6.855 but not the mechanism): DP-FedAvg of McMahan et al. 2018 --
       the server clips every received update to L2 norm C, averages and adds N(0, (zC/m)^2) noise; z is calibrated with the
       RDP accountant so that the total (eps, delta)-DP after all rounds equals the paper's eps (delta = 1e-5, user-level).
       Clipping is applied by the server, so an attacker cannot bypass it."""
    name = "TabNet-BCFL"
    def __init__(self, lr=3e-3, dp=True, chain=True, clip=1.0, target_eps=6.855, delta=1e-5):
        super().__init__(lr); self.dp, self.use_chain, self.C, self.eps_t, self.delta = dp, chain, clip, target_eps, delta

    def setup(self, data, cfg, seed):
        super().setup(data, cfg, seed); self.model = build("tabnet", self.d, seed)
        q = cfg["per_round"] / cfg["clients"]
        self.z = calibrate_noise(q, cfg["rounds"], self.delta, self.eps_t) if self.dp else 0.0
        self.acc = RdpAccountant(q, self.z) if self.dp else None
        self.chain = PermissionedChain(4) if self.use_chain else None
        if self.chain: self.chain.submit({"type": "init", "model_hash": sha256_array(get_vec(self.model))})

    def local(self, r, g, X, y, cfg, s): return local_train(self.model, g, X, y, cfg["local_epochs"], cfg["batch"], adam(self.lr), s)

    def aggregate(self, r, g, ups, ns, ids, rng):
        if self.chain:
            for u, k in zip(ups, ids):
                h = sha256_array(u); self.chain.submit({"type": "update", "round": r, "client": k, "claimed_hash": h, "update_hash": sha256_array(u)})
        if self.dp:
            cl = np.array([clip_l2(u, self.C) for u in ups])
            agg = g + (cl.sum(0) + gaussian_noise(cl.shape[1], self.z * self.C, rng)) / len(ups)
            self.acc.step()
        else:
            w = ns / ns.sum(); agg = g + (w[:, None] * ups).sum(0)
        if self.chain:
            self.chain.submit({"type": "aggregate", "round": r, "model_hash": sha256_array(agg)})
            # ERC-20-style incentive: one token per round is minted and split among the contributing clients by data share
            self.chain.submit({"type": "reward", "round": r, "rewards": {str(int(k)): float(n) / float(ns.sum()) for k, n in zip(ids, ns)}})
        return agg

    def report(self):
        o = dict(noise_multiplier=self.z, eps_dp=self.acc.epsilon(self.delta) if self.acc else float("inf"), delta=self.delta)
        if self.chain: o.update(blocks=len(self.chain.chain), pbft_messages=self.chain.messages, chain_valid=self.chain.verify(),
                                tokens_minted=float(sum(self.chain.balances().values())))
        return o


# ------------------------------------------------------------------------------------------------ PEFL
def mad_filter(U):
    """Anomaly-parameter detection of PEFL (Tian et al. 2025 [2], Algorithm 1, Eqs. 3-9), applied per model parameter k.
       U : (N, P) matrix of the N received (noisy) models/updates. Distances |w_ik - w_jk| are identical for weights and updates.
       Returns keep (N,P) = Eq. (2) participation mask, plus mu, mu_med, alpha."""
    N, P = U.shape
    D = np.abs(U[:, None, :] - U[None, :, :]).sum(1)                   # Eq. (3): sum_j |w_ik - w_jk|
    dD = D - np.median(D, 0)                                            # Eq. (5)-(6)
    T = np.abs(dD) / (np.abs(dD).sum(0) + 1e-12)                        # Eq. (4): l1 normalisation
    E = np.exp(T); mu = E / E.sum(0)                                    # Eq. (7): softmax
    mu_med = np.median(mu, 0)                                           # Eq. (8)
    s = np.sort(mu, 0); h = max(N // 2, 1)
    alpha = (s[N - h:].mean(0) - s[:h].mean(0)) / (mu_med + 1e-12) + 1  # Eq. (9): top-50% mean minus bottom-50% mean
    keep = mu <= alpha * mu_med                                         # Eq. (2)
    return keep, mu, mu_med, alpha


def mff_aggregate(g, U, keep):
    q = np.maximum(keep.sum(0), 1)                                      # Eq. (10): per-parameter average of participants
    return g + (keep * U).sum(0) / q


class PEFL(Base):
    """Tian et al. 2025 [2]:
       * trainers add Gaussian LDP noise to their model (sigma = sqrt(2 ln(1.25/delta)) * Df / eps, Eq. 1; the paper does not give Df,
         we clip the update to L2 norm Df = C and use eps = 1, delta = 1e-5 as in its CIFAR-10 setting);
       * server: parameter-wise MAD-style anomaly filtering (Algorithm 1) then average of participating parameters (Eq. 10);
       * committee: C = 4 validators run the MFF consensus (Algorithm 2): each member verifies ceil(2N/C) models' participation
         indices, a rotating block proposer recomputes the aggregation from the local models (Eq. 10), and the block (global-model
         hash, indices, consensus result) is committed through PBFT among the committee. If verification fails the round is discarded.
       Not reproduced: Go/HTTP networking, timeouts and offline-member handling (single-process simulation)."""
    name = "PEFL"
    def __init__(self, lr=0.1, eps=1.0, delta=1e-5, clip=0.1, committee=4, consensus=True, filter=True, ldp=True, tamper=False, byz=()):
        super().__init__(lr); self.eps, self.delta, self.C, self.nc = eps, delta, clip, committee
        self.f = dict(consensus=consensus, filter=filter, ldp=ldp); self.tamper, self.byz = tamper, byz

    def setup(self, data, cfg, seed):
        super().setup(data, cfg, seed); self.model = build("mlp", self.d, seed)
        self.sigma = math.sqrt(2 * math.log(1.25 / self.delta)) * self.C / self.eps if self.f["ldp"] else 0.0
        self.chain = PermissionedChain(self.nc) if self.f["consensus"] else None
        if self.chain: self.chain.submit({"type": "init", "model_hash": sha256_array(get_vec(self.model))})
        self.msgs = 0; self.failed = 0; self.participated = 0; self.uploads = {}

    def local(self, r, g, X, y, cfg, s): return local_train(self.model, g, X, y, cfg["local_epochs"], cfg["batch"], sgd(self.lr), s)

    def client_step(self, u, rng):
        if self.f["ldp"]: u = clip_l2(u, self.C) + gaussian_noise(len(u), self.sigma, rng)
        return u

    def aggregate(self, r, g, ups, ns, ids, rng):
        N = len(ups)
        for k in ids: self.uploads[int(k)] = self.uploads.get(int(k), 0) + 1       # a client's uploads all use its own data -> they compose
        if self.f["filter"]: keep, *_ = mad_filter(ups)
        else: keep = np.ones_like(ups, dtype=bool)
        agg = mff_aggregate(g, ups, keep)
        if self.tamper: agg = agg + 1e-3                              # a malicious server alters the global model
        if self.chain:
            C = self.nc; nv = math.ceil(2 * N / C); votes = 0
            for j in range(C):                                          # Algorithm 2, lines 8-12
                sub = rng.choice(N, nv, replace=False)
                kj = mad_filter(ups)[0] if self.f["filter"] else keep
                votes += (j not in self.byz) and np.array_equal(kj[sub], keep[sub])
                self.msgs += C - 1
            prop = (r // max(math.ceil(0.2 * C), 1)) % C                  # block proposer rotates every ceil(0.2C) rounds
            while prop in self.byz and len(self.byz) < C: prop = (prop + 1) % C   # unresponsive/malicious proposer: next member proposes
            agg_hat = mff_aggregate(g, ups, keep)                          # proposer re-aggregates from the local models
            ok = votes > C // 2 and np.allclose(agg_hat, agg) and prop not in self.byz
            self.msgs += C - 1
            if ok:
                ok = self.chain.submit({"type": "aggregate", "round": r, "model_hash": sha256_array(agg), "indices": [int(i) for i in ids],
                                        "proposer": prop, "consensus": "PASS"})
            if not ok: self.failed += 1; return g                      # error request -> round discarded
        self.participated += keep.mean() * 0 + 1
        return agg

    def report(self):
        o = dict(sigma=self.sigma, eps_ldp_round=self.eps if self.f["ldp"] else float("inf"), delta=self.delta, failed_rounds=self.failed)
        kmax = max(self.uploads.values()) if self.uploads else 0
        o["max_uploads"] = kmax
        if self.f["ldp"] and kmax:       # RDP composition of kmax Gaussian mechanisms with noise multiplier z = sigma / Df  (Df = clip bound)
            a_ = RdpAccountant(1.0, self.sigma / self.C); a_.step(kmax); o["eps_composed"] = float(a_.epsilon(self.delta))
        if self.chain: o.update(blocks=len(self.chain.chain), pbft_messages=self.chain.messages + self.msgs, chain_valid=self.chain.verify())
        return o


# ------------------------------------------------------------------------------------------------ BPPFL
class BPPFL(Base):
    """Asad & Otoum 2025 [3] (Algorithm 1): every participant encrypts its update with threshold Paillier (TPE), signs the ciphertext
       message with a t-of-n threshold Schnorr signature (TSA, t = 7 of the 10 participants), the server verifies the signature, multiplies
       the ciphertexts (Eq. 16), t participants jointly decrypt the aggregate (Eqs. 17-18) and the global model is updated as
       w <- w + (1/N) sum_i (w_i - w) (Eq. 31). Updates, signatures, aggregates and rewards are recorded on a Proof-of-Stake chain whose
       validators re-verify every signature (validation contract); the incentive contract credits stake (R_i = 0.01 * s_i per accepted update;
       the paper leaves Reward() unspecified).
       IMPORTANT (property of the paper's design, not of this code): TPE/TSA/blockchain protect confidentiality and authenticity; the paper
       defines no robust aggregation and no differential privacy, so an authenticated malicious participant is aggregated exactly like a
       benign one and, up to fixed-point rounding (2^-20) and uniform instead of size-weighted averaging, BPPFL computes the FedAvg result.
       Flags (ablations): crypto (TPE), sign (TSA), chain (PoS ledger)."""
    name = "BPPFL"
    _keys = None
    def __init__(self, lr=0.1, crypto=True, sign=True, chain=True, t=7, n=10, forged=()):
        super().__init__(lr); self.f = dict(crypto=crypto, sign=sign, chain=chain); self.t, self.nparts, self.forged = t, n, set(forged)

    def setup(self, data, cfg, seed):
        super().setup(data, cfg, seed); self.model = build("mlp", self.d, seed)
        from crypto import ThresholdPaillier, ThresholdSchnorr
        if BPPFL._keys is None or BPPFL._keys[0] != (self.t, self.nparts):
            BPPFL._keys = ((self.t, self.nparts), ThresholdPaillier(self.nparts, self.t, 1024, seed=7), ThresholdSchnorr(self.nparts, self.t, seed=8))
        _, self.tp, self.ts = BPPFL._keys
        self.tp.clipped = 0
        self.chain = PoSChain(self.nparts, quality_gate=(lambda c: c.get("sig") is not None and self.ts.verify(c["msg"].encode(), (int(c["sig"][0]), int(c["sig"][1])))) if self.f["sign"] else None, seed=seed) if self.f["chain"] else None
        if self.chain: self.chain.submit({"type": "init", "model_hash": sha256_array(get_vec(self.model))})
        self.rej = 0; self.rng_c = np.random.default_rng(seed + 5)

    def local(self, r, g, X, y, cfg, s): return local_train(self.model, g, X, y, cfg["local_epochs"], cfg["batch"], sgd(self.lr), s)

    def aggregate(self, r, g, ups, ns, ids, rng):
        parts = list(range(1, self.nparts + 1)); accepted = []; cts = []; sigs = {}
        for u, k in zip(ups, ids):
            ct = self.tp.encrypt(u) if self.f["crypto"] else None                        # client side: Eq. (15)
            h = sha256_array(np.concatenate([[int(c) % (2 ** 63) for c in ct]]).astype(np.int64)) if ct else sha256_array(u)
            msg = f"{k}|{r}|{h}".encode(); ok = True; sig = None
            if self.f["sign"]:
                signers = [int(i) for i in self.rng_c.choice(parts, self.t, replace=False)]
                sig = self.ts.sign(msg, signers)                                           # Eqs. (24)-(25)
                if k in self.forged: sig = (sig[0], (sig[1] + 1) % int(self.ts.q))         # forged / unauthenticated submission
                ok = self.ts.verify(msg, sig)                                              # server check, Eq. (26)
            if self.chain:
                ok_chain = self.chain.submit({"type": "update", "round": r, "client": int(k), "claimed_hash": h, "update_hash": h,
                                              "msg": msg.decode(), "sig": [str(sig[0]), str(sig[1])] if sig else None}) if ok else False
                ok = ok and ok_chain
            if ok: accepted.append(u); cts.append(ct)
            else: self.rej += 1
        if not accepted: return g
        N = len(accepted)
        if self.f["crypto"]:
            agg = self.tp.add(cts)                                                         # Eq. (16)
            dec = self.tp.decrypt_threshold(agg, [int(i) for i in self.rng_c.choice(parts, self.t, replace=False)])   # Eqs. (17)-(18)
            tot = np.array(self.tp.decode_sum(dec, len(g), N))
        else:
            tot = np.sum(accepted, 0)
        new = g + tot / N                                                                  # Eq. (31)
        if self.chain:
            self.chain.submit({"type": "aggregate", "round": r, "model_hash": sha256_array(new)})
            self.chain.submit({"type": "reward", "round": r, "rewards": {str(int(k) % self.nparts): 0.01 * self.chain.stake[int(k) % self.nparts] for k in ids}})
        return new

    def report(self):
        o = dict(rejected_updates=self.rej, fixed_point_clipped=int(self.tp.clipped), key_bits=int(self.tp.n.bit_length()), threshold=self.t)
        if self.chain: o.update(blocks=len(self.chain.chain), pbft_messages=self.chain.messages, chain_valid=self.chain.verify())
        return o
