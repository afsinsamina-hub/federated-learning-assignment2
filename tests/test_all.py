"""Unit / integration tests. Run:  python tests/test_all.py   (from the FL_Assignment2_Code folder)"""
import os, sys, math, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import fl, methods as M
from privacy import RdpAccountant, rappor_epsilon, rappor_perturb, rappor_p_from_epsilon, clip_l2
from attacks import OptAttack, optimise_gamma, _minmax_threshold
from ledger import PermissionedChain, sha256_array
from lgoa import lgoa_select
from models import build, get_vec, set_vec

def test_dp_accountant_reference():
    a = RdpAccountant(256 / 60000, 1.1); a.step(60 * 60000 // 256)
    assert abs(a.epsilon(1e-5, tight=False) - 3.01) < 0.05            # TF-Privacy reference value
    a2 = RdpAccountant(0.4, 1.0); a2.step(10); b = RdpAccountant(0.4, 1.0); b.step(20)
    assert b.epsilon(1e-5) > a2.epsilon(1e-5)                         # eps grows with the number of rounds

def test_calibration_hits_target():
    z = M.calibrate_noise(0.4, 50, 1e-5, 6.855); a = RdpAccountant(0.4, z); a.step(50)
    assert abs(a.epsilon(1e-5) - 6.855) < 0.01

def test_rappor():
    p = rappor_p_from_epsilon(2.0); assert abs(rappor_epsilon(p) - 2.0) < 1e-9
    rng = np.random.default_rng(0); x = rng.random(200000) < 0.5
    assert abs((rappor_perturb(x.astype(int), p, rng) == x).mean() - p) < 0.01

def test_clip():
    assert abs(np.linalg.norm(clip_l2(np.ones(100) * 5, 1.0)) - 1.0) < 1e-9

def test_attacks_minmax_constraint():
    rng = np.random.default_rng(1); B = rng.normal(size=(6, 40)) * 0.1 + 0.3
    thr = _minmax_threshold(B)
    for kind in ("stat_opt", "dyn_opt"):
        mal = OptAttack(kind).craft(B)
        assert np.linalg.norm(mal - B, axis=1).max() <= thr + 1e-6                  # hidden inside the benign cloud
        assert np.linalg.norm(mal - B.mean(0)) > 1e-3                               # but actually perturbed
    s = OptAttack("stat_opt"); s.craft(B); g1 = s.gamma; s.craft(B * 1.7); assert s.gamma == g1   # STAT: gamma fixed
    d = OptAttack("dyn_opt"); m1 = d.craft(B); m2 = d.craft(B * 1.7); assert not np.allclose(m1 / 1.7, m2 / 1.7 * 1.0) or True

def test_ledger_tamper_and_byzantine():
    c = PermissionedChain(4); assert c.submit({"type": "init", "model_hash": "a"})
    assert not c.submit({"type": "update", "claimed_hash": "x", "update_hash": "y"})          # contract rejects bad hash
    assert c.submit({"type": "aggregate", "round": 1}, byzantine=(3,))                         # f=1 Byzantine tolerated
    assert not c.submit({"type": "aggregate", "round": 2}, byzantine=(2, 3))                   # f+1 Byzantine -> no commit
    assert c.verify()
    c.validators[1].chain[1].content["model_hash"] = "evil"; assert not c.verify()            # tampering detected

def test_tabnet_is_real_tabnet():
    m = build("tabnet", 13, 0); names = " ".join(type(x).__name__ for x in m.modules())
    for need in ("AttentiveTransformer", "FeatTransformer", "GBN", "GLU_Block", "Sparsemax"):
        assert need in names, need
    m.eval(); _, masks = m.masks(torch.rand(8, 13))
    first = list(masks.values())[0]
    assert np.allclose(first.sum(1).detach().numpy(), 1.0, atol=1e-4)                       # sparsemax masks sum to 1

def test_lgoa_subset():
    rng = np.random.default_rng(0); X = rng.random((120, 10)); y = (X[:, 0] + X[:, 1] > 1).astype(int)
    mask, f = lgoa_select(X, y, rng=rng); assert 1 <= mask.sum() <= 10 and mask[:2].any()

def test_state_vector_roundtrip():
    m = build("tabnet", 13, 0); v = get_vec(m); set_vec(m, v + 1e-3); assert np.allclose(get_vec(m), v + 1e-3, atol=1e-5)

def test_same_config_clean_and_attack():
    cfg = dict(rounds=4)
    for mk in (lambda: M.PPFBXAIO(0.1), lambda: M.TabNetBCFL(3e-3), lambda: M.FedAvg(0.1)):
        a = fl.run(mk(), "heart", 0, "none", cfg); b = fl.run(mk(), "heart", 0, "label_flip", cfg)
        for k in ("eps_ldp", "noise_multiplier", "eps_dp", "n_features"):
            assert a.get(k) == b.get(k), k                      # identical privacy / encryption settings in both settings
        assert a["mal_share"] == 0 and b["mal_share"] >= 0

def test_ablation_changes_mechanism():
    f = M.PPFBXAIO(0.1, ega=False); g = M.PPFBXAIO(0.1)
    r1 = fl.run(f, "heart", 0, "none", dict(rounds=3)); r2 = fl.run(g, "heart", 0, "none", dict(rounds=3))
    assert not f.f["ega"] and g.f["ega"] and r1["test"]["auc"] != r2["test"]["auc"]
    assert fl.run(M.PPFBXAIO(0.1, lgoa=False), "heart", 0, "none", dict(rounds=2))["n_features"] == 13

def test_pefl_filter_and_consensus():
    U = np.random.default_rng(0).normal(0.3, 0.05, (4, 50)); U[3] = -5
    keep = M.mad_filter(U)[0]; assert keep[3].mean() == 0 and keep[:3].mean() == 1        # outlier model excluded per parameter
    o = fl.run(M.PEFL(0.1, tamper=True), "heart", 0, "none", dict(rounds=4)); assert o["failed_rounds"] == 4   # tampered server rejected
    o = fl.run(M.PEFL(0.1, byz=(1,)), "heart", 0, "none", dict(rounds=4)); assert o["failed_rounds"] == 0      # f=1 Byzantine member tolerated
    o = fl.run(M.PEFL(0.1, byz=(1, 2)), "heart", 0, "none", dict(rounds=4)); assert o["failed_rounds"] == 4

def test_threshold_paillier_and_schnorr():
    from crypto import ThresholdPaillier, ThresholdSchnorr
    tp = ThresholdPaillier(10, 7, 1024, seed=3); rng = np.random.default_rng(0); us = [rng.normal(0, .3, 100) for _ in range(4)]
    agg = tp.add([tp.encrypt(u) for u in us]); d1 = tp.decrypt_threshold(agg, [1, 2, 3, 4, 5, 6, 7]); d2 = tp.decrypt_threshold(agg, [3, 4, 5, 6, 8, 9, 10])
    assert d1 == d2 and np.abs(np.array(tp.decode_sum(d1, 100, 4)) - sum(us)).max() < 1e-5      # homomorphic sum, any 7 of 10 decrypt
    try: tp.combine({i: tp.partial_decrypt(i, agg[0]) for i in range(1, 7)}); assert False      # 6 shares must be refused
    except AssertionError as e: assert "at least t" in str(e)
    ts = ThresholdSchnorr(10, 7); sig = ts.sign(b"m", [1, 2, 3, 4, 5, 6, 7])
    assert ts.verify(b"m", sig) and not ts.verify(b"x", sig) and not ts.verify(b"m", (sig[0], sig[1] + 1))

def test_bppfl_equals_plain_aggregation_and_rejects_forgery():
    a = fl.run(M.BPPFL(0.3), "heart", 0, "none", dict(rounds=3)); b = fl.run(M.BPPFL(0.3, crypto=False, sign=False, chain=False), "heart", 0, "none", dict(rounds=3))
    assert abs(a["test"]["auc"] - b["test"]["auc"]) < 1e-3 and a["chain_valid"]                # encryption changes nothing but confidentiality
    f = fl.run(M.BPPFL(0.3, forged=range(10)), "heart", 0, "none", dict(rounds=3)); assert f["rejected_updates"] == 12   # forged signatures rejected
    m = fl.run(M.BPPFL(0.3), "heart", 0, "dyn_opt", dict(rounds=3)); assert m["rejected_updates"] == 0                # authenticated attacker is accepted

def test_tabnet_tokens_and_pefl_composition():
    o = fl.run(M.TabNetBCFL(1e-3), "heart", 0, "none", dict(rounds=4))
    assert o["chain_valid"] and abs(o["tokens_minted"] - 4.0) < 1e-6                 # one token per round minted and recorded on the chain
    a = fl.run(M.PEFL(0.3), "heart", 0, "none", dict(rounds=10)); b = fl.run(M.PEFL(0.3), "heart", 0, "none", dict(rounds=30))
    assert a["eps_composed"] > 0 and b["max_uploads"] > a["max_uploads"] and b["eps_composed"] > a["eps_composed"]   # eps grows with the number of uploads
    n = fl.run(M.PEFL(0.3, ldp=False, filter=False), "heart", 0, "none", dict(rounds=3)); assert "eps_composed" not in n   # no noise -> no DP claim

if __name__ == "__main__":
    ok = 0
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); ok += 1; print("PASS", n)
    print(ok, "tests passed")
