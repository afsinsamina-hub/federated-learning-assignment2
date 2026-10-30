"""Method registry. Learning rates come from results/tuning.json (same tuning protocol for every method)."""
import json, os
import methods as M

_T = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "tuning.json")
BEST = json.load(open(_T))["best"] if os.path.exists(_T) else {}
lr = lambda k, d: BEST.get(k, d)

MAIN = {   # name -> factory
    "FedAvg": lambda: M.FedAvg(lr("FedAvg", 1.0)),
    "PPFBXAIO": lambda: M.PPFBXAIO(lr("PPFBXAIO", 3.0)),
    "PEFL": lambda: M.PEFL(lr("PEFL", 0.3)),
    "TabNet-BCFL": lambda: M.TabNetBCFL(lr("TabNet-BCFL", 1e-3)),
    "TabNet-BCFL (no DP)": lambda: M.TabNetBCFL(lr("TabNet-BCFL-noDP", 3e-2), dp=False),
}
if hasattr(M, "BPPFL"):
    MAIN["BPPFL"] = lambda: M.BPPFL(lr("BPPFL", 1.0))

ABL = {    # ablations: each removes ONE mechanism; everything else (data, rounds, lr, attack) unchanged
    "PPFBXAIO / full": lambda: M.PPFBXAIO(lr("PPFBXAIO", 3.0)),
    "PPFBXAIO / w.o. LGOA": lambda: M.PPFBXAIO(lr("PPFBXAIO", 3.0), lgoa=False),
    "PPFBXAIO / w.o. EDBN (MLP)": lambda: M.PPFBXAIO(lr("FedAvg", 1.0), edbn=False),
    "PPFBXAIO / w.o. SHAP-EGA": lambda: M.PPFBXAIO(lr("PPFBXAIO", 3.0), ega=False),
    "PPFBXAIO / w.o. RAPPOR": lambda: M.PPFBXAIO(lr("PPFBXAIO", 3.0), ldp=False),
    "PPFBXAIO / w.o. blockchain": lambda: M.PPFBXAIO(lr("PPFBXAIO", 3.0), chain=False),
    "PEFL / w.o. anomaly filter": lambda: M.PEFL(lr("PEFL", 0.3), filter=False),
    "PEFL / w.o. LDP noise": lambda: M.PEFL(lr("PEFL", 0.3), ldp=False),
    "PEFL / w.o. LDP noise and filter": lambda: M.PEFL(lr("PEFL", 0.3), ldp=False, filter=False),
    "PEFL / w.o. MFF consensus": lambda: M.PEFL(lr("PEFL", 0.3), consensus=False),
    "BPPFL / w.o. TPE (plain aggregation)": lambda: M.BPPFL(lr("BPPFL", 1.0), crypto=False),
    "BPPFL / w.o. TSA signatures": lambda: M.BPPFL(lr("BPPFL", 1.0), sign=False),
    "BPPFL / w.o. blockchain": lambda: M.BPPFL(lr("BPPFL", 1.0), chain=False),
    "TabNet-BCFL / w.o. DP": lambda: M.TabNetBCFL(lr("TabNet-BCFL", 1e-3), dp=False),
    "TabNet-BCFL / w.o. blockchain": lambda: M.TabNetBCFL(lr("TabNet-BCFL", 1e-3), chain=False),
}
ABL_ATTACKS = ["none", "label_flip", "dyn_opt"]
ATTACKS = ["none", "label_flip", "noise", "stat_opt", "dyn_opt"]
DATASETS = ["heart", "breast_cancer", "diabetes"]
SEEDS = list(range(10))
