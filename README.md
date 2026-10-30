# Federated learning with blockchain, privacy and XAI for healthcare

Five methods compared under **one** protocol: FedAvg [McMahan 2017], PPFBXAIO [1], PEFL [2], BPPFL [3], TabNet-BCFL [4] (references in the report).
Datasets: Heart (Cleveland, 303), Breast Cancer Wisconsin (569), Pima Diabetes (768). 10 seeds.

## Run
```
pip install -r requirements.txt
python tests/test_all.py            # 15 unit/integration tests (about 2 min)
python src/tune.py                  # learning-rate selection (clean runs, seeds 100-102, validation AUC only)
python src/run_experiments.py       # all experiments, resumable -> results/raw/*.jsonl
python src/sweep_privacy.py         # privacy-budget sensitivity
python src/make_tables.py           # tables (results/all_tables.md, *.csv) and figures
python src/make_report_data.py && node report/build_report.js   # report (.docx)  [needs: npm i docx]
```

## Code map (src/)
| file | content |
|---|---|
| common.py | datasets, Min-Max on train statistics, stratified split, Dirichlet(0.5) non-IID partition, metrics |
| models.py | MLP; RBM/EDBN (CD-1 + entropy bound); **original TabNet** (pytorch-tabnet: sparsemax attentive transformer, GLU, ghost BN) |
| lgoa.py | Levy Grasshopper Optimisation (Alg. 1, Eqs. 3-15 of [1]) with kNN-CV fitness |
| privacy.py | clipping, Gaussian mechanism, **RDP accountant** (sampled Gaussian, Balle et al. conversion; reproduces TF-Privacy eps=3.01), RAPPOR (eps=2 ln(p/q)) |
| attacks.py | STAT-OPT / DYN-OPT of Shejwalkar & Houmansadr (Min-Max gamma optimisation), noise attack; label flip is applied on data |
| crypto.py | **threshold Paillier** (2048-bit, t=7 of 10, packed fixed point) and **threshold Schnorr** signatures (own implementation of the cited schemes) |
| ledger.py | permissioned chain: SHA-256 block hashing, 4 validators, PBFT 3-phase commit (2f+1), smart-contract validation, tamper detection; proof-of-stake chain for BPPFL |
| fl.py | **shared engine**: identical data, partitions, malicious clients, attacks, rounds, epochs, client sampling for all methods |
| methods.py | FedAvg, PPFBXAIO (LGOA+EDBN+SHAP-EGA+RAPPOR+PoC+PBFT), TabNet-BCFL (DP-FedAvg with RDP-calibrated eps), PEFL (anomaly filter + MFF consensus + LDP), BPPFL (threshold Paillier + Schnorr + PoS chain) |
| config.py / tune.py / run_experiments.py / sweep_privacy.py / make_tables.py | protocol, tuning, experiments, tables |

## Common protocol (all methods, clean and attacked runs)
10 clients, 4 sampled per round, 50 rounds, 5 local epochs, batch 32, Dirichlet alpha=0.5, 2 of 10 clients malicious when attacked
(same clients for every method/attack), identical seeds. Privacy / ledger settings are **identical in clean and attack runs**
(only what malicious clients send changes). Each method keeps its own architecture, optimiser and privacy mechanism, as in its paper;
the learning rate of each is chosen by the same procedure. Diverged models are scored as a constant classifier (AUC 0.5, flagged).

## Implementation choices where a paper is silent (all documented in code docstrings)
- PPFBXAIO: RAPPOR p=0.75 (eps=2.2) on top-k SHAP indicator bits; SHAP reference = mix of previous global-model SHAP and debiased RAPPOR frequencies; SHAP of submitted models is computed on a small trusted reference batch (validation split); LGOA runs on pooled training data as in the paper's pipeline.
- TabNet-BCFL: paper reports eps=6.855 but not the mechanism; we use user-level DP-FedAvg (server clipping C=1, Gaussian noise), noise multiplier calibrated by the RDP accountant to eps=6.855, delta=1e-5.
- BPPFL: Reward() of the incentive contract is unspecified (1 % of stake per accepted update); trusted dealer; seeded non-cryptographic RNG (simulation only).
- PEFL: Gaussian LDP with eps=1, delta=1e-5 and Df = clip bound 0.1 (paper leaves Df open); model is an MLP instead of LeNet/VGG.
- Chain: in-process simulation (no network); PBFT message counts are modelled, not measured on sockets.

## Limitations
- Single-process simulation; no real networking, view changes or smart-contract VM.
- User-level DP with only 10 clients and 14k-parameter TabNet gives near-chance accuracy at eps=6.855 (see results and the privacy sweep).
- BPPFL as published has no robust aggregation and no noise: after decryption it is the uniform mean (FedAvg up to 2^-20 rounding); its value is confidentiality/authentication, which is why its accuracy equals FedAvg's.
- BPPFL runs use real 2048-bit cryptography (about 1.4 s per round); gmpy2 is recommended (pip install gmpy2).
- Latency is measured inside the aggregation step; ledger messages are counted, not sent over a network.
- logs/experiments.log contains 24 FAIL lines written by an earlier engine version (TabNet without DP collapsed to non-finite weights under the extra-noise attack); the engine now scores such runs as a constant classifier and those runs were repeated.
- PEFL: the total privacy loss over a client's uploads is accounted by RDP composition (results key eps_composed, about 5.0 at delta=1e-5); TabNet-BCFL mints one ERC-20-style token per round (contract reward block).
