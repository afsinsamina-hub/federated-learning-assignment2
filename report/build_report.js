// Builds the revised report (Word) from results/report_data.json. Run: node report/build_report.js
const fs = require("fs"), path = require("path");
const d = require("docx");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ShadingType, AlignmentType, HeadingLevel, ImageRun, LevelFormat,
        Footer, PageNumber, BorderStyle } = d;
const ROOT = path.join(__dirname, ".."); const D = JSON.parse(fs.readFileSync(path.join(ROOT, "results/report_data.json"), "utf8"));
const FONT = "Times New Roman";
const mean = s => s.split(" ±")[0];
const AUC = (m, a) => mean(D.attack_auc.find(r => r[0] === m)[["none", "label_flip", "noise", "stat_opt", "dyn_opt"].indexOf(a) + 1]);
const ACC = (m, a) => mean(D.attack_acc.find(r => r[0] === m)[["none", "label_flip", "noise", "stat_opt", "dyn_opt"].indexOf(a) + 1]);
const ABL = (name, i) => mean(D.ablation.find(r => r[0] === name)[i]);

// ---------- helpers
const runs = (t, o = {}) => { // **bold** and _italic_ markup
  const out = []; const re = /(\*\*[^*]+\*\*|__[^_]+__)/g; let last = 0, m;
  while ((m = re.exec(t))) { if (m.index > last) out.push(new TextRun({ text: t.slice(last, m.index), font: FONT, ...o }));
    const s = m[0]; if (s.startsWith("**")) out.push(new TextRun({ text: s.slice(2, -2), bold: true, font: FONT, ...o })); else out.push(new TextRun({ text: s.slice(2, -2), italics: true, font: FONT, ...o })); last = m.index + s.length; }
  if (last < t.length) out.push(new TextRun({ text: t.slice(last), font: FONT, ...o })); return out; };
const P = (t, o = {}) => new Paragraph({ children: runs(t, { size: 22 }), spacing: { after: 110, line: 264 }, alignment: AlignmentType.JUSTIFIED, ...o });
const H1 = t => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, bold: true, font: FONT, size: 28 })], spacing: { before: 260, after: 120 }, keepNext: true });
const H2 = t => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, bold: true, font: FONT, size: 24 })], spacing: { before: 180, after: 90 }, keepNext: true });
const B = t => new Paragraph({ numbering: { reference: "bul", level: 0 }, children: runs(t, { size: 22 }), spacing: { after: 60, line: 260 } });
const CAP = t => new Paragraph({ children: runs(t, { size: 20, italics: true }), spacing: { before: 60, after: 160 }, alignment: AlignmentType.CENTER });
const TCAP = t => new Paragraph({ children: runs(t, { size: 20, bold: true }), spacing: { before: 140, after: 60 }, keepNext: true });
const bd = { style: BorderStyle.SINGLE, size: 4, color: "999999" }; const borders = { top: bd, bottom: bd, left: bd, right: bd };
function table(head, rows, widths, sz = 17, leftAll = false) {
  const tw = widths.reduce((a, b) => a + b, 0);
  const cell = (t, w, hd, first) => new TableCell({ borders, width: { size: w, type: WidthType.DXA }, margins: { top: 40, bottom: 40, left: 70, right: 70 },
    shading: hd ? { fill: "D9E2F3", type: ShadingType.CLEAR, color: "auto" } : undefined,
    children: [new Paragraph({ children: runs(String(t), { size: sz, bold: hd }), alignment: (first || leftAll) ? AlignmentType.LEFT : AlignmentType.CENTER })] });
  return new Table({ width: { size: tw, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: head.map((h, i) => cell(h, widths[i], true, i === 0)) }),
           ...rows.map(r => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, widths[i], false, i === 0)) }))] });
}
const FIG = (f, w, h) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 100 }, children: [new ImageRun({ type: "png", data: fs.readFileSync(path.join(ROOT, "results/figures", f)), transformation: { width: w, height: h },
  altText: { title: f, description: f, name: f } })] });
// math helpers: equations are written as plain formatted text (sub/superscripts) so they display in every viewer
class MathRadical { constructor(o) { this.c = o.children; } }
class MathSubSuperScript { constructor(o) { this.b = o.children; this.s = o.subScript; this.p = o.superScript; } }
const r = t => ({ t });
const sub = (b, s) => [{ t: b }, { t: s, sub: true }]; const sup = (b, s) => [{ t: b }, { t: s, sup: true }];
const simple = n => { const f = flatPre(n); return f.length <= 2 && f.every(x => !/[\s+−\-·/,]/.test(x.t)); };
const flatPre = n => Array.isArray(n) ? n.flatMap(flatPre) : [n];
const frac = (n, dn) => [simple(n) ? n : [{ t: "(" }, n, { t: ")" }], { t: "/" }, simple(dn) ? dn : [{ t: "(" }, dn, { t: ")" }]]; const par = ch => [{ t: "(" }, ch, { t: ")" }];
const sum = (lo, hi, ch) => hi ? [{ t: "∑" }, { t: lo, sub: true }, { t: hi, sup: true }, ...ch] : [{ t: "∑" }, { t: lo, sub: true }, ...ch];
const flat = n => Array.isArray(n) ? n.flatMap(flat) : n instanceof MathRadical ? [{ t: "sqrt(" }, ...flat(n.c), { t: ")" }]
  : n instanceof MathSubSuperScript ? [...flat(n.b), ...flat(n.s).map(x => ({ ...x, sub: true })), ...flat(n.p).map(x => ({ ...x, sup: true }))] : [n];
const EQ = (children, n) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 120 },
  children: [...flat(children).map(x => new TextRun({ text: x.t, font: "Cambria Math", size: 22, italics: false, subScript: !!x.sub, superScript: !!x.sup })),
             new TextRun({ text: `        (${n})`, font: FONT, size: 22 })] });

// ---------- content
const body = [];
const A = (...x) => x.forEach(e => { body.push(e); if (e instanceof Table) body.push(new Paragraph({ spacing: { after: 100 }, children: [] })); });
A(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: "Federated Learning with Blockchain, Privacy and Explainability for Healthcare Data: A Comparison of Five Methods", bold: true, font: FONT, size: 32 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 }, children: runs("**Technical Report – Assignment 2** | Advanced Machine Learning, Fall 2026", { size: 22 }) }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 }, children: runs("**Student:** Kadermeera Sahib Afsin Samina | **Student ID:** 22646207 | **Date:** 30 October 2026", { size: 22 }) }));


A(H1("Abstract"));
A(P(`Federated learning (FL) lets hospitals train a shared model without pooling patient records, but plain FL trusts the aggregator, is exposed to poisoned updates and leaves no audit trail. This report re-implements five FL methods from their papers – FedAvg, PPFBXAIO [1], PEFL [2], BPPFL [3] and TabNet-BCFL [4] – and evaluates them under one protocol on three clinical tabular datasets (Heart Disease, Breast Cancer Wisconsin, Pima Diabetes), 10 clients with Dirichlet non-IID labels, five attack settings and 10 random seeds (${D.n_runs} runs including ablations). Real components are used: the original TabNet network, a Renyi-DP accountant, threshold Paillier encryption with threshold Schnorr signatures (2048-bit), two permissioned/proof-of-stake ledgers, SHAP-based aggregation and a committee consensus. Measured test AUC without attack is ${AUC("FedAvg", "none")} for FedAvg, ${AUC("BPPFL", "none")} for BPPFL (statistically equal), ${AUC("PPFBXAIO", "none")} for PPFBXAIO, ${AUC("PEFL", "none")} for PEFL and ${AUC("TabNet-BCFL", "none")} for TabNet-BCFL at its stated privacy budget (${AUC("TabNet-BCFL (no DP)", "none")} without DP). The accuracy differences come mainly from the privacy mechanisms rather than the architectures: PEFL recovers to ${ABL("PEFL / w.o. LDP noise", 1)} without its local noise. In this small-client setting the robustness mechanisms gave no clear gain over plain averaging under the Min-Max attacks. BPPFL adds confidentiality and authentication at about ${Math.round(parseFloat(D.cost.find(r => r[0] === "BPPFL")[1]))} ms of aggregation latency per round.`));
A(P("**Keywords:** federated learning, blockchain, differential privacy, threshold homomorphic encryption, explainable AI, poisoning attacks, healthcare"));

A(H1("1. Introduction"));
A(P("Hospital records cannot be pooled for legal and ethical reasons, so federated learning [5] trains locally and aggregates only model updates. Remaining weaknesses are the trusted server, poisoned updates, missing audit trails and opaque models. The four methods studied here address them with blockchains, differential privacy (DP), cryptography and explainability. A method's name is not evidence that its mechanisms were reproduced; the aim here is therefore to implement each published mechanism, document every place where the paper is silent, and evaluate all methods under identical conditions."));
A(P("**Contributions.** (i) Re-implementation of the five methods with their original components (Sec. 3). (ii) One evaluation protocol with identical privacy settings in clean and attacked runs, five attack settings, ablations and 10 seeds (Sec. 4–5). (iii) Unit tests that check the mechanisms themselves (Sec. 6). (iv) An honest account of what could and could not be reproduced (Sec. 7)."));

A(H1("2. The methods and their reference algorithms"));
A(P("Table 1 lists each method with the algorithms it is built on. Equation numbers refer to the original papers."));
A(TCAP("Table 1 – Methods, components and the original algorithms they follow"));
A(table(["Method", "Components implemented", "Original algorithms (references)"], [
  ["FedAvg", "Size-weighted averaging of local updates.", "McMahan et al. 2017 [5]"],
  ["PPFBXAIO [1]", "Min-Max scaling; Levy Grasshopper Optimisation (LGOA) feature selection (Alg. 1, Eqs. 3–15); Entropy Deep Belief Network (stacked RBMs, CD-1 pre-training, entropy bound); SHAP explainability-guided aggregation; RAPPOR local DP; Proof of Contribution (Eq. 32); SHA-256 block hashing (Eq. 17) with a PBFT chain.", "Saremi et al. 2017 [18]; Mantegna 1994 [19]; Hinton 2002 [20]; Lundberg & Lee 2017 [23]; Erion et al. 2021 [24]; Erlingsson et al. 2014 [10]; Castro & Liskov 1999 [16]"],
  ["PEFL [2]", "Gaussian local DP on the uploaded model (Eq. 1); parameter-wise anomaly filter (Alg. 1, Eqs. 2–10); committee MFF consensus (Alg. 2) on a PBFT chain.", "Dwork & Roth 2014 [9]; PBFT [16]; Shejwalkar & Houmansadr 2021 [17] (attacks)"],
  ["BPPFL [3]", "Threshold Paillier encryption of updates (Eqs. 14–18); threshold Schnorr signature authentication (Eqs. 19–26); aggregation w + (1/N) sum(w_i – w) (Eq. 31); validation, aggregation and incentive contracts on a proof-of-stake chain (Eq. 8).", "Paillier 1999 [11]; Damgard & Jurik 2001 [12]; Fouque et al. 2000 [13]; Schnorr 1991 [14]; Shamir 1979 [15]"],
  ["TabNet-BCFL [4]", "Original TabNet (sparsemax masks, GLU transformer, ghost batch norm); FedAvg aggregation; Adam; DP on updates with epsilon = 6.855; hash-verified updates and contract bookkeeping on a chain.", "Arik & Pfister 2021 [21] (code [22]); McMahan et al. 2018 [6]; Mironov et al. 2019 [7]; Balle et al. 2020 [8]"],
], [1300, 4900, 3160], 17, true));

A(H1("3. Implementation"));
A(H2("3.1 Shared backbone"));
A(P("Local training is mini-batch gradient descent in PyTorch [29]. The global state vector contains all trainable parameters and the floating-point buffers (e.g. batch-norm statistics), so every method aggregates the complete model. Updates are differences to the global state. The federated protocol (client sampling, malicious clients, attack injection, evaluation) is implemented once (fl.py) and used by all methods."));
A(EQ([sub("w", "t+1"), r(" = "), sub("w", "t"), r(" + "), ...sum("i∈S", "", [frac([sub("n", "i")], [r("n")]), par([sub("w", "i"), r(" − "), sub("w", "t")])])], 1));
A(H2("3.2 Differential privacy and its accounting"));
A(P("Clipping and the Gaussian mechanism follow the sampled-Gaussian analysis of Mironov et al. [7]. For T rounds with client sampling probability q = m/K the accountant sums the Renyi divergence of order alpha over rounds and converts it to (epsilon, delta)-DP with the tight conversion of Balle et al. [8]. The implementation reproduces the TensorFlow-Privacy reference value epsilon = 3.01 (60,000 samples, batch 256, z = 1.1, 60 epochs, delta = 1e-5) in a unit test."));
A(EQ([r("ε = "), sub("min", "α"), par([sub("T ρ", "α"), r(" + ln"), par([frac([r("α−1")], [r("α")])]), r(" − "), frac([r("ln δ + ln α")], [r("α−1")])])], 2));
A(P("RAPPOR keeps each bit with probability p and flips it otherwise; the resulting local epsilon is given by Eq. (3). The Gaussian mechanism of PEFL (Eq. 1 of [2]) uses the noise level of Eq. (4)."));
A(EQ([r("ε = 2 ln"), par([frac([r("p")], [r("1 − p")])])], 3));
A(EQ([r("σ = "), new MathRadical({ children: [r("2 ln(1.25/δ)")] }), r(" Δf / ε")], 4));
A(H2("3.3 PPFBXAIO"));
A(P(`**Feature selection.** LGOA follows Algorithm 1 of [1]: the first grasshopper is random, the others are initialised from a Levy distribution (Mantegna's algorithm, beta = 1.5), positions are updated with the social-force term and the best position T plus a Levy step of scale 0.01 (Eqs. 8–15), with c decreasing from 1 to 1e-5. Fitness is 0.99 x (5-fold CV error of kNN) + 0.01 x (selected fraction). The algorithm selects ${D.feat.heart.toFixed(1)} of 13, ${D.feat.breast_cancer.toFixed(1)} of 30 and ${D.feat.diabetes.toFixed(1)} of 8 features on average; as in the paper's pipeline it runs on the pooled training data (a privacy compromise of the original design).`));
A(EQ([sub("X", "i"), r(" ← "), r("c "), sub("S", "i"), r(" + "), sub("T", "best"), r(" + 0.01 · Lévy,     "), r("c = "), sub("c", "max"), r(" − t "), frac([sub("c", "max"), r(" − "), sub("c", "min")], [r("L")])], 5));
A(P("**Model.** The EDBN is a stack of two Bernoulli RBMs with as many hidden nodes as inputs, pre-trained federatedly with contrastive divergence (CD-1) for the first five rounds and then fine-tuned end to end through a logistic output layer; the entropy lower bound on the hidden width is checked at set-up."));
A(P("**Explainability-guided aggregation (EGA).** The server computes the SHAP importance (expected-gradients SHAP) of every submitted model on a small trusted reference batch (taken from the validation split) and weights it by its agreement with a reference explanation that mixes the previous global model's importance with the debiased RAPPOR top-k reports of the clients. Proof-of-Contribution rewards follow Eq. 32 of [1]."));
A(EQ([sub("ω", "i"), r(" ∝ "), sub("n", "i"), r(" max"), par([r("cos("), sub("φ", "i"), r(", "), sub("φ", "ref"), r(")"), r(", 0")]), r(",     "), sub("R", "i"), r(" = "), frac([r("max(Δ"), sub("A", "i"), r(", 0) "), sub("n", "i")], [r("N")])], 6));
A(P("**Open choices (not given in the paper).** RAPPOR keeps a bit with p = 0.75 (epsilon = 2.2 per reported bit); the alignment is the cosine of importance vectors; the reference is the mixture described above. A pilot on separate seeds (100–102) with a centred (Pearson) alignment was not better overall (validation AUC 0.856/0.666/0.882/0.904 for none/noise/STAT/DYN versus 0.902/0.722/0.768/0.905 for cosine), so the first design was kept."));
A(H2("3.4 PEFL"));
A(P("Trainers upload the model with Gaussian noise (Eq. 1 of [2], epsilon = 1, delta = 1e-5; the paper does not give the sensitivity Delta f, so each update is clipped to norm 0.1 and Delta f = 0.1). The server applies the parameter-wise anomaly filter of Algorithm 1: distances to the other models, l1 normalisation, softmax, and the dynamic threshold alpha_k; a parameter of model i enters the average only if its final deviation is at most alpha_k times the median (Eq. 2)."));
A(EQ([sub("D", "i,k"), r(" = "), ...sum("j≠i", "", [r("|"), sub("w", "i,k"), r(" − "), sub("w", "j,k"), r("|")]), r(",     "), sub("μ", "i,k"), r(" ≤ "), sub("α", "k"), r(" "), sub("μ", "med,k"), r(",     "), sub("α", "k"), r(" = "), frac([sub("σ", "k"), r(" − "), sub("φ", "k")], [sub("μ", "med,k"), r("")]), r(" + 1")], 7));
A(P("The MFF consensus (Algorithm 2) runs among a committee of C = 4 validators: each member re-derives the participation indices of ceil(2N/C) models, a rotating block proposer recomputes the aggregate from the local models and compares it with the server's global model, and the block (model hash, indices, result) is committed through PBFT. A tampered global model is rejected and the round is discarded; the tests show that one faulty member is tolerated and two are not. The paper's LeNet/VGG networks are replaced by the same MLP used for FedAvg."));
A(H2("3.5 BPPFL"));
A(P("A trusted authority deals a (t = 7)-of-(n = 10) threshold Paillier key (2048-bit modulus) and a threshold Schnorr key (2048-bit field, 256-bit subgroup). Each participant encrypts its update in fixed point (2^-20 resolution, 32-bit range, several parameters packed per ciphertext), the cooperating signers produce a collective Schnorr signature on the ciphertext message, and the server verifies it, multiplies the ciphertexts (Eq. 16 of [3]) and seven participants jointly decrypt the sum. The chain is a proof-of-stake ledger of ten nodes in which the proposer is drawn with probability proportional to stake and each validator re-verifies the signature (validation contract); an incentive contract adds stake (the paper leaves Reward() open: 1 % of the stake per accepted update)."));
A(EQ([sub("c", "agg"), r(" = "), new MathSubSuperScript({ children: [r("∏")], subScript: [r("i=1")], superScript: [r("N")] }), sub("c", "i"), r(" mod "), sup("N", "2"), r(",     "), sub("w", "t+1"), r(" = "), sub("w", "t"), r(" + "), frac([r("1")], [r("N")]), ...sum("i", "", [par([sub("w", "i"), r(" − "), sub("w", "t")])])], 8));
A(P("**Property of the design.** TPE protects confidentiality and TSA authenticates participants; the paper defines neither a robust aggregation rule nor noise. After decryption the result is the uniform mean of the submitted updates, so an authenticated malicious participant is aggregated like a benign one (tests in Sec. 6 verify this and that forged signatures are rejected). Hence, up to fixed-point rounding and the uniform instead of size-weighted mean, BPPFL computes FedAvg; the differences between the two in Sec. 5 come from this weighting."));
A(H2("3.6 TabNet-BCFL, chains and tests of the ledger"));
A(P("The model is the original TabNet [21] from the reference implementation [22] (3 decision steps, n_d = n_a = 8, gamma = 1.3, sparsemax masks, ghost batch normalisation, sparsity regulariser), trained with Adam and BCE. The paper reports epsilon = 6.855 but not the mechanism; DP-FedAvg [6] is used: the server clips each update to C = 1, averages and adds Gaussian noise, and the noise multiplier is calibrated by the accountant so that the total after all rounds equals epsilon = 6.855 at delta = 1e-5 (user-level DP). Each update hash is validated by a smart contract before PBFT commit, and an ERC-20-style incentive contract mints one token per round, split among the contributing clients by data share (the paper does not give the reward rule)."));
A(P("The PBFT ledger has four validators, SHA-256 block hashes (the JSON rule of Eq. 17 of [1]), the three PBFT phases with 2f+1 votes, a contract that rejects updates whose hash differs from the claimed one, per-validator chain copies and chain verification. It is an in-process simulation: no sockets, view changes or smart-contract virtual machine."));
A(H2("3.7 Attacks"));
A(P("Two of the ten clients are malicious (the same clients for every method and attack). A malicious client replaces its update after any client-side privacy step, since an attacker is not bound by the protocol. Label flipping trains on inverted labels. The extra-noise attack adds Gaussian noise with a norm ten times the mean benign update norm. STAT-OPT and DYN-OPT follow Shejwalkar and Houmansadr [17]: the malicious update is u_b + gamma p, where u_b is the mean benign update and gamma is the largest value that keeps the update inside the cloud of benign updates (Min-Max criterion); STAT-OPT uses p = –sign(u_b) with gamma fixed after the first round, DYN-OPT uses p = –std of the benign updates with gamma re-optimised each round."));
A(EQ([sub("max", "i"), r("‖"), sub("u", "b"), r(" + γ p − "), sub("u", "i"), r("‖ ≤ "), sub("max", "i,j"), r("‖"), sub("u", "i"), r(" − "), sub("u", "j"), r("‖")], 9));

A(H1("4. Experimental protocol"));
A(P("**Data.** Heart Disease (Cleveland, 303 patients, label 1 = disease) [26], Breast Cancer Wisconsin (569, 1 = malignant) [27], Pima Diabetes (768; impossible zeros treated as missing and imputed with the training median) [28]. Stratified 70/15/15 split; Min-Max scaling and imputation use training statistics only; the test set is never used for training or selection."));
A(P("**Federation.** 10 clients with Dirichlet(0.5) label skew [25], 4 clients sampled per round, 50 rounds, 5 local epochs, batch 32, 2 malicious clients when attacked. Seeds 0–9 (data split, partition, initialisation, sampling, malicious clients and noise all depend on the seed). Each method keeps its own architecture, optimiser and privacy mechanism, as in its paper; the learning rate of each was selected by the same procedure (clean runs, seeds 100–102, mean validation AUC; results/tuning.json). BPPFL's grid was run with encryption switched off, which gives the same aggregate (test in Sec. 6)."));
A(P("**Same conditions everywhere.** The configuration of each method is created once and used for the clean run and all attacks; the privacy and encryption parameters in the result files are identical (asserted by a test). Models that diverge numerically (weights become non-finite) are scored as a constant classifier (AUC 0.5) and flagged; this occurred only for TabNet without DP under the extra-noise attack (30 of 30 runs)."));
A(TCAP("Table 2 – Privacy and security properties of each method as configured"));
A(table(["Method", "Privacy / security mechanism", "Formal guarantee"], [
  ["FedAvg", "none", "none"],
  ["PPFBXAIO", "RAPPOR on top-k SHAP bits; PBFT chain; hash integrity", "epsilon = 2.2 per reported bit (local); no guarantee for the model updates"],
  ["PEFL", "Gaussian LDP on the uploaded model; PBFT committee chain", "(1, 1e-5)-LDP per upload by the paper's formula; about 5.0 (delta 1e-5) over a client's ~26 uploads by RDP composition"],
  ["BPPFL", "2048-bit threshold Paillier; threshold Schnorr; PoS chain", "cryptographic confidentiality of individual updates; no DP"],
  ["TabNet-BCFL", "Server-side clipping + Gaussian noise; PBFT chain", "(6.855, 1e-5)-DP, user level (RDP accountant)"],
], [1500, 4300, 3560], 17, true));

A(H1("5. Results"));
A(P("All values are means ± standard deviations over 10 seeds (3 datasets averaged first where stated). All tables, per-dataset tables and raw runs are in the repository (results/all_tables.md, results/all_runs.csv)."));
A(H2("5.1 Performance without attacks"));
A(TCAP("Table 3 – Clean test performance per dataset (10 seeds)"));
A(table(["Dataset", "Method", "Accuracy", "Precision", "Recall", "F1", "AUC"], D.clean, [1250, 1750, 1260, 1260, 1260, 1260, 1320], 15));
A(FIG("fig_clean_auc.png", 560, 280)); A(CAP("Figure 1 – Clean test AUC per dataset (mean ± std, 10 seeds); the dotted line is chance level."));
A(P(`FedAvg and BPPFL are the most accurate (AUC ${AUC("FedAvg", "none")} and ${AUC("BPPFL", "none")}); a paired Wilcoxon test over the 30 dataset-seed pairs finds no difference (p = 0.64). PPFBXAIO is ${AUC("PPFBXAIO", "none")}, lower than FedAvg (p < 0.001, Table 8), and no method outperforms FedAvg without attacks. PEFL (${AUC("PEFL", "none")}) and TabNet-BCFL (${AUC("TabNet-BCFL", "none")}, chance level) have lower AUC because of their DP noise: PEFL without its local noise reaches ${ABL("PEFL / w.o. LDP noise", 1)} and TabNet-BCFL without DP ${AUC("TabNet-BCFL (no DP)", "none")}. TabNet also scores lower than the MLP on these datasets of a few hundred rows, even without DP. Accuracy, precision, recall and F1 use the fixed threshold 0.5, so they are sensitive to calibration (for PPFBXAIO on Pima Diabetes the EDBN is less well calibrated: AUC 0.776 but accuracy 0.588); AUC is threshold-independent and is used for the comparisons.`));
A(H2("5.2 Robustness to poisoning attacks"));
A(TCAP("Table 4 – Test AUC under attack (2 of 10 clients malicious; 3 datasets averaged, 10 seeds)"));
A(table(["Method", "No attack", "Label flip", "Extra noise", "STAT-OPT", "DYN-OPT"], D.attack_auc, [2100, 1450, 1450, 1450, 1450, 1450], 16));
A(FIG("fig_attacks_auc.png", 560, 280)); A(CAP("Figure 2 – Test AUC under each attack (3 datasets averaged, mean ± std over 10 seeds)."));
A(P(`With 2 of 10 clients malicious, plain averaging is hurt little by the attacks designed to stay inside the benign cloud (FedAvg ${AUC("FedAvg", "stat_opt")} under STAT-OPT and ${AUC("FedAvg", "dyn_opt")} under DYN-OPT versus ${AUC("FedAvg", "none")} clean); label flipping is the most damaging (${AUC("FedAvg", "label_flip")}, with large variance because of a few collapsed runs). BPPFL behaves like FedAvg, as its design implies; its higher label-flip value (${AUC("BPPFL", "label_flip")}) comes from uniform instead of size-weighted averaging, which gives the flipped clients less weight. PPFBXAIO performs at or below the FedAvg level under the attacks (${AUC("PPFBXAIO", "label_flip")}, ${AUC("PPFBXAIO", "noise")}, ${AUC("PPFBXAIO", "stat_opt")}, ${AUC("PPFBXAIO", "dyn_opt")}) and is lower under extra noise and STAT-OPT. Table 5 shows that removing its SHAP-alignment weighting does not lower its AUC. A likely reason (not isolated by a separate experiment) is that importance vectors are non-negative and therefore always have a high cosine similarity, so the weighting hardly separates malicious from benign models; a centred alignment was not better on pilot seeds (Sec. 3.3). PEFL and TabNet-BCFL have the lowest AUC because of their DP noise (Sec. 5.5). PEFL without its local noise reaches the FedAvg level and exceeds it under label flipping (0.889 versus 0.837); the variant without noise and without the filter gives 0.903 / 0.875 / 0.899 (no attack / label flip / DYN-OPT), so the filter adds at most 0.014 AUC (label flip), which is within seed-to-seed variation, and the advantage over FedAvg under label flipping comes mostly from the uniform instead of size-weighted averaging that PEFL shares with BPPFL.`));
A(P("These results should be read with the scale of the experiment in mind: only 4 clients are sampled per round, so 0–3 attackers are present in a round, and the Min-Max attacks are bounded by the spread of the benign updates by construction. A larger federation or a stronger attacker may change the ranking; the code allows both."));
A(H2("5.3 Ablations"));
A(TCAP("Table 5 – Ablations (test AUC, 3 datasets averaged, 10 seeds; last column: aggregation latency per round in ms, clean)"));
A(table(["Variant (one mechanism removed)", "No attack", "Label flip", "DYN-OPT", "ms/round"], D.ablation, [3700, 1400, 1400, 1400, 1460], 15));
A(P("Removing mechanisms of PPFBXAIO does not reduce accuracy – the variants without LGOA, EDBN (replaced by the MLP), RAPPOR or SHAP-EGA are at or above the full method – so the data do not show that any single component contributes accuracy or robustness in this setting; the without-EGA row also shows that the SHAP computation accounts for almost all of its latency (1.8 ms instead of about 177 ms). For PEFL, the anomaly filter lowers accuracy when the LDP noise is present (the noise makes benign parameters look like outliers) and gives no benefit under DYN-OPT; the consensus does not change accuracy because it verifies the server rather than the models. For BPPFL the three ablations leave accuracy unchanged by construction and differ only in cost: encryption accounts for about 1.4 s of the 1.5 s per round. For TabNet-BCFL, removing DP restores usable accuracy; the chain does not change it."));
A(H2("5.4 Computational cost"));
A(TCAP("Table 6 – Aggregation-phase latency per round, throughput and ledger activity (clean runs, mean over datasets and seeds)"));
A(table(["Method", "Latency (ms)", "Updates/s", "Ledger messages / run", "Blocks / run"], D.cost, [2200, 1500, 1500, 2300, 1860], 16));
A(P("Latency is the wall-clock time inside the aggregation step on a 2-core CPU (for BPPFL it includes the clients' encryption and signing, which run in the same step; for PPFBXAIO the clients' own SHAP computation is not included). Ledger messages are counted by the simulation, not measured on a network. The costs reflect the mechanisms: threshold Paillier (BPPFL) and SHAP (PPFBXAIO) dominate, whereas the blockchain bookkeeping itself is cheap in this in-process setting (TabNet-BCFL: 0.5 ms without versus 1.6 ms with the chain)."));
A(H2("5.5 Sensitivity to the privacy budget"));
A(TCAP("Table 7 – Test AUC of the two DP methods for larger privacy budgets (3 datasets averaged, 10 seeds)"));
A(table(["Method", "epsilon", "No attack", "DYN-OPT"], D.sweep, [2400, 1800, 2580, 2580], 16));
A(FIG("fig_privacy_sweep.png", 580, 218)); A(CAP("Figure 3 – Effect of the privacy budget (PEFL: local epsilon per upload; TabNet-BCFL: central epsilon over all rounds)."));
A(P("PEFL recovers its accuracy at epsilon of about 10 without attack and at 100 under DYN-OPT. TabNet-BCFL with DP stays close to chance even at epsilon = 5000: user-level noise on 14,000 parameters with four clients per round dominates, and the clipping at C = 1 limits the progress of Adam updates. User-level DP with 10 clients is therefore not a practical operating point for this network; the TabNet-BCFL (no DP) row isolates the architecture and the chain."));
A(H2("5.6 Statistical comparison with FedAvg"));
A(TCAP("Table 8 – Mean AUC difference to FedAvg (Wilcoxon p, 30 dataset-seed pairs, not corrected for multiple comparisons)"));
A(table(["Method", "No attack", "Label flip", "Extra noise", "STAT-OPT", "DYN-OPT"], D.sig, [1900, 1490, 1490, 1490, 1490, 1500], 15));

A(H1("6. Verification of the implementation"));
A(P("The repository contains 15 automated tests (tests/test_all.py), all passing, that check the mechanisms rather than the accuracy:"));
[
 "RDP accountant reproduces the TensorFlow-Privacy reference epsilon = 3.01; the calibrated noise multiplier gives exactly epsilon = 6.855; clipping bounds the norm; RAPPOR's empirical keep-rate and epsilon match Eq. (3).",
 "STAT-OPT and DYN-OPT updates satisfy the Min-Max constraint and differ from the benign mean; STAT-OPT keeps gamma fixed while DYN-OPT re-optimises it.",
 "Ledger: a bad update hash is rejected by the contract; one Byzantine validator is tolerated and two are not; altering a block on one validator is detected by chain verification.",
 "TabNet contains the attentive transformer, feature transformer, ghost batch norm, GLU and sparsemax modules and its masks sum to one; the state vector round-trips exactly.",
 "LGOA returns a non-empty subset containing the informative features of a synthetic problem.",
 "PEFL: the filter excludes an outlier model parameter-wise; a tampered server model is rejected by the committee in every round; one faulty committee member is tolerated and two stop the consensus.",
 "BPPFL: any 7 of 10 decryption shares give the same plaintext and the homomorphic sum equals the plaintext sum to 1e-5; 6 shares are refused; the Schnorr signature verifies and a modified message or signature fails; the encrypted run equals the plain run (AUC difference < 0.001); forged signatures are rejected; an authenticated attacker is accepted.",
 "The privacy parameters are identical in clean and attacked runs of every method, and each ablation changes the mechanism it names (e.g. the number of features is 13 without LGOA).",
 "TabNet-BCFL mints exactly one token per round on a valid chain; PEFL's composed epsilon grows with the number of uploads and is absent when the noise is removed."
].forEach(t => A(B(t)));

A(H1("7. Discussion and scope"));
A(P("The accuracy gains reported in the original papers were not reproduced here, which is expected because the setting differs from theirs (small tabular datasets, 10 clients, a small MLP instead of image CNNs). The conclusions therefore apply to this setting."));
A(B("**Scope of the comparison.** Two of ten clients are malicious and only 4 clients are sampled per round, so attacks are bounded by construction; larger federations or adaptive attacks were not studied."));
A(B("**Simulation.** Everything runs in one process: the chains and the cryptography are implemented and tested, but there is no network, view change or smart-contract virtual machine."));
A(B("**Parameters the papers leave open** (RAPPOR keep-rate, SHAP reference, PEFL sensitivity, BPPFL reward rule, TabNet-BCFL noise mechanism) are my documented choices (Sec. 3). The privacy guarantees of the methods (user-level, per upload, per bit) are not directly comparable."));
A(H1("8. Conclusion"));
A(P("Re-implementing the five methods from their papers and evaluating them under one protocol gives the following conclusions. Plain FedAvg and BPPFL are the most accurate; BPPFL adds real confidentiality and authentication but, as published, no robust aggregation of its own; the explainability-guided aggregation of PPFBXAIO gave no advantage over plain averaging in this setting; and the formal differential privacy of PEFL and TabNet-BCFL reduces their accuracy on datasets of a few hundred patients, with TabNet-BCFL close to chance level at its stated budget. These are measurements of my implementation in one setting, with every deviation from the papers documented, and the package allows the experiments to be repeated and extended."));

A(H1("9. Reproducibility and resources"));
A(B("**Code and data package** (preprocessing, models, training, attacks, ledgers, cryptography, tests, logs, raw results, tables, figures): submitted with this report (folder FL_Assignment2_Code / archive). Online copy: https://github.com/afsinsamina-hub/federated-learning-assignment2 (updated to this version)."));
A(B("**Reproduce:** pip install -r requirements.txt; python tests/test_all.py; python src/tune.py; python src/run_experiments.py (resumable; several CPU-hours in total for 2,250 runs, most of them for BPPFL); python src/sweep_privacy.py; python src/make_tables.py; python src/make_report_data.py; node report/build_report.js."));
A(B("**Files:** src/common.py (data, split, Dirichlet partition, metrics); models.py (MLP, EDBN, TabNet); lgoa.py; privacy.py; attacks.py; ledger.py; crypto.py; fl.py (shared engine); methods.py; config.py; tune.py; run_experiments.py; sweep_privacy.py; make_tables.py."));
A(B("**Software:** PyTorch [29], pytorch-tabnet [22], SHAP [23, 24], gmpy2 for fast big-integer arithmetic, scikit-learn [30], NumPy [31], pandas [32], SciPy [33], Matplotlib [34]. The threshold Paillier and threshold Schnorr code is my own implementation of the cited schemes; no source code of the original papers was used."));
A(B("**Use of AI tools:** the code and the report were developed with the assistance of an AI coding tool. All numbers in this report were produced by running this code (logs/experiments.log); accuracy, AUC and the other metrics reproduce with the same seeds and package versions, while latency and throughput depend on the machine."));

A(H1("References"));
const REFS = [
 "T. Bhardwaj and K. Sumangali, \"An explainable federated blockchain framework with privacy-preserving AI optimization for securing healthcare data,\" Scientific Reports, vol. 15, art. 21799, 2025. doi:10.1038/s41598-025-04083-4.",
 "L. Tian, F. Lin, J. Gan, R. Jia, Z. Zheng and M. Li, \"PEFL: Privacy-preserved and efficient federated learning with blockchain,\" IEEE Internet of Things Journal, vol. 12, no. 3, pp. 3305–3317, 2025. doi:10.1109/JIOT.2024.3479328.",
 "M. Asad and S. Otoum, \"BPPFL: a blockchain-based framework for privacy-preserving federated learning,\" Cluster Computing, vol. 28, art. 126, 2025. doi:10.1007/s10586-024-04834-4.",
 "Y. Otoum, C. Hu, E. H. Said and A. Nayak, \"Enhancing heart disease prediction with federated learning and blockchain integration,\" Future Internet, vol. 16, no. 10, p. 372, 2024. doi:10.3390/fi16100372.",
 "H. B. McMahan, E. Moore, D. Ramage, S. Hampson and B. Agüera y Arcas, \"Communication-efficient learning of deep networks from decentralized data,\" in Proc. AISTATS, PMLR 54, pp. 1273–1282, 2017.",
 "H. B. McMahan, D. Ramage, K. Talwar and L. Zhang, \"Learning differentially private recurrent language models,\" in Proc. ICLR, 2018.",
 "I. Mironov, K. Talwar and L. Zhang, \"Renyi differential privacy of the sampled Gaussian mechanism,\" arXiv:1908.10530, 2019.",
 "B. Balle, G. Barthe, M. Gaboardi, J. Hsu and T. Sato, \"Hypothesis testing interpretations and Renyi differential privacy,\" in Proc. AISTATS, PMLR 108, pp. 2496–2506, 2020.",
 "C. Dwork and A. Roth, \"The algorithmic foundations of differential privacy,\" Foundations and Trends in Theoretical Computer Science, vol. 9, no. 3–4, pp. 211–407, 2014.",
 "Ú. Erlingsson, V. Pihur and A. Korolova, \"RAPPOR: Randomized aggregatable privacy-preserving ordinal response,\" in Proc. ACM CCS, pp. 1054–1067, 2014.",
 "P. Paillier, \"Public-key cryptosystems based on composite degree residuosity classes,\" in EUROCRYPT 1999, LNCS 1592, pp. 223–238, 1999.",
 "I. Damgård and M. Jurik, \"A generalisation, a simplification and some applications of Paillier's probabilistic public-key system,\" in PKC 2001, LNCS 1992, pp. 119–136, 2001.",
 "P.-A. Fouque, G. Poupard and J. Stern, \"Sharing decryption in the context of voting or lotteries,\" in Financial Cryptography 2000, LNCS 1962, pp. 90–104, 2001.",
 "C. P. Schnorr, \"Efficient signature generation by smart cards,\" Journal of Cryptology, vol. 4, no. 3, pp. 161–174, 1991.",
 "A. Shamir, \"How to share a secret,\" Communications of the ACM, vol. 22, no. 11, pp. 612–613, 1979.",
 "M. Castro and B. Liskov, \"Practical Byzantine fault tolerance,\" in Proc. USENIX OSDI, pp. 173–186, 1999.",
 "V. Shejwalkar and A. Houmansadr, \"Manipulating the Byzantine: Optimizing model poisoning attacks and defenses for federated learning,\" in Proc. NDSS, 2021.",
 "S. Saremi, S. Mirjalili and A. Lewis, \"Grasshopper optimisation algorithm: Theory and application,\" Advances in Engineering Software, vol. 105, pp. 30–47, 2017.",
 "R. N. Mantegna, \"Fast, accurate algorithm for numerical simulation of Lévy stable stochastic processes,\" Physical Review E, vol. 49, no. 5, pp. 4677–4683, 1994.",
 "G. E. Hinton, \"Training products of experts by minimizing contrastive divergence,\" Neural Computation, vol. 14, no. 8, pp. 1771–1800, 2002.",
 "S. Ö. Arik and T. Pfister, \"TabNet: Attentive interpretable tabular learning,\" in Proc. AAAI, vol. 35, no. 8, pp. 6679–6687, 2021.",
 "Dreamquark-ai, \"pytorch-tabnet: PyTorch implementation of TabNet,\" https://github.com/dreamquark-ai/tabnet.",
 "S. M. Lundberg and S.-I. Lee, \"A unified approach to interpreting model predictions,\" in Advances in Neural Information Processing Systems 30, pp. 4765–4774, 2017.",
 "G. Erion, J. D. Janizek, P. Sturmfels, S. M. Lundberg and S.-I. Lee, \"Improving performance of deep learning models with axiomatic attribution priors and expected gradients,\" Nature Machine Intelligence, vol. 3, pp. 620–631, 2021.",
 "T.-M. H. Hsu, H. Qi and M. Brown, \"Measuring the effects of non-identical data distribution for federated visual classification,\" arXiv:1909.06335, 2019.",
 "R. Detrano et al., \"International application of a new probability algorithm for the diagnosis of coronary artery disease,\" American Journal of Cardiology, vol. 64, no. 5, pp. 304–310, 1989.",
 "W. N. Street, W. H. Wolberg and O. L. Mangasarian, \"Nuclear feature extraction for breast tumor diagnosis,\" in Proc. SPIE 1905, pp. 861–870, 1993.",
 "J. W. Smith, J. E. Everhart, W. C. Dickson, W. C. Knowler and R. S. Johannes, \"Using the ADAP learning algorithm to forecast the onset of diabetes mellitus,\" in Proc. Annual Symposium on Computer Applications in Medical Care, pp. 261–265, 1988.",
 "A. Paszke et al., \"PyTorch: An imperative style, high-performance deep learning library,\" in Advances in Neural Information Processing Systems 32, 2019.",
 "F. Pedregosa et al., \"Scikit-learn: Machine learning in Python,\" Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.",
 "C. R. Harris et al., \"Array programming with NumPy,\" Nature, vol. 585, pp. 357–362, 2020.",
 "W. McKinney, \"Data structures for statistical computing in Python,\" in Proc. 9th Python in Science Conference, pp. 56–61, 2010.",
 "P. Virtanen et al., \"SciPy 1.0: fundamental algorithms for scientific computing in Python,\" Nature Methods, vol. 17, pp. 261–272, 2020.",
 "J. D. Hunter, \"Matplotlib: A 2D graphics environment,\" Computing in Science & Engineering, vol. 9, no. 3, pp. 90–95, 2007."];
REFS.forEach((t, i) => A(new Paragraph({ children: runs(`[${i + 1}] ${t}`, { size: 19 }), spacing: { after: 50 }, indent: { left: 460, hanging: 460 } })));

const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: 22 } } } },
  numbering: { config: [{ reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{ properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1300, bottom: 1300, left: 1273, right: 1273 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: ["Page ", PageNumber.CURRENT], font: FONT, size: 18 })] })] }) },
    children: body }] });
Packer.toBuffer(doc).then(b => { const out = path.join(ROOT, "Kadermeera_Sahib_Afsin_samina_22646207_Assignment_2_Report.docx"); fs.writeFileSync(out, b); console.log("written", out); });
