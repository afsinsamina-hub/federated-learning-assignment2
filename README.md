# Assignment 2 – Federated Learning with Blockchain, Privacy and XAI for Healthcare
Student: Kadermeera Sahib Afsin Samina (22646207). Same research area as Assignment 1.
Setting: Option 1 – dataset-based evaluation (5 models x 3 datasets, 10 seeds).

## Environment
Python 3.12, numpy, pandas, scipy, scikit-learn 1.8, phe (python-paillier) 1.5 + gmpy2, matplotlib, tabulate. No GPU / deep-learning framework needed.
`pip install -r requirements.txt`

## Data (in data/)
- heart.csv – UCI Heart Disease (Cleveland, 303 patients), CSV copy: https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv
- diabetes.csv – Pima Indians Diabetes (768), CSV copy: https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv
- Note: in heart.csv the label coding appears reversed (label 1 has higher max heart rate, lower ST depression, less angina). It is used as given; accuracy and AUC are unaffected, precision/recall/F1 for Heart refer to label 1 (report Section 4.1).
- Breast Cancer Wisconsin (Diagnostic, 569) – loaded with scikit-learn.
- MIMIC-III was not used (credentialed PhysioNet access required); Pima Diabetes is the third dataset.

## Reproduce
```
python src/run_experiments.py   # ~20-25 min on 1 CPU core -> results/raw_results.json, logs/experiments.log
python src/make_report.py       # result tables (CSV/Markdown) and Figures 2-7 -> results/
python src/make_pipeline_figure.py   # Figure 1 (pipeline diagram)
```
## Files
- src/fl_sim.py – models (MLP, attentive MLP), LGOA, hash-chain ledger, Paillier aggregation, attacks, FL loop for FedAvg / PPFBXAIO / PEFL / BPPFL / TabNet-BCFL
- src/common.py – data loading, split, Dirichlet partition, metrics (Accuracy, Precision, Recall, F1, AUC)
- configs/example_config.yaml – documents all experiment settings (the values themselves are the defaults in src/fl_sim.py and src/run_experiments.py)
- requirements.txt – package versions used
- data/ – heart.csv, diabetes.csv; logs/ – run log; results/ – raw results, tables, figures

## Important notes
All five methods are simplified re-implementations (no official code used); see Table 2 and Section 6 of the report for every deviation.
The code was developed with the assistance of an AI coding tool (Claude, Anthropic); see Section 8 of the report.
