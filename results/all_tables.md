### Table 4.1 – Clean performance (mean ± std, 10 seeds)
| Dataset                 | Model       | Accuracy (%)   | Precision     | Recall        | F1            | AUC           | Test loss     |   #features |
|:------------------------|:------------|:---------------|:--------------|:--------------|:--------------|:--------------|:--------------|------------:|
| UCI Heart Disease       | FedAvg      | 81.09 ± 4.59   | 0.830 ± 0.030 | 0.824 ± 0.113 | 0.822 ± 0.053 | 0.896 ± 0.034 | 0.430 ± 0.078 |        13   |
| UCI Heart Disease       | PPFBXAIO    | 72.83 ± 10.86  | 0.741 ± 0.091 | 0.764 ± 0.126 | 0.751 ± 0.106 | 0.795 ± 0.092 | 0.537 ± 0.138 |         6.1 |
| UCI Heart Disease       | PEFL        | 76.30 ± 8.35   | 0.874 ± 0.083 | 0.684 ± 0.214 | 0.742 ± 0.127 | 0.885 ± 0.051 | 0.557 ± 0.195 |        13   |
| UCI Heart Disease       | BPPFL       | 81.09 ± 4.59   | 0.830 ± 0.030 | 0.824 ± 0.113 | 0.822 ± 0.053 | 0.896 ± 0.034 | 0.430 ± 0.078 |        13   |
| UCI Heart Disease       | TabNet-BCFL | 80.22 ± 5.93   | 0.818 ± 0.061 | 0.832 ± 0.126 | 0.817 ± 0.069 | 0.884 ± 0.042 | 0.478 ± 0.151 |        13   |
| Breast Cancer Wisconsin | FedAvg      | 94.53 ± 2.33   | 0.959 ± 0.049 | 0.897 ± 0.087 | 0.923 ± 0.037 | 0.990 ± 0.008 | 0.134 ± 0.039 |        30   |
| Breast Cancer Wisconsin | PPFBXAIO    | 95.00 ± 2.05   | 0.946 ± 0.035 | 0.919 ± 0.042 | 0.932 ± 0.028 | 0.984 ± 0.014 | 0.167 ± 0.038 |        13.1 |
| Breast Cancer Wisconsin | PEFL        | 94.30 ± 3.22   | 0.926 ± 0.072 | 0.928 ± 0.047 | 0.925 ± 0.038 | 0.987 ± 0.010 | 0.146 ± 0.054 |        30   |
| Breast Cancer Wisconsin | BPPFL       | 94.53 ± 2.33   | 0.959 ± 0.049 | 0.897 ± 0.087 | 0.923 ± 0.037 | 0.990 ± 0.008 | 0.134 ± 0.039 |        30   |
| Breast Cancer Wisconsin | TabNet-BCFL | 93.95 ± 2.25   | 0.929 ± 0.060 | 0.912 ± 0.053 | 0.918 ± 0.029 | 0.988 ± 0.009 | 0.130 ± 0.044 |        30   |
| Pima Diabetes           | FedAvg      | 71.98 ± 5.22   | 0.705 ± 0.146 | 0.490 ± 0.305 | 0.498 ± 0.202 | 0.821 ± 0.036 | 0.606 ± 0.135 |         8   |
| Pima Diabetes           | PPFBXAIO    | 70.17 ± 3.84   | 0.604 ± 0.259 | 0.352 ± 0.189 | 0.414 ± 0.211 | 0.733 ± 0.132 | 0.559 ± 0.054 |         4.5 |
| Pima Diabetes           | PEFL        | 71.29 ± 4.34   | 0.526 ± 0.284 | 0.410 ± 0.311 | 0.424 ± 0.270 | 0.819 ± 0.039 | 0.639 ± 0.200 |         8   |
| Pima Diabetes           | BPPFL       | 71.98 ± 5.22   | 0.705 ± 0.146 | 0.490 ± 0.305 | 0.498 ± 0.202 | 0.821 ± 0.036 | 0.606 ± 0.135 |         8   |
| Pima Diabetes           | TabNet-BCFL | 72.41 ± 3.01   | 0.674 ± 0.109 | 0.508 ± 0.194 | 0.543 ± 0.087 | 0.820 ± 0.038 | 0.539 ± 0.057 |         8   |

### Table 4.1b – Paired comparison with FedAvg (clean, accuracy)
| Dataset                 | Model       |   Mean Δacc vs FedAvg (pp) |   Wilcoxon p |
|:------------------------|:------------|---------------------------:|-------------:|
| UCI Heart Disease       | PPFBXAIO    |                      -8.26 |        0.029 |
| UCI Heart Disease       | PEFL        |                      -4.78 |        0.176 |
| UCI Heart Disease       | BPPFL       |                       0    |        1     |
| UCI Heart Disease       | TabNet-BCFL |                      -0.87 |        0.688 |
| Breast Cancer Wisconsin | PPFBXAIO    |                       0.47 |        0.578 |
| Breast Cancer Wisconsin | PEFL        |                      -0.23 |        0.773 |
| Breast Cancer Wisconsin | BPPFL       |                       0    |        1     |
| Breast Cancer Wisconsin | TabNet-BCFL |                      -0.58 |        0.438 |
| Pima Diabetes           | PPFBXAIO    |                      -1.81 |        0.238 |
| Pima Diabetes           | PEFL        |                      -0.69 |        0.443 |
| Pima Diabetes           | BPPFL       |                       0    |        1     |
| Pima Diabetes           | TabNet-BCFL |                       0.43 |        1     |

### Table 4.3 – Measured cost (simulation on 1 CPU core; relative comparison only)
| Model       | Round latency (ms)   | Throughput (tx/s)   | Ledger valid    |
|:------------|:---------------------|:--------------------|:----------------|
| FedAvg      | 1.3 ± 0.3            | 3331 ± 827          | n/a (no ledger) |
| PPFBXAIO    | 1.5 ± 0.3            | 3396 ± 695          | 30/30 runs      |
| PEFL        | 2.0 ± 0.4            | 2524 ± 429          | 30/30 runs      |
| BPPFL       | 328.6 ± 162.6        | 19 ± 8              | 30/30 runs      |
| TabNet-BCFL | 2.5 ± 0.5            | 2072 ± 501          | 30/30 runs      |


Paillier aggregation matched plaintext aggregation to max abs error 5.55e-17.

### Table 4.2 – Accuracy (%) under poisoning attacks (mean ± std, 10 seeds, 20% malicious clients)
| Dataset                 | Model       | Clean       | Extra noise   | Label-flip (20% clients)   | STAT-OPT    | DYN-OPT     |
|:------------------------|:------------|:------------|:--------------|:---------------------------|:------------|:------------|
| UCI Heart Disease       | FedAvg      | 81.1 ± 4.6  | 76.3 ± 7.4    | 71.1 ± 15.1                | 54.6 ± 14.7 | 57.2 ± 19.7 |
| UCI Heart Disease       | PPFBXAIO    | 72.8 ± 10.9 | 74.6 ± 11.3   | 71.7 ± 11.7                | 71.5 ± 12.0 | 71.5 ± 11.5 |
| UCI Heart Disease       | PEFL        | 76.3 ± 8.4  | 74.1 ± 8.2    | 71.3 ± 13.7                | 72.8 ± 14.2 | 65.2 ± 17.1 |
| UCI Heart Disease       | BPPFL       | 81.1 ± 4.6  | 76.3 ± 7.4    | 71.1 ± 15.1                | 54.6 ± 14.7 | 57.2 ± 19.7 |
| UCI Heart Disease       | TabNet-BCFL | 80.2 ± 5.9  | 73.7 ± 9.1    | 78.0 ± 6.9                 | 72.4 ± 10.8 | 59.6 ± 19.9 |
| Breast Cancer Wisconsin | FedAvg      | 94.5 ± 2.3  | 92.1 ± 2.3    | 90.3 ± 7.1                 | 58.0 ± 19.9 | 87.7 ± 10.2 |
| Breast Cancer Wisconsin | PPFBXAIO    | 95.0 ± 2.1  | 93.8 ± 1.6    | 93.6 ± 2.5                 | 93.6 ± 2.5  | 93.7 ± 2.4  |
| Breast Cancer Wisconsin | PEFL        | 94.3 ± 3.2  | 94.4 ± 2.5    | 94.2 ± 3.4                 | 93.4 ± 3.9  | 92.4 ± 3.1  |
| Breast Cancer Wisconsin | BPPFL       | 94.5 ± 2.3  | 92.1 ± 2.3    | 90.3 ± 7.1                 | 58.0 ± 19.9 | 87.7 ± 10.2 |
| Breast Cancer Wisconsin | TabNet-BCFL | 94.0 ± 2.2  | 94.8 ± 2.0    | 94.2 ± 3.0                 | 93.4 ± 3.0  | 89.5 ± 9.6  |
| Pima Diabetes           | FedAvg      | 72.0 ± 5.2  | 64.8 ± 11.3   | 59.7 ± 17.9                | 45.3 ± 16.9 | 62.4 ± 15.2 |
| Pima Diabetes           | PPFBXAIO    | 70.2 ± 3.8  | 73.2 ± 4.4    | 70.6 ± 3.7                 | 70.4 ± 4.0  | 70.3 ± 3.8  |
| Pima Diabetes           | PEFL        | 71.3 ± 4.3  | 67.5 ± 12.2   | 69.8 ± 5.0                 | 70.0 ± 4.8  | 65.0 ± 11.8 |
| Pima Diabetes           | BPPFL       | 72.0 ± 5.2  | 64.8 ± 11.3   | 59.7 ± 17.9                | 45.3 ± 16.9 | 62.4 ± 15.2 |
| Pima Diabetes           | TabNet-BCFL | 72.4 ± 3.0  | 72.8 ± 5.3    | 68.7 ± 12.8                | 67.8 ± 14.4 | 64.8 ± 11.3 |

### Table 4.4 – Ablation of PPFBXAIO (mean ± std, 10 seeds)
| Dataset                 | Variant                                        | Clean acc (%)   | DYN-OPT acc (%)   | Label-flip acc (%)   |
|:------------------------|:-----------------------------------------------|:----------------|:------------------|:---------------------|
| UCI Heart Disease       | Full PPFBXAIO                                  | 72.8 ± 10.9     | 71.5 ± 11.5       | 71.7 ± 11.7          |
| UCI Heart Disease       | w/o LGOA feature selection                     | 79.6 ± 7.3      | 77.2 ± 7.3        | 76.7 ± 8.3           |
| UCI Heart Disease       | w/o EGA (explainability weights)               | 72.8 ± 10.9     | 71.5 ± 11.5       | 72.0 ± 11.7          |
| UCI Heart Disease       | w/o PoC validation (ledger contribution check) | 74.6 ± 11.5     | 68.0 ± 15.4       | 72.0 ± 10.0          |
| Breast Cancer Wisconsin | Full PPFBXAIO                                  | 95.0 ± 2.1      | 93.7 ± 2.4        | 93.6 ± 2.5           |
| Breast Cancer Wisconsin | w/o LGOA feature selection                     | 93.6 ± 3.1      | 93.0 ± 3.3        | 93.3 ± 3.1           |
| Breast Cancer Wisconsin | w/o EGA (explainability weights)               | 95.0 ± 2.1      | 93.7 ± 2.4        | 93.6 ± 2.5           |
| Breast Cancer Wisconsin | w/o PoC validation (ledger contribution check) | 94.3 ± 2.1      | 69.4 ± 12.5       | 69.1 ± 23.4          |
| Pima Diabetes           | Full PPFBXAIO                                  | 70.2 ± 3.8      | 70.3 ± 3.8        | 70.6 ± 3.7           |
| Pima Diabetes           | w/o LGOA feature selection                     | 71.0 ± 4.1      | 70.3 ± 4.0        | 70.1 ± 4.0           |
| Pima Diabetes           | w/o EGA (explainability weights)               | 70.3 ± 3.9      | 70.3 ± 3.8        | 70.3 ± 3.7           |
| Pima Diabetes           | w/o PoC validation (ledger contribution check) | 73.4 ± 3.5      | 63.9 ± 11.0       | 62.2 ± 15.6          |