"""
run.py  –  Example usage of RPG-AE with synthetic provenance-log data.

The synthetic dataset:
  * 200 "normal" processes with moderate, correlated behavioural features.
  * 20  "anomalous" processes that exhibit rare, extreme combinations.

Run:
    python run.py
"""

import numpy as np
import pandas as pd
from model import RPGAE, FeatureExtractor

SEED = 42
np.random.seed(SEED)

# --- 1. Synthetic data -----------------

FEATURE_NAMES = [
    "file_reads", "file_writes", "net_connections",
    "child_processes", "registry_accesses", "duration_s",
    "cpu_pct", "mem_mb",
]

n_normal   = 200
n_anomaly  = 20
n_total    = n_normal + n_anomaly
d          = len(FEATURE_NAMES)

# Normal processes: low-variance Gaussian
X_normal = np.abs(np.random.randn(n_normal, d) * [5, 3, 2, 1, 4, 30, 10, 200]
                  + [10, 5, 3, 1, 8, 60, 20, 400])

# Anomalous processes: extreme / rare combinations
X_anomaly = np.abs(np.random.randn(n_anomaly, d) * [20, 15, 10, 5, 20, 120, 40, 800]
                   + [80, 60, 40, 20, 60, 300, 80, 2000])

X = np.vstack([X_normal, X_anomaly]).astype(np.float32)
true_labels = np.array([0] * n_normal + [1] * n_anomaly)   # 1 = anomaly
process_ids = [f"proc_{i:04d}" for i in range(n_total)]

print("=" * 60)
print("  RPG-AE  –  Demo with synthetic provenance-log data")
print("=" * 60)
print(f"  Processes: {n_total}  ({n_normal} normal, {n_anomaly} anomalous)")
print(f"  Features : {d}")
print()

# -- 2. Fit RPG-AE --------------------

model = RPGAE(
    k            = 10,
    min_support  = 0.01,
    max_support  = 0.15,
    alpha        = 0.5,
    hidden_dim   = 64,
    emb_dim      = 32,
    lr           = 1e-3,
    epochs       = 200,
    device       = "cpu",
    feature_names = FEATURE_NAMES,
)

model.fit(X, verbose=True)

# --- 3. Anomaly ranking ------------------------

ranking = model.anomaly_ranking(process_ids=process_ids)

print("\n── Top-20 ranked anomalies ------------------")
print(ranking.head(20).to_string(index=False))

# --- 4. Evaluation (AUPRC / precision@k) ------------------

from sklearn.metrics import (average_precision_score,
                              roc_auc_score,
                              precision_score)

scores = model.s_boosted_

auc_roc = roc_auc_score(true_labels, scores)
auc_prc = average_precision_score(true_labels, scores)

# Precision @ k=20
top20_ids = set(ranking.head(20)["process_id"].tolist())
pred_labels = np.array([1 if process_ids[i] in top20_ids else 0
                         for i in range(n_total)])
prec_at_20 = (pred_labels * true_labels).sum() / 20

print("\-- Evaluation metrics ------------------")
print(f"  AUC-ROC       : {auc_roc:.4f}")
print(f"  AUC-PRC       : {auc_prc:.4f}")
print(f"  Precision@20  : {prec_at_20:.4f}")

# --- 5. Rare patterns discovered ---------------------

print(f"\n-- Rare patterns discovered: {len(model.rare_patterns_)} ------------─")
for i, p in enumerate(list(model.rare_patterns_)[:10], 1):
    print(f"  {i:>2}. {set(p)}")
if len(model.rare_patterns_) > 10:
    print(f"  … and {len(model.rare_patterns_) - 10} more")

print("\nDone.")