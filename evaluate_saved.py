import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

OUT = Path("outputs/evaluation")
OUT.mkdir(parents=True, exist_ok=True)

# ---- 1. Load labels (and sort the regions, the CSV order is mixed up) ----
labels = pd.read_csv("data/raw/labeled_anomalies.csv")
row = labels[labels["chan_id"] == "P-1"].iloc[0]
regions = sorted(ast.literal_eval(row["anomaly_sequences"]))
print("Sorted regions:", regions)

# ---- 2. Load saved predictions ----
mv = pd.read_csv("outputs/multivariate/predictions.csv")
ls = pd.read_csv("outputs/lstm/predictions.csv")

# ---- 3. Sanity checks (alignment) ----
assert (mv["timestep"].values == ls["timestep"].values).all(), "timesteps differ"
assert (mv["actual_anomaly"].values == ls["actual_anomaly"].values).all(), "labels differ"

y_true = mv["actual_anomaly"].values.astype(int)
mask = np.zeros(len(mv), dtype=int)
for s, e in regions:
    mask[s:e + 1] = 1  # inclusive end
assert (mask == y_true).all(), "labels do not match the CSV regions"
print("Alignment OK. Anomalous points:", y_true.sum(), "of", len(y_true))

# ---- 4. The models we evaluate (score, prediction) ----
models = {
    "Univariate IF": (mv["univariate_score"].values, mv["univariate_prediction"].values),
    "Multivariate IF": (mv["multivariate_score"].values, mv["multivariate_prediction"].values),
    "LSTM Autoencoder": (ls["reconstruction_error"].values, ls["predicted_anomaly"].values),
}


def get_runs(pred):
    """Return (start, end) of each consecutive block of alarms."""
    p = np.asarray(pred).astype(int)
    d = np.diff(np.concatenate([[0], p, [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    return list(zip(starts, ends))


rows = []
for name, (score, pred) in models.items():
    pred = np.asarray(pred).astype(int)

    # direction check: alarms should have higher scores than non-alarms
    print(f"\n{name}: min score of alarms = {score[pred == 1].min():.4f}, "
          f"max score of non-alarms = {score[pred == 0].max():.4f}")

    roc = roc_auc_score(y_true, score)
    pr = average_precision_score(y_true, score)

    # event-level
    detected = 0
    delays = []
    per_region = []
    for s, e in regions:
        hit = np.where(pred[s:e + 1] == 1)[0]
        if len(hit) > 0:
            detected += 1
            delays.append(int(hit[0]))
            per_region.append(f"{s}-{e}: hit (delay {hit[0]})")
        else:
            per_region.append(f"{s}-{e}: missed")

    runs = get_runs(pred)
    false_events = 0
    for rs, re_ in runs:
        overlaps = any(re_ >= s and rs <= e for s, e in regions)
        if not overlaps:
            false_events += 1

    normal_points = int((y_true == 0).sum())
    fa_points = int(((pred == 1) & (y_true == 0)).sum())

    rows.append({
        "model": name,
        "ROC-AUC": round(roc, 4),
        "PR-AUC": round(pr, 4),
        "PR-AUC random baseline": round(y_true.mean(), 4),
        "regions detected": f"{detected}/{len(regions)}",
        "mean delay (steps)": round(float(np.mean(delays)), 1) if delays else None,
        "false-alarm events": false_events,
        "false-alarm points per 1000 normal": round(1000 * fa_points / normal_points, 2),
    })
    print("  regions:", "; ".join(per_region))

result = pd.DataFrame(rows)
print("\n=== SCOREBOARD ===")
print(result.to_string(index=False))
result.to_csv(OUT / "saved_predictions_evaluation.csv", index=False)
print("\nSaved to", OUT / "saved_predictions_evaluation.csv")