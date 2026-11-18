import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score

OUT = Path("outputs/temporal_if")
OUT.mkdir(parents=True, exist_ok=True)

# ---- load data ----
train = pd.read_parquet("data/raw/data/train/P-1.parquet")
test = pd.read_parquet("data/raw/data/test/P-1.parquet")
cmd_cols = [c for c in train.columns if c.startswith("cmd_")]

labels = pd.read_csv("data/raw/labeled_anomalies.csv")
row = labels[labels["chan_id"] == "P-1"].iloc[0]
regions = sorted(ast.literal_eval(row["anomaly_sequences"]))

y_true = np.zeros(len(test), dtype=int)
for s, e in regions:
    y_true[s:e + 1] = 1
assert len(test) == 8505 and y_true.sum() == 751


# ---- feature engineering (only looks at the PAST) ----
def make_features(df):
    v = df["value"]
    f = pd.DataFrame(index=df.index)
    f["value"] = v
    for w in (10, 50):
        r = v.rolling(w, min_periods=1)
        f[f"mean_{w}"] = r.mean()
        f[f"std_{w}"] = r.std().fillna(0)
        f[f"min_{w}"] = r.min()
        f[f"max_{w}"] = r.max()
    d = v.diff().fillna(0)
    f["diff_1"] = d
    f["absdiff_mean_10"] = d.abs().rolling(10, min_periods=1).mean()
    active = (df[cmd_cols] != 0).sum(axis=1)
    f["cmd_active_now"] = active
    f["cmd_active_sum_50"] = active.rolling(50, min_periods=1).sum()
    return f


train_feat = make_features(train)
test_feat = make_features(test)
print("Number of features:", train_feat.shape[1])

# ---- chronological split ----
n_fit = int(0.8 * len(train_feat))
X_fit = train_feat.iloc[:n_fit]
X_val = train_feat.iloc[n_fit:]
print("fit:", len(X_fit), "val:", len(X_val), "test:", len(test_feat))

# ---- model ----
model = IsolationForest(n_estimators=300, contamination="auto",
                        random_state=42, n_jobs=-1)
model.fit(X_fit)

val_score = -model.score_samples(X_val)    # higher = more anomalous
test_score = -model.score_samples(test_feat)

roc = roc_auc_score(y_true, test_score)
pr = average_precision_score(y_true, test_score)
print(f"\nThreshold-free metrics -> ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f} "
      f"(random baseline PR-AUC = {y_true.mean():.4f})")


def get_runs(pred):
    d = np.diff(np.concatenate([[0], pred, [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


rows = []
for q in (95, 97.5, 99, 99.5):
    thr = np.percentile(val_score, q)      # threshold from VALIDATION only
    pred = (test_score > thr).astype(int)

    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    rate_in = pred[y_true == 1].mean()
    rate_out = pred[y_true == 0].mean()

    detected, delays = 0, []
    for s, e in regions:
        hit = np.where(pred[s:e + 1] == 1)[0]
        if len(hit):
            detected += 1
            delays.append(int(hit[0]))
    false_events = sum(
        1 for rs, re_ in get_runs(pred)
        if not any(re_ >= s and rs <= e for s, e in regions)
    )

    rows.append({
        "val percentile": q, "threshold": round(thr, 4),
        "precision": round(prec, 4), "recall": round(rec, 4), "F1": round(f1, 4),
        "alarm rate inside anomalies": round(rate_in, 4),
        "alarm rate outside": round(rate_out, 4),
        "ratio in/out": round(rate_in / rate_out, 2) if rate_out > 0 else None,
        "regions detected": f"{detected}/{len(regions)}",
        "mean delay": round(float(np.mean(delays)), 1) if delays else None,
        "false-alarm events": false_events,
        "TP": tp, "FP": fp, "FN": fn,
    })

result = pd.DataFrame(rows)
print("\n=== THRESHOLD SENSITIVITY (thresholds from validation only) ===")
print(result.to_string(index=False))

result.to_csv(OUT / "threshold_sensitivity.csv", index=False)
pd.DataFrame({"timestep": test["timestep"], "actual_anomaly": y_true,
              "anomaly_score": test_score}).to_csv(OUT / "test_scores.csv", index=False)
pd.DataFrame({"val_score": val_score}).to_csv(OUT / "validation_scores.csv", index=False)
print("\nSaved results to", OUT)
