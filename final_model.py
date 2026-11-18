import ast
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             confusion_matrix)

OUT = Path("outputs/final")
OUT.mkdir(parents=True, exist_ok=True)

CONFIG = {"channel": "P-1", "model": "IsolationForest", "n_estimators": 300,
          "random_state": 42, "ewma_span": 100, "fit_fraction": 0.8,
          "threshold_percentiles": [95, 99]}

train = pd.read_parquet("data/raw/data/train/P-1.parquet")
test = pd.read_parquet("data/raw/data/test/P-1.parquet")
cmd_cols = [c for c in train.columns if c.startswith("cmd_")]

labels = pd.read_csv("data/raw/labeled_anomalies.csv")
row = labels[labels["chan_id"] == "P-1"].iloc[0]
regions = sorted(ast.literal_eval(row["anomaly_sequences"]))
y = np.zeros(len(test), dtype=int)
for s, e in regions:
    y[s:e + 1] = 1


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


def ewma(x, span):
    return pd.Series(x).ewm(span=span, adjust=False).mean().values


def get_runs(pred):
    d = np.diff(np.concatenate([[0], pred, [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


ftr, fte = make_features(train), make_features(test)
n_fit = int(CONFIG["fit_fraction"] * len(ftr))

model = IsolationForest(n_estimators=CONFIG["n_estimators"], contamination="auto",
                        random_state=CONFIG["random_state"], n_jobs=-1)
model.fit(ftr.iloc[:n_fit])
joblib.dump(model, OUT / "isolation_forest.joblib")

val_raw = -model.score_samples(ftr.iloc[n_fit:])
test_raw = -model.score_samples(fte)
val_s = ewma(val_raw, CONFIG["ewma_span"])
test_s = ewma(test_raw, CONFIG["ewma_span"])

roc = roc_auc_score(y, test_s)
pr = average_precision_score(y, test_s)

pred_df = pd.DataFrame({"timestep": test["timestep"], "value": test["value"],
                        "actual_anomaly": y, "raw_score": test_raw,
                        "smoothed_score": test_s})
metric_rows, thresholds = [], {}
for q in CONFIG["threshold_percentiles"]:
    thr = float(np.percentile(val_s, q))     # validation only
    thresholds[str(q)] = thr
    pred = (test_s > thr).astype(int)
    pred_df[f"pred_p{q}"] = pred
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    detected, delays = 0, []
    for s, e in regions:
        hit = np.where(pred[s:e + 1] == 1)[0]
        if len(hit):
            detected += 1
            delays.append(int(hit[0]))
    false_events = sum(1 for rs, re_ in get_runs(pred)
                       if not any(re_ >= s and rs <= e for s, e in regions))
    metric_rows.append({
        "percentile": q, "threshold": round(thr, 6),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "precision": round(prec, 4), "recall": round(rec, 4), "F1": round(f1, 4),
        "ROC-AUC": round(roc, 4), "PR-AUC": round(pr, 4),
        "random PR-AUC": round(y.mean(), 4),
        "regions detected": f"{detected}/{len(regions)}",
        "mean delay": round(float(np.mean(delays)), 1) if delays else None,
        "false-alarm events": false_events,
    })

metrics = pd.DataFrame(metric_rows)
pred_df.to_csv(OUT / "predictions.csv", index=False)
pd.DataFrame({"val_smoothed_score": val_s}).to_csv(OUT / "validation_scores.csv", index=False)
metrics.to_csv(OUT / "metrics.csv", index=False)
with open(OUT / "config.json", "w") as f:
    json.dump({**CONFIG, "thresholds": thresholds, "regions": regions,
               "features": list(ftr.columns)}, f, indent=2)

print(metrics.T.to_string())
print("\nSaved to", OUT, ":", sorted(p.name for p in OUT.iterdir()))