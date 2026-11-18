import ast
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score

OUT = Path("outputs/temporal_if")
OUT.mkdir(parents=True, exist_ok=True)

train = pd.read_parquet("data/raw/data/train/P-1.parquet")
test = pd.read_parquet("data/raw/data/test/P-1.parquet")
cmd_cols = [c for c in train.columns if c.startswith("cmd_")]

labels = pd.read_csv("data/raw/labeled_anomalies.csv")
row = labels[labels["chan_id"] == "P-1"].iloc[0]
regions = sorted(ast.literal_eval(row["anomaly_sequences"]))
y_true = np.zeros(len(test), dtype=int)
for s, e in regions:
    y_true[s:e + 1] = 1


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
n_fit = int(0.8 * len(train_feat))

model = IsolationForest(n_estimators=300, contamination="auto",
                        random_state=42, n_jobs=-1)
model.fit(train_feat.iloc[:n_fit])
val_raw = -model.score_samples(train_feat.iloc[n_fit:])
test_raw = -model.score_samples(test_feat)


def get_runs(pred):
    d = np.diff(np.concatenate([[0], pred, [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def ewma(x, span):
    return pd.Series(x).ewm(span=span, adjust=False).mean().values


GRACE = 100  # alarms this close after a region ends are "tail", not counted as stricter false events
rows = []
plot = {}
for span in (50, 100, 150, 200, 300):
    val_s = ewma(val_raw, span)
    test_s = ewma(test_raw, span)
    roc = roc_auc_score(y_true, test_s)
    pr = average_precision_score(y_true, test_s)
    for q in (95, 99):
        thr = np.percentile(val_s, q)   # validation only
        pred = (test_s > thr).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / y_true.sum()
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        detected, delays = 0, []
        for s, e in regions:
            hit = np.where(pred[s:e + 1] == 1)[0]
            if len(hit):
                detected += 1
                delays.append(int(hit[0]))
        runs = get_runs(pred)
        loose = sum(1 for rs, re_ in runs
                    if not any(re_ >= s and rs <= e for s, e in regions))
        strict = sum(1 for rs, re_ in runs
                     if not any(re_ >= s and rs <= e + GRACE for s, e in regions))
        rows.append({
            "span": span, "val pct": q, "ROC-AUC": round(roc, 4), "PR-AUC": round(pr, 4),
            "precision": round(prec, 3), "recall": round(rec, 3), "F1": round(f1, 3),
            "outside rate": round(pred[y_true == 0].mean(), 4),
            "expected": round((100 - q) / 100, 3),
            "regions": f"{detected}/3",
            "mean delay": round(float(np.mean(delays)), 1) if delays else None,
            "false events": loose, "strict false events": strict,
        })
        if q == 95 and span in (100, 200, 300):
            plot[span] = (test_s, thr, pred)

res = pd.DataFrame(rows)
res.to_csv(OUT / "smoothing_sweep.csv", index=False)
print("Random baseline PR-AUC:", round(y_true.mean(), 4))
for q in (95, 99):
    print(f"\n=== Validation percentile {q} ===")
    print(res[res["val pct"] == q].drop(columns=["val pct"]).to_string(index=False))

fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
for ax, span in zip(axes, (100, 200, 300)):
    score, thr, pred = plot[span]
    ax.plot(score, color="#4C78A8", linewidth=0.8, label="smoothed score")
    ax.axhline(thr, color="red", linestyle="--", label="95th pct threshold")
    for i, (s, e) in enumerate(regions):
        ax.axvspan(s, e, color="orange", alpha=0.3, label="true anomaly" if i == 0 else None)
    alarm_idx = np.where(pred == 1)[0]
    ax.scatter(alarm_idx, np.full(len(alarm_idx), score.min()), s=2, color="black", label="alarm")
    ax.set_title(f"all features, EWMA span {span}")
    ax.legend(loc="upper right", fontsize=8)
axes[-1].set_xlabel("test timestep")
plt.tight_layout()
plt.savefig(OUT / "sweep_timeline.png", dpi=150)
print("\nSaved:", OUT / "smoothing_sweep.csv", "and", OUT / "sweep_timeline.png")