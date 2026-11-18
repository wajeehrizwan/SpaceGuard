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

all_cols = list(train_feat.columns)
no_cmd_cols = [c for c in all_cols if not c.startswith("cmd_")]
feature_sets = {"all features": all_cols, "no commands": no_cmd_cols}


def get_runs(pred):
    d = np.diff(np.concatenate([[0], pred, [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def ewma(x, span):
    # causal smoothing: only uses past values
    return pd.Series(x).ewm(span=span, adjust=False).mean().values


rows = []
plot_data = {}
for fs_name, cols in feature_sets.items():
    model = IsolationForest(n_estimators=300, contamination="auto",
                            random_state=42, n_jobs=-1)
    model.fit(train_feat.iloc[:n_fit][cols])
    val_raw = -model.score_samples(train_feat.iloc[n_fit:][cols])
    test_raw = -model.score_samples(test_feat[cols])

    for span in (1, 10, 25, 50, 100):
        val_s = ewma(val_raw, span)
        test_s = ewma(test_raw, span)
        roc = roc_auc_score(y_true, test_s)
        pr = average_precision_score(y_true, test_s)

        for q in (95, 97.5, 99):
            thr = np.percentile(val_s, q)       # validation only
            pred = (test_s > thr).astype(int)
            tp = int(((pred == 1) & (y_true == 1)).sum())
            fp = int(((pred == 1) & (y_true == 0)).sum())
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / y_true.sum()
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            r_in = pred[y_true == 1].mean()
            r_out = pred[y_true == 0].mean()
            detected, delays = 0, []
            for s, e in regions:
                hit = np.where(pred[s:e + 1] == 1)[0]
                if len(hit):
                    detected += 1
                    delays.append(int(hit[0]))
            false_events = sum(
                1 for rs, re_ in get_runs(pred)
                if not any(re_ >= s and rs <= e for s, e in regions))
            rows.append({
                "features": fs_name, "span": span, "val pct": q,
                "ROC-AUC": round(roc, 4), "PR-AUC": round(pr, 4),
                "precision": round(prec, 3), "recall": round(rec, 3), "F1": round(f1, 3),
                "ratio in/out": round(r_in / r_out, 2) if r_out > 0 else None,
                "regions": f"{detected}/3",
                "mean delay": round(float(np.mean(delays)), 1) if delays else None,
                "false events": false_events,
            })
        if span == 50:
            plot_data[fs_name] = (test_s, np.percentile(val_s, 95))

res = pd.DataFrame(rows)
res.to_csv(OUT / "smoothing_results.csv", index=False)

print("=== Threshold-free metrics by smoothing span ===")
auc = res[res["val pct"] == 95][["features", "span", "ROC-AUC", "PR-AUC"]]
print(auc.to_string(index=False))
print("\n=== Event-level results at the 95th validation percentile ===")
print(res[res["val pct"] == 95].drop(columns=["ROC-AUC", "PR-AUC"]).to_string(index=False))
print("\n=== Event-level results at the 99th validation percentile ===")
print(res[res["val pct"] == 99].drop(columns=["ROC-AUC", "PR-AUC"]).to_string(index=False))

# ---- chart: smoothed score over time ----
fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
for ax, (name, (score, thr)) in zip(axes, plot_data.items()):
    ax.plot(score, color="#4C78A8", linewidth=0.8, label="smoothed anomaly score")
    ax.axhline(thr, color="red", linestyle="--", label="95th pct threshold (validation)")
    for i, (s, e) in enumerate(regions):
        ax.axvspan(s, e, color="orange", alpha=0.3,
                   label="true anomaly" if i == 0 else None)
    ax.set_title(f"{name} (EWMA span 50)")
    ax.legend(loc="upper right")
axes[1].set_xlabel("test timestep")
plt.tight_layout()
plt.savefig(OUT / "smoothed_scores.png", dpi=150)
print("\nSaved:", OUT / "smoothing_results.csv", "and", OUT / "smoothed_scores.png")