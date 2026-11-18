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
SPAN = 100

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


def ewma(x, span):
    return pd.Series(x).ewm(span=span, adjust=False).mean().values


train_feat = make_features(train)
test_feat = make_features(test)
n_fit = int(0.8 * len(train_feat))
fit = train_feat.iloc[:n_fit]

rows = []
z_all = {}
for col in train_feat.columns:
    center = fit[col].median()
    scale = max(fit[col].std(), 1e-9)
    z = np.clip(np.abs(test_feat[col].values - center) / scale, 0, 10)
    z_all[col] = z
    s = ewma(z, SPAN)
    rows.append({"method": f"single feature: {col}",
                 "ROC-AUC": round(roc_auc_score(y_true, s), 4),
                 "PR-AUC": round(average_precision_score(y_true, s), 4)})

# simple combination: average of all clipped distances (still no model)
avg = ewma(np.mean(list(z_all.values()), axis=0), SPAN)
rows.append({"method": "average of all features (no model)",
             "ROC-AUC": round(roc_auc_score(y_true, avg), 4),
             "PR-AUC": round(average_precision_score(y_true, avg), 4)})

# reference: our Isolation Forest, same settings as before
model = IsolationForest(n_estimators=300, contamination="auto", random_state=42, n_jobs=-1)
model.fit(fit)
iso = ewma(-model.score_samples(test_feat), SPAN)
rows.append({"method": "Isolation Forest, all features (our main model)",
             "ROC-AUC": round(roc_auc_score(y_true, iso), 4),
             "PR-AUC": round(average_precision_score(y_true, iso), 4)})

res = pd.DataFrame(rows).sort_values("PR-AUC", ascending=False).reset_index(drop=True)
print("Random baseline PR-AUC:", round(y_true.mean(), 4), "| smoothing span:", SPAN)
print(res.to_string(index=False))
res.to_csv(OUT / "single_feature_baseline.csv", index=False)

# chart
plot = res.iloc[::-1]
colors = ["#E15759" if "Isolation" in m else "#4C78A8" if "average" in m else "#9AA5B1"
          for m in plot["method"]]
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(plot["method"], plot["PR-AUC"], color=colors)
ax.axvline(y_true.mean(), color="black", linestyle="--", label="random (0.088)")
ax.set_xlabel("PR-AUC (test, smoothed span 100)")
ax.set_title("Do we need the model? Single features vs Isolation Forest")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "single_feature_baseline.png", dpi=150)
print("\nSaved:", OUT / "single_feature_baseline.csv", "and", OUT / "single_feature_baseline.png")