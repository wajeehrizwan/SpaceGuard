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

# ---- load data and labels ----
train = pd.read_parquet("data/raw/data/train/P-1.parquet")
test = pd.read_parquet("data/raw/data/test/P-1.parquet")
cmd_cols = [c for c in train.columns if c.startswith("cmd_")]

labels = pd.read_csv("data/raw/labeled_anomalies.csv")
row = labels[labels["chan_id"] == "P-1"].iloc[0]
regions = sorted(ast.literal_eval(row["anomaly_sequences"]))
y_true = np.zeros(len(test), dtype=int)
for s, e in regions:
    y_true[s:e + 1] = 1


# ---- same features as temporal_if.py (past-only windows) ----
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

base = ["value", "diff_1"]
w10 = ["mean_10", "std_10", "min_10", "max_10", "absdiff_mean_10"]
w50 = ["mean_50", "std_50", "min_50", "max_50"]
cmd = ["cmd_active_now", "cmd_active_sum_50"]

configs = {
    "value only": ["value"],
    "no commands": base + w10 + w50,
    "commands only": cmd,
    "all features": base + w10 + w50 + cmd,
    "short window + cmd": base + w10 + cmd,
    "long window + cmd": base + w50 + cmd,
}
seeds = [0, 1, 2, 3, 4]

rows = []
for name, cols in configs.items():
    rocs, prs = [], []
    for seed in seeds:
        X_fit = train_feat.iloc[:n_fit][cols]
        model = IsolationForest(n_estimators=300, contamination="auto",
                                random_state=seed, n_jobs=-1)
        model.fit(X_fit)
        score = -model.score_samples(test_feat[cols])
        rocs.append(roc_auc_score(y_true, score))
        prs.append(average_precision_score(y_true, score))
    rows.append({
        "feature set": name, "n_features": len(cols),
        "ROC-AUC mean": round(np.mean(rocs), 4), "ROC-AUC std": round(np.std(rocs), 4),
        "PR-AUC mean": round(np.mean(prs), 4), "PR-AUC std": round(np.std(prs), 4),
    })

res = pd.DataFrame(rows)
print("Random baseline PR-AUC:", round(y_true.mean(), 4))
print(res.to_string(index=False))
res.to_csv(OUT / "ablation.csv", index=False)

# ---- simple chart ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].barh(res["feature set"], res["ROC-AUC mean"], xerr=res["ROC-AUC std"], color="#4C78A8")
ax[0].axvline(0.5, color="red", linestyle="--", label="random (0.5)")
ax[0].set_title("ROC-AUC (mean +/- std over 5 seeds)")
ax[0].legend()
ax[1].barh(res["feature set"], res["PR-AUC mean"], xerr=res["PR-AUC std"], color="#59A14F")
ax[1].axvline(y_true.mean(), color="red", linestyle="--", label="random (0.088)")
ax[1].set_title("PR-AUC (mean +/- std over 5 seeds)")
ax[1].legend()
plt.tight_layout()
plt.savefig(OUT / "ablation.png", dpi=150)
print("\nSaved:", OUT / "ablation.csv", "and", OUT / "ablation.png")