import ast
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("outputs/temporal_if")
OUT.mkdir(parents=True, exist_ok=True)

train = pd.read_parquet("data/raw/data/train/P-1.parquet")
test = pd.read_parquet("data/raw/data/test/P-1.parquet")

labels = pd.read_csv("data/raw/labeled_anomalies.csv")
row = labels[labels["chan_id"] == "P-1"].iloc[0]
regions = sorted(ast.literal_eval(row["anomaly_sequences"]))
y_true = np.zeros(len(test), dtype=int)
for s, e in regions:
    y_true[s:e + 1] = 1


def absdiff10(df):
    d = df["value"].diff().fillna(0).abs()
    return d.rolling(10, min_periods=1).mean()


def std10(df):
    return df["value"].rolling(10, min_periods=1).std().fillna(0)


n_fit = int(0.8 * len(train))
fit = train.iloc[:n_fit]

summary_rows = []
for name, fn in [("absdiff_mean_10", absdiff10), ("std_10", std10)]:
    ref = fn(train).iloc[:n_fit]
    lo, med, hi = ref.quantile(0.05), ref.median(), ref.quantile(0.95)
    t = fn(test)
    print(f"\n=== {name} ===")
    print(f"training reference -> 5th pct: {lo:.4f} | median: {med:.4f} | 95th pct: {hi:.4f}")
    groups = [("normal (all)", t[y_true == 0])]
    for s, e in regions:
        groups.append((f"region {s}-{e}", t.iloc[s:e + 1]))
    for gname, g in groups:
        summary_rows.append({
            "feature": name, "group": gname, "n": len(g),
            "median": round(g.median(), 4),
            "median / train median": round(g.median() / med, 2) if med > 0 else None,
            "share below train 5th pct (quiet)": round((g < lo).mean(), 3),
            "share above train 95th pct (bursty)": round((g > hi).mean(), 3),
        })

summary = pd.DataFrame(summary_rows)
print("\n" + summary.to_string(index=False))
summary.to_csv(OUT / "variability_direction.csv", index=False)

# ---- chart ----
t10 = absdiff10(test)
fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
axes[0].plot(test["value"].values, color="#4C78A8", linewidth=0.7)
axes[0].set_title("Test telemetry value")
axes[1].plot(t10.values, color="#59A14F", linewidth=0.7)
ref = absdiff10(train).iloc[:n_fit]
axes[1].axhline(ref.quantile(0.05), color="blue", linestyle=":", label="train 5th pct (quiet)")
axes[1].axhline(ref.median(), color="gray", linestyle="--", label="train median")
axes[1].axhline(ref.quantile(0.95), color="red", linestyle=":", label="train 95th pct (bursty)")
axes[1].set_title("absdiff_mean_10 (recent average movement)")
axes[1].legend(loc="upper right", fontsize=8)
for ax in axes:
    for i, (s, e) in enumerate(regions):
        ax.axvspan(s, e, color="orange", alpha=0.3, label="true anomaly" if (i == 0 and ax is axes[0]) else None)
axes[0].legend(loc="upper right", fontsize=8)
axes[1].set_xlabel("test timestep")
plt.tight_layout()
plt.savefig(OUT / "variability_direction.png", dpi=150)
print("\nSaved:", OUT / "variability_direction.csv", "and", OUT / "variability_direction.png")