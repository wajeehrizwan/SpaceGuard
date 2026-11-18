import ast
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score

OUT = Path("outputs/generalization")
OUT.mkdir(parents=True, exist_ok=True)
RAW = Path("data/raw")
SPAN = 100

labels = pd.read_csv(RAW / "labeled_anomalies.csv")
ids = sorted(c for c in labels["chan_id"].unique() if c != "P-1")
channels = ids[::8]                       # fixed, label-free selection rule
print("Channels tested:", channels)


def fetch(split, ch):
    path = RAW / "data" / split / f"{ch}.parquet"
    if not path.exists():
        hf_hub_download(repo_id="appleparan/telemanom", repo_type="dataset",
                        filename=f"data/{split}/{ch}.parquet", local_dir=str(RAW))
    return pd.read_parquet(path)


def make_features(df, cmd_cols):
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
    if cmd_cols:
        active = (df[cmd_cols] != 0).sum(axis=1)
        f["cmd_active_now"] = active
        f["cmd_active_sum_50"] = active.rolling(50, min_periods=1).sum()
    return f


def ewma(x, span):
    return pd.Series(x).ewm(span=span, adjust=False).mean().values


def get_runs(pred):
    d = np.diff(np.concatenate([[0], pred, [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


rows = []
for ch in channels:
    try:
        tr, te = fetch("train", ch), fetch("test", ch)
    except Exception as ex:
        print(f"[skip] {ch}: download failed ({ex})")
        continue
    row = labels[labels["chan_id"] == ch].iloc[0]
    regions = sorted(ast.literal_eval(row["anomaly_sequences"]))
    y = np.zeros(len(te), dtype=int)
    for s, e in regions:
        y[s:min(e, len(te) - 1) + 1] = 1
    if y.sum() == 0 or y.sum() == len(y):
        print(f"[skip] {ch}: labels unusable")
        continue
    if tr["value"].std() < 1e-9:
        print(f"[skip] {ch}: constant training telemetry")
        continue

    cmd_cols = [c for c in tr.columns if c.startswith("cmd_")]
    ftr, fte = make_features(tr, cmd_cols), make_features(te, cmd_cols)
    n_fit = int(0.8 * len(ftr))
    if n_fit < 100 or len(ftr) - n_fit < 30:
        print(f"[skip] {ch}: training data too short")
        continue

    scores = {}
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=42, n_jobs=-1)
    iso.fit(ftr.iloc[:n_fit])
    scores["A: Isolation Forest"] = (-iso.score_samples(ftr.iloc[n_fit:]),
                                     -iso.score_samples(fte))
    ref = ftr["absdiff_mean_10"].iloc[:n_fit]
    med, scale = ref.median(), max(ref.std(), 1e-9)
    scores["B: absdiff two-sided"] = tuple(
        np.clip(np.abs(f["absdiff_mean_10"].values - med) / scale, 0, 10)
        for f in (ftr.iloc[n_fit:], fte))
    scores["C: absdiff quiet-only"] = tuple(
        np.clip((med - f["absdiff_mean_10"].values) / scale, 0, 10)
        for f in (ftr.iloc[n_fit:], fte))

    for name, (val_raw, test_raw) in scores.items():
        val_s, test_s = ewma(val_raw, SPAN), ewma(test_raw, SPAN)
        thr = np.percentile(val_s, 99)
        pred = (test_s > thr).astype(int)
        detected = sum(1 for s, e in regions if pred[s:e + 1].any())
        false_events = sum(1 for rs, re_ in get_runs(pred)
                           if not any(re_ >= s and rs <= e for s, e in regions))
        pr = average_precision_score(y, test_s)
        rows.append({
            "channel": ch, "method": name, "anomaly share": round(y.mean(), 3),
            "ROC-AUC": round(roc_auc_score(y, test_s), 3),
            "PR-AUC": round(pr, 3), "PR lift (x random)": round(pr / y.mean(), 2),
            "regions hit @99": f"{detected}/{len(regions)}",
            "false events @99": false_events,
        })
    print("done", ch)

res = pd.DataFrame(rows)
res.to_csv(OUT / "generalization_results.csv", index=False)
print("\n=== PER CHANNEL ===")
print(res.to_string(index=False))

print("\n=== SUMMARY OVER CHANNELS ===")
summ = res.groupby("method").agg(
    channels=("channel", "count"),
    median_ROC=("ROC-AUC", "median"),
    median_PR_lift=("PR lift (x random)", "median"),
    ROC_above_0_5=("ROC-AUC", lambda s: int((s > 0.5).sum())),
    PR_lift_above_1=("PR lift (x random)", lambda s: int((s > 1).sum())),
).round(3)
print(summ.to_string())
summ.to_csv(OUT / "generalization_summary.csv")

piv = res.pivot(index="channel", columns="method", values="ROC-AUC")
piv.plot(kind="bar", figsize=(12, 5))
plt.axhline(0.5, color="red", linestyle="--")
plt.ylabel("ROC-AUC (test)")
plt.title("Generalization to unseen channels (red = chance)")
plt.tight_layout()
plt.savefig(OUT / "generalization.png", dpi=150)
print("\nSaved to", OUT)