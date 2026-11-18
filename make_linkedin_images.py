import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("outputs/linkedin")
OUT.mkdir(parents=True, exist_ok=True)

pred = pd.read_csv("outputs/final/predictions.csv")
metrics = pd.read_csv("outputs/final/metrics.csv")
cfg = json.load(open("outputs/final/config.json"))
regions = cfg["regions"]
m99 = metrics[metrics["percentile"] == 99].iloc[0]

BG, FG, BLUE, ORANGE, RED = "#0E1117", "#E6E6E6", "#5AB0FF", "#F5A623", "#FF4B4B"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                     "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG,
                     "ytick.color": FG, "axes.edgecolor": "#444", "font.size": 11})
FOOT = ("SpaceGuard | channel P-1, public Telemanom benchmark | offline university project, "
        "not NASA-endorsed, not flight-ready")


def finish(fig, name):
    fig.text(0.5, 0.012, FOOT, ha="center", fontsize=8, color="#888")
    fig.savefig(OUT / name, dpi=100)
    plt.close(fig)
    print("saved", OUT / name)


# ---------- Image 1: scoreboard ----------
ev_path = Path("outputs/evaluation/saved_predictions_evaluation.csv")
rows = []
if ev_path.exists():
    ev = pd.read_csv(ev_path)
    rows += [(r["model"], r["PR-AUC"]) for _, r in ev.iterrows()]
rows.append(("Final: Isolation Forest\n+ rolling features + smoothing", float(m99["PR-AUC"])))
rand = float(m99["random PR-AUC"])
fig, ax = plt.subplots(figsize=(12, 6.27))
names = [r[0] for r in rows][::-1]
vals = [r[1] for r in rows][::-1]
colors = [BLUE if "Final" in n else "#666" for n in names]
ax.barh(names, vals, color=colors)
ax.axvline(rand, color=ORANGE, linestyle="--")
ax.text(rand + 0.005, -0.45, f"random guessing = {rand:.2f}", color=ORANGE, fontsize=10)
for i, v in enumerate(vals):
    ax.text(v + 0.008, i, f"{v:.2f}", va="center", fontweight="bold")
ax.set_xlim(0, max(vals) * 1.2)
ax.set_xlabel("PR-AUC on test data (higher is better)")
ax.set_title("From coin-flip to ~5x better than random", fontsize=18, fontweight="bold", loc="left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(rect=(0, 0.04, 1, 1))
finish(fig, "1_scoreboard.png")

# ---------- Image 2: zoom on each anomaly ----------
fig, axes = plt.subplots(1, 3, figsize=(12, 6.27))
for k, (ax, (s, e)) in enumerate(zip(axes, regions), 1):
    lo, hi = max(0, s - 150), min(len(pred) - 1, e + 150)
    sub = pred.iloc[lo:hi + 1]
    ax.plot(sub["timestep"], sub["value"], color=BLUE, linewidth=0.8)
    ax.axvspan(s, e, color=ORANGE, alpha=0.25)
    al = sub[sub["pred_p99"] == 1]
    ax.scatter(al["timestep"], al["value"], s=10, color=RED, zorder=3)
    ax.set_title(f"Anomaly {k} (steps {s}-{e})", fontsize=12)
    ax.set_xlabel("timestep")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("telemetry value")
fig.suptitle("Orange = labeled anomaly   |   Red = model alarms (99th percentile threshold)",
             fontsize=13, y=0.97)
fig.tight_layout(rect=(0, 0.04, 1, 0.94))
finish(fig, "2_zoom_anomalies.png")

# ---------- Image 3: score timeline ----------
fig, ax = plt.subplots(figsize=(12, 6.27))
ax.plot(pred["timestep"], pred["smoothed_score"], color=BLUE, linewidth=1.2)
for p, c in (("95", "#FFD166"), ("99", RED)):
    ax.axhline(cfg["thresholds"][p], color=c, linestyle="--", linewidth=1)
    ax.text(len(pred) * 1.005, cfg["thresholds"][p], f"{p}th pct", color=c, va="center", fontsize=9)
for i, (s, e) in enumerate(regions):
    ax.axvspan(s, e, color=ORANGE, alpha=0.25, label="labeled anomaly" if i == 0 else None)
ax.set_xlabel("test timestep")
ax.set_ylabel("smoothed anomaly score")
ax.set_title(f"Anomaly score over time | 99th pct: {m99['regions detected']} events found, "
             f"precision {m99['precision']:.2f}, recall {m99['recall']:.2f}",
             fontsize=14, fontweight="bold", loc="left")
ax.legend(loc="upper left")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout(rect=(0, 0.04, 0.97, 1))
finish(fig, "3_score_timeline.png")