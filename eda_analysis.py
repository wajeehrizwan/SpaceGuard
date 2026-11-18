
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from pathlib import Path

# ==========================================
# SPACEGUARD - EXPLORATORY DATA ANALYSIS
# NASA SMAP | CHANNEL P-1
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs" / "eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHANNEL = "P-1"

# ==========================================
# LOAD DATA
# ==========================================

train = pd.read_parquet(
    DATA_DIR / "data" / "train" / f"{CHANNEL}.parquet"
)

test = pd.read_parquet(
    DATA_DIR / "data" / "test" / f"{CHANNEL}.parquet"
)

labels = pd.read_csv(
    DATA_DIR / "labeled_anomalies.csv"
)

channel_label = labels[
    labels["chan_id"] == CHANNEL
].iloc[0]

anomaly_sequences = ast.literal_eval(
    channel_label["anomaly_sequences"]
)

anomaly_sequences = sorted(anomaly_sequences)

# ==========================================
# 1. TRAINING TELEMETRY
# ==========================================

plt.figure(figsize=(16, 5))

plt.plot(
    train["timestep"],
    train["value"],
    linewidth=0.8
)

plt.title(
    "NASA SMAP - P-1 Training Telemetry",
    fontsize=15
)

plt.xlabel("Timestep")
plt.ylabel("Telemetry Value")
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "training_telemetry.png",
    dpi=200
)

plt.show()

# ==========================================
# 2. TEST TELEMETRY + ANOMALIES
# ==========================================

plt.figure(figsize=(18, 6))

plt.plot(
    test["timestep"],
    test["value"],
    linewidth=0.8,
    label="Observed Telemetry"
)

for i, (start, end) in enumerate(anomaly_sequences):

    plt.axvspan(
        start,
        end,
        color="red",
        alpha=0.2,
        label="Labeled Anomaly" if i == 0 else None
    )

plt.title(
    "NASA SMAP - P-1 Test Telemetry and Anomaly Intervals",
    fontsize=15
)

plt.xlabel("Timestep")
plt.ylabel("Telemetry Value")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "test_anomaly_timeline.png",
    dpi=200
)

plt.show()

# ==========================================
# 3. COMMAND ACTIVITY HEATMAP
# ==========================================

cmd_cols = [
    col for col in test.columns
    if col.startswith("cmd_")
]

active_cmd_cols = [
    col for col in cmd_cols
    if test[col].sum() > 0
]

if active_cmd_cols:

    command_matrix = test[active_cmd_cols].T

    plt.figure(figsize=(18, 7))

    plt.imshow(
        command_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis"
    )

    plt.yticks(
        range(len(active_cmd_cols)),
        active_cmd_cols
    )

    plt.xlabel("Timestep")
    plt.ylabel("Command Features")

    plt.title(
        "P-1 Command Feature Activity",
        fontsize=15
    )

    plt.colorbar(label="Command Activation")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "command_activity.png",
        dpi=200
    )

    plt.show()

# ==========================================
# 4. STATISTICAL REPORT
# ==========================================

report = pd.DataFrame({
    "Training": train["value"].describe(),
    "Testing": test["value"].describe()
})

print("\n========== STATISTICAL COMPARISON ==========")
print(report)

print("\n========== ANOMALY INTERVALS ==========")

total_anomalies = 0

for start, end in anomaly_sequences:

    length = end - start + 1
    total_anomalies += length

    print(
        f"Start: {start} | End: {end} | "
        f"Length: {length}"
    )

print("\nTotal labeled anomaly readings:", total_anomalies)

print(
    "Percentage of test stream:",
    round(total_anomalies / len(test) * 100, 2),
    "%"
)

# ==========================================
# 5. EXPORT STATISTICAL REPORT
# ==========================================

report.to_csv(
    OUTPUT_DIR / "statistical_report.csv"
)

print("\n==========================================")
print("SPACEGUARD EDA COMPLETED")
print("==========================================")
print("Saved visualizations to:", OUTPUT_DIR)
