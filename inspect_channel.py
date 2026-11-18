
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"

CHANNEL = "P-1"

train = pd.read_parquet(
    DATA_DIR / "data" / "train" / f"{CHANNEL}.parquet"
)

test = pd.read_parquet(
    DATA_DIR / "data" / "test" / f"{CHANNEL}.parquet"
)

labels = pd.read_csv(DATA_DIR / "labeled_anomalies.csv")

channel_labels = labels[
    labels["chan_id"] == CHANNEL
]

print("\n========== P-1 DATASET ==========")

print("Training shape:", train.shape)
print("Testing shape:", test.shape)

print("\n========== TRAIN SIGNAL ==========")

print(train["value"].describe())
print("Unique values:", train["value"].nunique())
print("Standard deviation:", train["value"].std())

print("\n========== TEST SIGNAL ==========")

print(test["value"].describe())
print("Unique values:", test["value"].nunique())
print("Standard deviation:", test["value"].std())

print("\n========== ANOMALY LABELS ==========")

print(channel_labels.to_string(index=False))

print("\n========== COMMAND FEATURES ==========")

cmd_cols = [
    col for col in train.columns
    if col.startswith("cmd_")
]

print("Number of command features:", len(cmd_cols))
print("Active command counts:")
print(train[cmd_cols].sum().sort_values(ascending=False))

print("\nP-1 INVESTIGATION COMPLETE!")
