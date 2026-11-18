
import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"

train_path = DATA_DIR / "data" / "train" / "A-1.parquet"
test_path = DATA_DIR / "data" / "test" / "A-1.parquet"
labels_path = DATA_DIR / "labeled_anomalies.csv"

# ==========================================
# LOAD DATA
# ==========================================

print("\nLOADING NASA TELEMETRY DATA...")

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)
labels = pd.read_csv(labels_path)

# ==========================================
# BASIC DATASET INFORMATION
# ==========================================

print("\n========== TRAIN DATA ==========")
print(train.head(10))
print("\nTrain shape:", train.shape)
print("\nTrain columns:", train.columns.tolist())

print("\n========== TEST DATA ==========")
print(test.head(10))
print("\nTest shape:", test.shape)
print("\nTest columns:", test.columns.tolist())

# ==========================================
# ANOMALY LABEL INFORMATION
# ==========================================

print("\n========== ANOMALY LABELS ==========")
print(labels.head(10))
print("\nLabel shape:", labels.shape)
print("\nLabel columns:", labels.columns.tolist())

# ==========================================
# DATA QUALITY
# ==========================================

print("\n========== DATA QUALITY ==========")

print("Missing training values:",
      train.isnull().sum().sum())

print("Missing testing values:",
      test.isnull().sum().sum())

print("Duplicate training rows:",
      train.duplicated().sum())

# ==========================================
# STATISTICAL ANALYSIS
# ==========================================

print("\n========== TRAIN STATISTICS ==========")

print(train.describe())

if "value" in train.columns:

    print("\nTelemetry mean:", train["value"].mean())
    print("Telemetry std:", train["value"].std())
    print("Telemetry minimum:", train["value"].min())
    print("Telemetry maximum:", train["value"].max())

print("\nINVESTIGATION COMPLETE!")
