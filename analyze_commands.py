
import pandas as pd
import numpy as np
import ast
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"

test = pd.read_parquet(
    DATA_DIR / "data" / "test" / "P-1.parquet"
)

labels = pd.read_csv(
    DATA_DIR / "labeled_anomalies.csv"
)

row = labels[labels["chan_id"] == "P-1"].iloc[0]

intervals = ast.literal_eval(row["anomaly_sequences"])

command_columns = [
    col for col in test.columns if col.startswith("cmd_")
]

test["is_anomaly"] = 0

for start, end in intervals:
    test.loc[start:end, "is_anomaly"] = 1

normal = test[test["is_anomaly"] == 0]
anomalous = test[test["is_anomaly"] == 1]

print("\n========== COMMAND FEATURE ANALYSIS ==========")

print("Total command features:", len(command_columns))
print("Normal readings:", len(normal))
print("Anomalous readings:", len(anomalous))

results = []

for col in command_columns:

    normal_rate = (normal[col] != 0).mean()
    anomaly_rate = (anomalous[col] != 0).mean()

    results.append({
        "command": col,
        "normal_activity_%": normal_rate * 100,
        "anomaly_activity_%": anomaly_rate * 100,
        "difference_%": (anomaly_rate - normal_rate) * 100
    })

comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    "difference_%",
    key=lambda x: x.abs(),
    ascending=False
)

print("\n========== COMMAND ACTIVITY DIFFERENCES ==========")

print(comparison.round(3).to_string(index=False))

print("\n========== MOST ACTIVE COMMANDS DURING ANOMALIES ==========")

print(
    anomalous[command_columns]
    .ne(0)
    .mean()
    .mul(100)
    .sort_values(ascending=False)
    .head(10)
    .round(3)
)

comparison.to_csv(
    BASE_DIR / "outputs" / "eda" / "command_anomaly_comparison.csv",
    index=False
)

print("\nAnalysis saved successfully.")
