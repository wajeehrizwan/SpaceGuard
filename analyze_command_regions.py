
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"

test = pd.read_parquet(
    DATA_DIR / "data" / "test" / "P-1.parquet"
)

command_columns = [
    col for col in test.columns
    if col.startswith("cmd_")
]

intervals = [
    (2149, 2349),
    (3539, 3779),
    (4536, 4844)
]

CONTEXT = 200

print("\n========== REGIONAL COMMAND ANALYSIS ==========")

for number, (start, end) in enumerate(intervals, 1):

    anomaly = test.iloc[start:end + 1]

    before = test.iloc[
        max(0, start - CONTEXT):start
    ]

    after = test.iloc[
        end + 1:min(len(test), end + CONTEXT + 1)
    ]

    print(f"\n{'=' * 55}")
    print(f"ANOMALY REGION {number}: {start}-{end}")
    print(f"{'=' * 55}")

    print("\nCommand activity percentages:")

    comparison = pd.DataFrame({
        "Before": before[command_columns].ne(0).mean() * 100,
        "Anomaly": anomaly[command_columns].ne(0).mean() * 100,
        "After": after[command_columns].ne(0).mean() * 100
    })

    comparison["Anomaly - Before"] = (
        comparison["Anomaly"] - comparison["Before"]
    )

    comparison = comparison.sort_values(
        "Anomaly - Before",
        key=lambda x: x.abs(),
        ascending=False
    )

    print(comparison.head(12).round(3).to_string())

    print("\nTotal command activations:")
    print("Before:", int(before[command_columns].ne(0).sum().sum()))
    print("Anomaly:", int(anomaly[command_columns].ne(0).sum().sum()))
    print("After:", int(after[command_columns].ne(0).sum().sum()))

print("\nREGIONAL ANALYSIS COMPLETED")
