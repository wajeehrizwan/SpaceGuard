
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

results = pd.read_csv(
    BASE_DIR / "outputs/lstm/predictions.csv"
)

normal = results[
    results["actual_anomaly"] == 0
]["reconstruction_error"]

anomalous = results[
    results["actual_anomaly"] == 1
]["reconstruction_error"]

print("\n========== RECONSTRUCTION ERROR ANALYSIS ==========")

print("\nNORMAL TELEMETRY")
print(normal.describe().round(5))

print("\nANOMALOUS TELEMETRY")
print(anomalous.describe().round(5))

print("\n========== ERROR COMPARISON ==========")

print("Normal mean:", round(normal.mean(), 5))
print("Anomaly mean:", round(anomalous.mean(), 5))

print("Normal median:", round(normal.median(), 5))
print("Anomaly median:", round(anomalous.median(), 5))

print(
    "Mean error ratio:",
    round(anomalous.mean() / max(normal.mean(), 1e-10), 4)
)

print("\n========== DETECTION ANALYSIS ==========")

threshold = results[
    "reconstruction_error"
][
    results["predicted_anomaly"] == 1
]

if len(threshold) > 0:
    actual_threshold = results.loc[
        results["predicted_anomaly"] == 1,
        "reconstruction_error"
    ].min()
else:
    actual_threshold = float("inf")

print("Flagged readings:", int(results["predicted_anomaly"].sum()))
print("Detected anomalies:", int(
    ((results["actual_anomaly"] == 1) &
     (results["predicted_anomaly"] == 1)).sum()
))

print("\n========== INDIVIDUAL ANOMALY REGIONS ==========")

intervals = [
    (2149, 2349),
    (3539, 3779),
    (4536, 4844)
]

for i, (start, end) in enumerate(intervals, 1):

    region = results.iloc[start:end + 1]

    detected = int(region["predicted_anomaly"].sum())

    print(f"\nAnomaly {i}: {start}-{end}")
    print("Readings:", len(region))
    print("Mean reconstruction error:",
          round(region["reconstruction_error"].mean(), 5))
    print("Maximum reconstruction error:",
          round(region["reconstruction_error"].max(), 5))
    print("Detected readings:", detected)
    print("Detection percentage:",
          round(detected / len(region) * 100, 2), "%")

print("\n========== ERROR PERCENTILES ==========")

print(
    results["reconstruction_error"]
    .quantile([0.50, 0.90, 0.95, 0.99, 0.999])
    .round(5)
)

print("\nANALYSIS COMPLETED")
