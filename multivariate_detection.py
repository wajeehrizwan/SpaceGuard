
import ast
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

SEED = 42
PERCENTILES = [95, 97.5, 99, 99.5]

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs" / "threshold_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_parquet(
    DATA_DIR / "data" / "train" / "P-1.parquet"
)
test = pd.read_parquet(
    DATA_DIR / "data" / "test" / "P-1.parquet"
)

labels = pd.read_csv(DATA_DIR / "labeled_anomalies.csv")
row = labels[labels["chan_id"] == "P-1"].iloc[0]
intervals = ast.literal_eval(row["anomaly_sequences"])

y_true = np.zeros(len(test), dtype=int)

for start, end in intervals:
    y_true[start:end + 1] = 1

command_columns = [
    col for col in train.columns if col.startswith("cmd_")
]

feature_sets = {
    "Univariate": ["value"],
    "Multivariate": ["value"] + command_columns
}

split = int(len(train) * 0.80)
results = []

for model_name, features in feature_sets.items():

    print("\n" + "=" * 55)
    print(f"MODEL: {model_name}")
    print("=" * 55)

    X_fit = train.iloc[:split][features].to_numpy(dtype=np.float64)
    X_validation = train.iloc[split:][features].to_numpy(dtype=np.float64)
    X_test = test[features].to_numpy(dtype=np.float64)

    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=SEED,
        n_jobs=-1
    )

    model.fit(X_fit)

    validation_scores = -model.score_samples(X_validation)
    test_scores = -model.score_samples(X_test)

    auc = roc_auc_score(y_true, test_scores)

    print("ROC-AUC:", round(auc, 4))

    for percentile in PERCENTILES:

        threshold = np.percentile(
            validation_scores,
            percentile
        )

        predictions = (test_scores > threshold).astype(int)

        precision = precision_score(
            y_true, predictions, zero_division=0
        )
        recall = recall_score(
            y_true, predictions, zero_division=0
        )
        f1 = f1_score(
            y_true, predictions, zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            y_true, predictions
        ).ravel()

        results.append({
            "model": model_name,
            "threshold_percentile": percentile,
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": auc,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn
        })

        print(
            f"\nThreshold percentile: {percentile}"
            f"\nPrecision: {precision:.4f}"
            f"\nRecall: {recall:.4f}"
            f"\nF1: {f1:.4f}"
            f"\nTrue positives: {tp}"
            f"\nFalse positives: {fp}"
        )

comparison = pd.DataFrame(results)

comparison.to_csv(
    OUTPUT_DIR / "threshold_comparison.csv",
    index=False
)

print("\n========== THRESHOLD COMPARISON ==========")
print(comparison.to_string(index=False))

print("\nSaved to:", OUTPUT_DIR)
print("THRESHOLD ANALYSIS COMPLETED")
