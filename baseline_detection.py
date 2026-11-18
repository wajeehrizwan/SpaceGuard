
import pandas as pd
import numpy as np
import ast

from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# ==========================================
# SPACEGUARD - BASELINE ANOMALY DETECTION
# NASA SMAP | CHANNEL P-1
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs" / "baseline"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHANNEL = "P-1"
RANDOM_STATE = 42

# ==========================================
# LOAD DATA
# ==========================================

train = pd.read_parquet(
    DATA_DIR / "data" / "train" / f"{CHANNEL}.parquet"
)

test = pd.read_parquet(
    DATA_DIR / "data" / "test" / f"{CHANNEL}.parquet"
)

labels = pd.read_csv(DATA_DIR / "labeled_anomalies.csv")

channel_label = labels[
    labels["chan_id"] == CHANNEL
].iloc[0]

anomaly_sequences = ast.literal_eval(
    channel_label["anomaly_sequences"]
)

X_train = train[["value"]].to_numpy()
X_test = test[["value"]].to_numpy()

# ==========================================
# GENERATE GROUND-TRUTH LABELS
# ==========================================

y_true = np.zeros(len(test), dtype=int)

for start, end in anomaly_sequences:
    if start < 0 or end >= len(test) or start > end:
        raise ValueError(f"Invalid anomaly interval: {start}-{end}")

    y_true[start:end + 1] = 1

print("\n========== DATASET ==========")
print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))
print("Actual anomalies:", y_true.sum())
print("Anomaly percentage:", round(y_true.mean() * 100, 2), "%")

# ==========================================
# EVALUATION FUNCTION
# ==========================================

def evaluate_model(name, y_true, predictions, scores):

    print(f"\n========== {name} ==========")

    precision = precision_score(
        y_true, predictions, zero_division=0
    )

    recall = recall_score(
        y_true, predictions, zero_division=0
    )

    f1 = f1_score(
        y_true, predictions, zero_division=0
    )

    auc = roc_auc_score(y_true, scores)

    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC-AUC:", round(auc, 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, predictions))

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        predictions,
        zero_division=0
    ))

    return {
        "model": name,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc
    }


results = []

# ==========================================
# MODEL 1: STATISTICAL Z-SCORE
# ==========================================

print("\nTraining Z-score baseline...")

train_mean = X_train.mean()
train_std = X_train.std()

if train_std == 0:
    raise ValueError("Training signal has zero variance.")

train_z = np.abs((X_train.ravel() - train_mean) / train_std)
test_z = np.abs((X_test.ravel() - train_mean) / train_std)

# Threshold determined exclusively from training data
z_threshold = np.quantile(train_z, 0.99)

z_predictions = (test_z > z_threshold).astype(int)

results.append(
    evaluate_model(
        "Z-Score",
        y_true,
        z_predictions,
        test_z
    )
)

print("Z-score threshold:", round(z_threshold, 4))

# ==========================================
# MODEL 2: ISOLATION FOREST
# ==========================================

print("\nTraining Isolation Forest...")

isolation_forest = IsolationForest(
    n_estimators=300,
    contamination="auto",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

isolation_forest.fit(X_train)

# Higher scores represent greater anomaly likelihood
train_scores = -isolation_forest.score_samples(X_train)
test_scores = -isolation_forest.score_samples(X_test)

# Calibrate threshold using training scores only
if_threshold = np.quantile(train_scores, 0.99)

if_predictions = (
    test_scores > if_threshold
).astype(int)

results.append(
    evaluate_model(
        "Isolation Forest",
        y_true,
        if_predictions,
        test_scores
    )
)

print("Isolation Forest threshold:", round(if_threshold, 4))

# ==========================================
# SAVE RESULTS
# ==========================================

comparison = pd.DataFrame(results)

comparison.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)

predictions_df = pd.DataFrame({
    "timestep": test["timestep"],
    "actual_anomaly": y_true,
    "z_score": test_z,
    "z_score_prediction": z_predictions,
    "isolation_forest_score": test_scores,
    "isolation_forest_prediction": if_predictions
})

predictions_df.to_csv(
    OUTPUT_DIR / "predictions.csv",
    index=False
)

print("\n========== MODEL COMPARISON ==========")
print(comparison.to_string(index=False))

print("\nResults saved to:", OUTPUT_DIR)
print("\nBASELINE EXPERIMENT COMPLETED!")
