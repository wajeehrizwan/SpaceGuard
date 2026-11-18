
import ast
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ==========================================
# CONFIGURATION
# ==========================================

SEED = 42
WINDOW_SIZE = 50
HIDDEN_SIZE = 32
LATENT_SIZE = 16
BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 0.001
THRESHOLD_PERCENTILE = 99

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs" / "lstm"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")

# ==========================================
# LOAD DATA
# ==========================================

train_df = pd.read_parquet(
    DATA_DIR / "data" / "train" / "P-1.parquet"
)

test_df = pd.read_parquet(
    DATA_DIR / "data" / "test" / "P-1.parquet"
)

labels = pd.read_csv(DATA_DIR / "labeled_anomalies.csv")

channel_label = labels[
    labels["chan_id"] == "P-1"
].iloc[0]

anomaly_sequences = ast.literal_eval(
    channel_label["anomaly_sequences"]
)

train_values = train_df["value"].to_numpy(dtype=np.float32)
test_values = test_df["value"].to_numpy(dtype=np.float32)

y_true = np.zeros(len(test_values), dtype=np.int32)

for start, end in anomaly_sequences:
    y_true[start:end + 1] = 1

# ==========================================
# CHRONOLOGICAL TRAIN / VALIDATION SPLIT
# ==========================================

split_index = int(len(train_values) * 0.80)

train_raw = train_values[:split_index]
validation_raw = train_values[split_index:]

# Fit scaler ONLY on training portion.
mean = train_raw.mean()
std = train_raw.std()

if std == 0:
    raise ValueError("Training telemetry has zero variance.")

train_scaled = (train_raw - mean) / std
validation_scaled = (validation_raw - mean) / std
test_scaled = (test_values - mean) / std

# ==========================================
# CREATE SEQUENTIAL WINDOWS
# ==========================================

def create_windows(values, window_size):
    if len(values) < window_size:
        raise ValueError("Not enough readings for window creation.")

    windows = np.stack([
        values[i:i + window_size]
        for i in range(len(values) - window_size + 1)
    ])

    return torch.tensor(
        windows,
        dtype=torch.float32
    ).unsqueeze(-1)


X_train = create_windows(train_scaled, WINDOW_SIZE)
X_validation = create_windows(validation_scaled, WINDOW_SIZE)
X_test = create_windows(test_scaled, WINDOW_SIZE)

train_loader = DataLoader(
    TensorDataset(X_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

print("\n========== DATA PREPARATION ==========")
print("Training readings:", len(train_raw))
print("Validation readings:", len(validation_raw))
print("Testing readings:", len(test_values))
print("Training windows:", len(X_train))
print("Validation windows:", len(X_validation))
print("Testing windows:", len(X_test))
print("Window size:", WINDOW_SIZE)

# ==========================================
# LSTM AUTOENCODER
# ==========================================

class LSTMAutoencoder(nn.Module):

    def __init__(self, hidden_size, latent_size):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.compress = nn.Linear(
            hidden_size,
            latent_size
        )

        self.decoder_input = nn.Linear(
            latent_size,
            hidden_size
        )

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.output_layer = nn.Linear(
            hidden_size,
            1
        )

    def forward(self, x):

        _, (hidden, _) = self.encoder(x)

        latent = self.compress(hidden[-1])

        repeated = self.decoder_input(latent).unsqueeze(1)

        repeated = repeated.repeat(
            1,
            x.size(1),
            1
        )

        decoded, _ = self.decoder(repeated)

        reconstruction = self.output_layer(decoded)

        return reconstruction


model = LSTMAutoencoder(
    HIDDEN_SIZE,
    LATENT_SIZE
).to(DEVICE)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ==========================================
# TRAINING
# ==========================================

print("\n========== TRAINING LSTM AUTOENCODER ==========")

best_validation_loss = float("inf")
best_state = None
patience = 6
patience_counter = 0

for epoch in range(EPOCHS):

    model.train()
    training_loss = 0.0

    for (batch,) in train_loader:

        batch = batch.to(DEVICE)

        optimizer.zero_grad()

        reconstruction = model(batch)

        loss = criterion(reconstruction, batch)

        loss.backward()

        optimizer.step()

        training_loss += loss.item() * len(batch)

    training_loss /= len(X_train)

    model.eval()
    validation_loss = 0.0

    with torch.no_grad():

        for start in range(0, len(X_validation), BATCH_SIZE):

            batch = X_validation[
                start:start + BATCH_SIZE
            ].to(DEVICE)

            reconstruction = model(batch)

            loss = criterion(reconstruction, batch)

            validation_loss += loss.item() * len(batch)

    validation_loss /= len(X_validation)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {training_loss:.6f} | "
        f"Validation Loss: {validation_loss:.6f}"
    )

    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss

        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

        patience_counter = 0

    else:

        patience_counter += 1

    if patience_counter >= patience:

        print("Early stopping triggered.")

        break


if best_state is None:
    raise RuntimeError("Model training failed.")

model.load_state_dict(best_state)

torch.save(
    model.state_dict(),
    OUTPUT_DIR / "lstm_autoencoder.pth"
)

# ==========================================
# POINT-LEVEL RECONSTRUCTION ERRORS
# ==========================================

def calculate_point_errors(model, windows, total_length):

    model.eval()

    error_sum = np.zeros(total_length, dtype=np.float64)
    count = np.zeros(total_length, dtype=np.int32)

    with torch.no_grad():

        for start in range(0, len(windows), BATCH_SIZE):

            batch = windows[
                start:start + BATCH_SIZE
            ].to(DEVICE)

            reconstruction = model(batch)

            errors = (
                reconstruction - batch
            ).pow(2).squeeze(-1).cpu().numpy()

            for j, window_errors in enumerate(errors):

                window_start = start + j
                window_end = window_start + WINDOW_SIZE

                error_sum[window_start:window_end] += window_errors
                count[window_start:window_end] += 1

    return error_sum / np.maximum(count, 1)


validation_errors = calculate_point_errors(
    model,
    X_validation,
    len(validation_scaled)
)

test_errors = calculate_point_errors(
    model,
    X_test,
    len(test_scaled)
)

# ==========================================
# THRESHOLD CALIBRATION
# ==========================================

threshold = np.percentile(
    validation_errors,
    THRESHOLD_PERCENTILE
)

predictions = (
    test_errors > threshold
).astype(np.int32)

print("\n========== THRESHOLD ==========")
print("Validation percentile:", THRESHOLD_PERCENTILE)
print("Reconstruction error threshold:", round(threshold, 6))

# ==========================================
# EVALUATION
# ==========================================

precision = precision_score(
    y_true,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_true,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_true,
    predictions,
    zero_division=0
)

auc = roc_auc_score(y_true, test_errors)

print("\n========== LSTM AUTOENCODER RESULTS ==========")
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("ROC-AUC:", round(auc, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, predictions))

# ==========================================
# SAVE RESULTS
# ==========================================

results = pd.DataFrame({
    "timestep": np.arange(len(test_values)),
    "actual_anomaly": y_true,
    "reconstruction_error": test_errors,
    "predicted_anomaly": predictions
})

results.to_csv(
    OUTPUT_DIR / "predictions.csv",
    index=False
)

metrics = pd.DataFrame([{
    "model": "LSTM Autoencoder",
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": auc,
    "threshold": threshold,
    "window_size": WINDOW_SIZE
}])

metrics.to_csv(
    OUTPUT_DIR / "metrics.csv",
    index=False
)

# ==========================================
# VISUALIZATION
# ==========================================

import matplotlib.pyplot as plt

fig, axes = plt.subplots(
    2, 1,
    figsize=(16, 9),
    sharex=True
)

axes[0].plot(test_values, linewidth=0.7)
axes[0].set_title("P-1 Test Telemetry")

axes[1].plot(
    test_errors,
    linewidth=0.7,
    label="Reconstruction Error"
)

axes[1].axhline(
    threshold,
    color="red",
    linestyle="--",
    label="Validation Threshold"
)

for start, end in anomaly_sequences:

    axes[0].axvspan(
        start, end,
        color="red",
        alpha=0.15
    )

    axes[1].axvspan(
        start, end,
        color="red",
        alpha=0.15
    )

axes[0].set_ylabel("Telemetry")
axes[1].set_ylabel("Reconstruction Error")
axes[1].set_xlabel("Timestep")

axes[0].legend(["Telemetry", "Labeled anomalies"])
axes[1].legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "reconstruction_errors.png",
    dpi=180
)

print("\n==========================================")
print("SPACEGUARD LSTM EXPERIMENT COMPLETED")
print("==========================================")
print("Model:", OUTPUT_DIR / "lstm_autoencoder.pth")
print("Predictions:", OUTPUT_DIR / "predictions.csv")
print("Metrics:", OUTPUT_DIR / "metrics.csv")
print("Plot:", OUTPUT_DIR / "reconstruction_errors.png")
