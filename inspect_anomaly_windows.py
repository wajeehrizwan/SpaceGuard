
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

test = pd.read_parquet(
    BASE_DIR / "data/raw/data/test/P-1.parquet"
)

anomalies = [
    (2149, 2349),
    (3539, 3779),
    (4536, 4844)
]

fig, axes = plt.subplots(3, 1, figsize=(16, 12))

for i, (start, end) in enumerate(anomalies):

    context = 200

    left = max(0, start - context)
    right = min(len(test), end + context)

    x = range(left, right)
    y = test["value"].iloc[left:right]

    axes[i].plot(x, y, linewidth=0.8)

    axes[i].axvspan(
        start,
        end,
        color="red",
        alpha=0.2,
        label="Labeled anomaly"
    )

    axes[i].axvline(start, color="red", linestyle="--")
    axes[i].axvline(end, color="red", linestyle="--")

    axes[i].set_title(
        f"Anomaly {i+1}: Timesteps {start}-{end}"
    )

    axes[i].set_ylabel("Telemetry")
    axes[i].legend()

axes[-1].set_xlabel("Timestep")

plt.tight_layout()

output = BASE_DIR / "outputs/eda/anomaly_zoom.png"
plt.savefig(output, dpi=180)

plt.show()

print(f"Saved to: {output}")
