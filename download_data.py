
from huggingface_hub import hf_hub_download
from pathlib import Path

# ==========================================
# NASA TELEMETRY DATASET DOWNLOADER
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "raw"

DATA_DIR.mkdir(parents=True, exist_ok=True)

REPO_ID = "appleparan/telemanom"

FILES = [
    "data/train/A-1.parquet",
    "data/test/A-1.parquet",
    "labeled_anomalies.csv"
]

# ==========================================
# DOWNLOAD DATASET
# ==========================================

for filename in FILES:

    print("\n" + "=" * 50)
    print(f"Downloading: {filename}")
    print("=" * 50)

    try:
        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=str(DATA_DIR)
        )

        print(f"SUCCESS: {downloaded_path}")

    except Exception as error:
        print(f"DOWNLOAD FAILED: {filename}")
        print(f"Error: {error}")
        raise

print("\n" + "=" * 50)
print("NASA TELEMETRY DATASET DOWNLOAD COMPLETE!")
print("=" * 50)
