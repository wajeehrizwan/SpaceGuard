
from huggingface_hub import hf_hub_download
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "raw"

REPO_ID = "appleparan/telemanom"
CHANNEL = "P-1"

FILES = [
    f"data/train/{CHANNEL}.parquet",
    f"data/test/{CHANNEL}.parquet"
]

for filename in FILES:
    print(f"\nDownloading {filename}")

    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
        local_dir=str(DATA_DIR)
    )

    print("Saved:", path)

print("\nP-1 DOWNLOAD COMPLETE!")
