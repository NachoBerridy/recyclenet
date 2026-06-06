"""
RecycleNet Dataset Downloader

Downloads all required public datasets from Kaggle and stores them
inside the project's data/raw directory.

Requirements:
    pip install kagglehub

Usage:
    python scripts/download_dataset.py
"""

from pathlib import Path
import shutil
import kagglehub


def main():
    print("=" * 60)
    print("RecycleNet - Dataset Downloader")
    print("=" * 60)

    project_root = Path(__file__).resolve().parent.parent

    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Dataset 1
    # ------------------------------------------------------------------
    print("\nDownloading Dataset 1 (Base Classification Dataset)...")

    ds1_path = kagglehub.dataset_download(
        "zlatan599/garbage-dataset-classification"
    )

    target_ds1 = raw_dir / "garbage-dataset-classification"

    if not target_ds1.exists():
        shutil.copytree(ds1_path, target_ds1)
        print(f"Dataset 1 saved to: {target_ds1}")
    else:
        print("Dataset 1 already exists. Skipping copy.")

    # ------------------------------------------------------------------
    # Dataset 2
    # ------------------------------------------------------------------
    print("\nDownloading Dataset 2 (Extended Waste Dataset)...")

    ds2_path = kagglehub.dataset_download(
        "mostafaabla/garbage-classification"
    )

    target_ds2 = raw_dir / "garbage-classification"

    if not target_ds2.exists():
        shutil.copytree(ds2_path, target_ds2)
        print(f"Dataset 2 saved to: {target_ds2}")
    else:
        print("Dataset 2 already exists. Skipping copy.")

    print("\nDatasets downloaded successfully.")
    print(f"Location: {raw_dir}")


if __name__ == "__main__":
    main()