import argparse
import os
from pathlib import Path

from .config import RAW_DIR
from .utils import ensure_dir


def kaggle_credentials_available() -> bool:
    env_ok = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
    file_ok = (Path.home() / ".kaggle" / "kaggle.json").exists()
    return env_ok or file_ok


def download_dataset(dataset: str, output_dir: Path, force: bool) -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as error:
        raise RuntimeError(
            "Kaggle package is not installed. Install dependencies with: pip install -r requirements.txt"
        ) from error

    if not kaggle_credentials_available():
        raise RuntimeError(
            "Kaggle credentials not found. Set KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json"
        )

    ensure_dir(output_dir)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        dataset=dataset,
        path=str(output_dir),
        unzip=True,
        quiet=False,
        force=force,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Download MRL Eye dataset from Kaggle.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="imadeddinedjerarda/mrl-eye-dataset",
        help="Kaggle dataset slug in owner/dataset format.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RAW_DIR,
        help="Where to download/unzip dataset.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download if files already exist.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Downloading dataset: {args.dataset}")
    print(f"Destination: {args.output_dir}")
    download_dataset(dataset=args.dataset, output_dir=args.output_dir, force=args.force)
    print("Download complete.")


if __name__ == "__main__":
    main()
