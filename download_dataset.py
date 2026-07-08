"""
Download the yastrebksv/TrackNet tennis dataset from Google Drive.

Usage:
    python download_dataset.py
    python download_dataset.py --output datasets/trackNet

The dataset is downloaded from:
    https://drive.google.com/drive/folders/11r0RUaQHX7I3ANkaYG4jOxXK1OYo01Ut

After downloading, train with:
    python train.py --data datasets/trackNet --dataset-type v1 --batch 4 --epochs 30
"""

import argparse
import sys
from pathlib import Path


FOLDER_ID = "11r0RUaQHX7I3ANkaYG4jOxXK1OYo01Ut"


def parse_args():
    parser = argparse.ArgumentParser(description="Download TrackNet tennis dataset")
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/trackNet",
        help="Destination directory (default: datasets/trackNet)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)

    try:
        import gdown
    except ImportError:
        print("gdown is not installed. Run: pip install gdown==6.1.0")
        sys.exit(1)

    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"Directory '{output_dir}' already exists and is not empty.")
        answer = input("Continue and overwrite? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading TrackNet dataset to '{output_dir}' …")
    url = f"https://drive.google.com/drive/folders/{FOLDER_ID}"
    gdown.download_folder(url, output=str(output_dir), quiet=False, use_cookies=False)

    print(f"\nDataset ready at: {output_dir.resolve()}")
    print("\nTo start training:")
    print(f"  python train.py --data {args.output} --dataset-type v1 --batch 4 --epochs 30")


if __name__ == "__main__":
    main()
