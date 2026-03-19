"""
Preprocess the raw S3DIS dataset and (optionally) train the segmentation model.

Usage
-----
    # Step 1 – Preprocess raw S3DIS into HDF5 blocks
    python -m scripts.preprocess --config configs/default.yaml

    # Step 2 – Train
    python -m scripts.train --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.s3dis_dataset import preprocess_and_cache


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Preprocess S3DIS dataset")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = cfg["dataset"]

    print("=" * 60)
    print("  PropTech3D  –  S3DIS Preprocessing")
    print("=" * 60)
    print(f"  Raw root       : {ds['root']}")
    print(f"  Processed dir  : {ds['processed']}")
    print(f"  Block size     : {ds['block_size']} m")
    print(f"  Stride         : {ds['stride']} m")
    print(f"  Num points     : {ds['num_points']}")
    print()

    preprocess_and_cache(
        root=ds["root"],
        out_dir=ds["processed"],
        num_points=ds["num_points"],
        block_size=ds["block_size"],
        stride=ds["stride"],
    )

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
