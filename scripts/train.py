"""
Train the point-cloud semantic segmentation model.

Usage
-----
    python -m scripts.train --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.train import Trainer


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
        # Keep cosine scheduler T_max in sync
        if cfg["train"]["scheduler"]["name"] == "cosine":
            cfg["train"]["scheduler"]["T_max"] = args.epochs

    print("=" * 60)
    print("  PropTech3D  –  Training")
    print("=" * 60)

    trainer = Trainer(cfg)
    if args.resume:
        trainer.load(args.resume)

    trainer.run()


if __name__ == "__main__":
    main()
