"""
Training loop for point-cloud semantic segmentation.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.s3dis_dataset import S3DISDataset, NUM_CLASSES
from src.models.pointnet import build_model, pointnet_loss


# ──────────────────────────────────────────────
#  Metrics
# ──────────────────────────────────────────────

def compute_metrics(
    pred: np.ndarray, target: np.ndarray, num_classes: int
) -> dict:
    """Compute OA, mAcc, mIoU from flat arrays."""
    oa = (pred == target).mean()

    per_class_iou = []
    per_class_acc = []
    for c in range(num_classes):
        tp = ((pred == c) & (target == c)).sum()
        fp = ((pred == c) & (target != c)).sum()
        fn = ((pred != c) & (target == c)).sum()
        total_c = (target == c).sum()

        iou = tp / max(tp + fp + fn, 1)
        acc = tp / max(total_c, 1)
        per_class_iou.append(iou)
        per_class_acc.append(acc)

    return {
        "OA": float(oa),
        "mAcc": float(np.mean(per_class_acc)),
        "mIoU": float(np.mean(per_class_iou)),
        "per_class_iou": [float(v) for v in per_class_iou],
    }


# ──────────────────────────────────────────────
#  Trainer
# ──────────────────────────────────────────────

class Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Data
        ds_cfg = cfg["dataset"]
        self.train_ds = S3DISDataset(
            ds_cfg["processed"], ds_cfg["test_area"], split="train"
        )
        self.val_ds = S3DISDataset(
            ds_cfg["processed"], ds_cfg["test_area"], split="test", augment=False
        )
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=cfg["train"]["batch_size"],
            shuffle=True,
            num_workers=cfg["train"]["num_workers"],
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=cfg["train"]["batch_size"],
            shuffle=False,
            num_workers=cfg["train"]["num_workers"],
            pin_memory=True,
        )

        # Model
        m_cfg = cfg["model"]
        self.model = build_model(
            m_cfg["name"],
            m_cfg["input_channels"],
            ds_cfg["num_classes"],
            m_cfg["dropout"],
        ).to(self.device)

        # Optimiser & scheduler
        t_cfg = cfg["train"]
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=t_cfg["lr"],
            weight_decay=t_cfg["weight_decay"],
        )
        sched_cfg = t_cfg["scheduler"]
        if sched_cfg["name"] == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=sched_cfg["T_max"]
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.5
            )

        # Paths
        self.ckpt_dir = Path(cfg["paths"]["checkpoints"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(cfg["paths"]["logs"])
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.best_miou = 0.0
        self.start_epoch = 1

    # -------- train one epoch --------
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        losses = []
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [train]", leave=False)
        for pts, lbl in pbar:
            pts = pts.to(self.device)    # (B, N, 9)
            lbl = lbl.to(self.device)    # (B, N)

            logits, t_feat = self.model(pts)
            loss = pointnet_loss(logits, lbl, t_feat)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return float(np.mean(losses))

    # -------- evaluate --------
    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        all_pred, all_target = [], []
        for pts, lbl in tqdm(self.val_loader, desc="  [eval]", leave=False):
            pts = pts.to(self.device)
            logits, _ = self.model(pts)
            pred = logits.argmax(dim=-1).cpu().numpy()
            all_pred.append(pred.reshape(-1))
            all_target.append(lbl.numpy().reshape(-1))

        all_pred = np.concatenate(all_pred)
        all_target = np.concatenate(all_target)
        return compute_metrics(all_pred, all_target, NUM_CLASSES)

    # -------- full training run --------
    def run(self):
        num_epochs = self.cfg["train"]["epochs"]
        print(f"Training on {self.device}  |  {len(self.train_ds)} train, "
              f"{len(self.val_ds)} val blocks")
        print(f"Model: {self.cfg['model']['name']}  |  "
              f"Params: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.start_epoch, num_epochs + 1):
            t0 = time.time()
            loss = self.train_epoch(epoch)
            metrics = self.evaluate()
            self.scheduler.step()

            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d}/{num_epochs}  "
                f"loss={loss:.4f}  OA={metrics['OA']:.4f}  "
                f"mAcc={metrics['mAcc']:.4f}  mIoU={metrics['mIoU']:.4f}  "
                f"lr={lr:.6f}  ({elapsed:.1f}s)"
            )

            # Save latest
            self._save(epoch, metrics, self.ckpt_dir / "latest.pth")

            # Save best
            if metrics["mIoU"] > self.best_miou:
                self.best_miou = metrics["mIoU"]
                self._save(epoch, metrics, self.ckpt_dir / "best.pth")
                print(f"  ★ New best mIoU = {self.best_miou:.4f}")

        print("Training complete.")

    # -------- checkpoint I/O --------
    def _save(self, epoch: int, metrics: dict, path: Path):
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "best_miou": self.best_miou,
        }, path)

    def load(self, path: str | Path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_miou = ckpt.get("best_miou", ckpt["metrics"].get("mIoU", 0.0))
        print(f"Loaded checkpoint from {path}  (epoch {ckpt['epoch']}, "
              f"mIoU={ckpt['metrics']['mIoU']:.4f})")
