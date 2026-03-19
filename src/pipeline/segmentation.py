"""
Inference module – run a trained model on an arbitrary point cloud.

Handles block decomposition, batched GPU inference, and vote aggregation
back to the original point cloud.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.data.s3dis_dataset import load_room_for_inference, NUM_CLASSES
from src.models.pointnet import build_model


class Segmentor:
    """Wraps a trained model for whole-room inference."""

    def __init__(
        self,
        checkpoint: str | Path,
        model_name: str = "pointnet",
        input_channels: int = 9,
        num_classes: int = NUM_CLASSES,
        device: str | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.num_classes = num_classes

        self.model = build_model(model_name, input_channels, num_classes)
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        # Auto-detect model architecture from checkpoint if available
        saved_name = ckpt.get("model_name")
        if saved_name and saved_name != model_name:
            self.model = build_model(saved_name, input_channels, num_classes)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict_room(
        self,
        room_path: str | Path,
        block_size: float = 1.0,
        stride: float = 0.5,
        num_points: int = 4096,
        batch_size: int = 64,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run segmentation on a single room.

        Parameters
        ----------
        room_path : path to a room directory, .ply, .txt, or .npy file.

        Returns
        -------
        raw_points : (N, 6) original XYZ + RGB
        labels     : (N,)   predicted class per point (majority vote)
        """
        raw_points, block_points, block_indices = load_room_for_inference(
            room_path, block_size, stride, num_points
        )
        N = raw_points.shape[0]
        vote_counts = np.zeros((N, self.num_classes), dtype=np.float32)

        num_blocks = block_points.shape[0]
        for start in range(0, num_blocks, batch_size):
            end = min(start + batch_size, num_blocks)
            batch = torch.from_numpy(block_points[start:end]).to(self.device)
            logits, _ = self.model(batch)          # (B, P, C)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            indices = block_indices[start:end]     # (B, P)
            for b in range(probs.shape[0]):
                np.add.at(vote_counts, indices[b], probs[b])

        labels = vote_counts.argmax(axis=1)
        return raw_points, labels
