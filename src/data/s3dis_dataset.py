"""
S3DIS dataset loader and preprocessor.

Handles the Stanford Large-Scale 3D Indoor Spaces Dataset:
  - 6 areas, 272 rooms, 13 semantic classes
  - Classes: ceiling, floor, wall, beam, column, window, door,
             table, chair, sofa, bookcase, board, clutter

The raw dataset is expected at `{root}/Area_X/room_name/*.txt` where each
txt file contains XYZ RGB per line for one object instance.

Preprocessing creates HDF5 caches of spatially-blocked, sub-sampled rooms
so that training is I/O-efficient.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

# S3DIS class list (official ordering)
S3DIS_CLASSES = [
    "ceiling", "floor", "wall", "beam", "column",
    "window", "door", "table", "chair", "sofa",
    "bookcase", "board", "clutter",
]
NUM_CLASSES = len(S3DIS_CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(S3DIS_CLASSES)}

# Colours for visualisation (RGB 0-255)
CLASS_COLOURS = np.array([
    [0, 255, 0],      # ceiling   – green
    [0, 0, 255],      # floor     – blue
    [0, 255, 255],    # wall      – cyan
    [255, 255, 0],    # beam      – yellow
    [255, 0, 255],    # column    – magenta
    [100, 100, 255],  # window    – light-blue
    [200, 200, 100],  # door      – khaki
    [170, 120, 200],  # table     – lavender
    [255, 0, 0],      # chair     – red
    [200, 100, 100],  # sofa      – salmon
    [10, 200, 100],   # bookcase  – teal
    [200, 200, 200],  # board     – grey
    [50, 50, 50],     # clutter   – dark-grey
], dtype=np.uint8)


# ──────────────────────────────────────────────
#  Raw S3DIS parser
# ──────────────────────────────────────────────

def parse_room(room_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a single S3DIS room directory into points (N, 6) and labels (N,).

    Each sub-file in the room dir is one annotated object instance
    named ``<ClassName>_<id>.txt``.
    """
    room_dir = Path(room_dir)
    annotations_dir = room_dir / "Annotations"
    if not annotations_dir.exists():
        raise FileNotFoundError(f"No Annotations dir in {room_dir}")

    all_points: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for fpath in sorted(annotations_dir.glob("*.txt")):
        class_name = fpath.stem.rsplit("_", 1)[0]
        if class_name not in CLASS_TO_IDX:
            class_name = "clutter"
        label = CLASS_TO_IDX[class_name]

        try:
            pts = np.loadtxt(fpath, dtype=np.float32)
        except ValueError:
            # Some files have malformed lines – skip them
            continue
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        if pts.shape[1] < 6:
            continue

        pts = pts[:, :6]  # XYZ RGB
        all_points.append(pts)
        all_labels.append(np.full(len(pts), label, dtype=np.int64))

    if not all_points:
        return np.empty((0, 6), dtype=np.float32), np.empty((0,), dtype=np.int64)

    points = np.concatenate(all_points, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return points, labels


def collect_rooms(root: str | Path) -> list[dict]:
    """Return metadata for every room across all 6 areas."""
    root = Path(root)
    rooms = []
    for area_idx in range(1, 7):
        area_dir = root / f"Area_{area_idx}"
        if not area_dir.exists():
            continue
        for room_dir in sorted(area_dir.iterdir()):
            if not room_dir.is_dir():
                continue
            rooms.append({
                "area": area_idx,
                "name": room_dir.name,
                "path": str(room_dir),
            })
    return rooms


# ──────────────────────────────────────────────
#  Block sampling
# ──────────────────────────────────────────────

def _decompose_blocks(
    points: np.ndarray,
    num_points: int = 4096,
    block_size: float = 1.0,
    stride: float = 0.5,
    return_indices: bool = False,
) -> tuple[np.ndarray, ...]:
    """Shared block decomposition used by training and inference paths.

    Parameters
    ----------
    points : (N, >=6) XYZ + RGB
    return_indices : if True, also return indices into the original array.

    Returns
    -------
    block_features : (B, num_points, 9) or empty
    block_indices  : (B, num_points) only if return_indices=True
    """
    xyz = points[:, :3]
    rgb = points[:, 3:6] / 255.0

    coord_min = xyz.min(axis=0)
    coord_max = xyz.max(axis=0)

    room_range = coord_max - coord_min
    room_range[room_range == 0] = 1.0
    xyz_norm = (xyz - coord_min) / room_range

    grid_x = np.arange(coord_min[0], coord_max[0], stride)
    grid_y = np.arange(coord_min[1], coord_max[1], stride)

    block_pts_list: list[np.ndarray] = []
    block_idx_list: list[np.ndarray] = []

    for gx in grid_x:
        for gy in grid_y:
            mask = (
                (xyz[:, 0] >= gx) & (xyz[:, 0] < gx + block_size) &
                (xyz[:, 1] >= gy) & (xyz[:, 1] < gy + block_size)
            )
            indices = np.where(mask)[0]
            if len(indices) < 100:
                continue

            bxyz = xyz[indices] - np.array([gx, gy, coord_min[2]])
            brgb = rgb[indices]
            bnorm = xyz_norm[indices]
            bfeat = np.concatenate(
                [bxyz, brgb, bnorm], axis=1
            ).astype(np.float32)

            n = len(indices)
            if n >= num_points:
                choice = np.random.choice(n, num_points, replace=False)
            else:
                choice = np.random.choice(n, num_points, replace=True)

            block_pts_list.append(bfeat[choice])
            if return_indices:
                block_idx_list.append(indices[choice])

    if not block_pts_list:
        empty_pts = np.empty((0, num_points, 9), dtype=np.float32)
        if return_indices:
            return empty_pts, np.empty((0, num_points), dtype=np.int64)
        return (empty_pts,)

    block_features = np.stack(block_pts_list)
    if return_indices:
        return block_features, np.stack(block_idx_list)
    return (block_features,)


def room_to_blocks(
    points: np.ndarray,
    labels: np.ndarray,
    num_points: int = 4096,
    block_size: float = 1.0,
    stride: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a room into fixed-size spatial blocks and sub-sample.

    Returns
    -------
    block_points : (B, num_points, 9)  – XYZ, RGB, normalised-XYZ
    block_labels : (B, num_points)
    """
    result = _decompose_blocks(
        points, num_points, block_size, stride, return_indices=True
    )
    block_features, block_indices = result

    if block_features.shape[0] == 0:
        return (
            np.empty((0, num_points, 9), dtype=np.float32),
            np.empty((0, num_points), dtype=np.int64),
        )

    block_labels = labels[block_indices]
    return block_features, block_labels


# ──────────────────────────────────────────────
#  HDF5 cache creation
# ──────────────────────────────────────────────

def preprocess_and_cache(
    root: str | Path,
    out_dir: str | Path,
    num_points: int = 4096,
    block_size: float = 1.0,
    stride: float = 0.5,
) -> None:
    """Preprocess all S3DIS rooms into per-area HDF5 caches."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rooms = collect_rooms(root)
    area_data: dict[int, list] = {}
    for r in rooms:
        area_data.setdefault(r["area"], []).append(r)

    for area_idx, area_rooms in sorted(area_data.items()):
        h5_path = out_dir / f"area_{area_idx}.h5"
        if h5_path.exists():
            print(f"[skip] {h5_path} already exists")
            continue

        all_pts, all_lbl = [], []
        for room in tqdm(area_rooms, desc=f"Area {area_idx}"):
            try:
                points, labels = parse_room(room["path"])
            except Exception as e:
                print(f"  [warn] {room['name']}: {e}")
                continue
            bp, bl = room_to_blocks(
                points, labels, num_points, block_size, stride
            )
            if bp.shape[0] == 0:
                continue
            all_pts.append(bp)
            all_lbl.append(bl)

        if not all_pts:
            print(f"  [warn] Area {area_idx}: no valid blocks")
            continue

        all_pts = np.concatenate(all_pts)
        all_lbl = np.concatenate(all_lbl)

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("points", data=all_pts, compression="gzip")
            f.create_dataset("labels", data=all_lbl, compression="gzip")
        print(f"[done] {h5_path}  ({all_pts.shape[0]} blocks)")


# ──────────────────────────────────────────────
#  PyTorch Dataset
# ──────────────────────────────────────────────

class S3DISDataset(Dataset):
    """PyTorch dataset backed by preprocessed HDF5 caches.

    Parameters
    ----------
    processed_dir : path to the directory containing ``area_X.h5`` files.
    test_area     : area index (1-6) held out for testing.
    split         : ``"train"`` or ``"test"``.
    augment       : apply random rotation + jitter at training time.
    """

    def __init__(
        self,
        processed_dir: str | Path,
        test_area: int = 5,
        split: str = "train",
        augment: bool = True,
    ):
        super().__init__()
        self.split = split
        self.augment = augment and (split == "train")

        processed_dir = Path(processed_dir)
        areas = list(range(1, 7))
        if split == "train":
            areas = [a for a in areas if a != test_area]
        else:
            areas = [test_area]

        all_pts, all_lbl = [], []
        for a in areas:
            h5_path = processed_dir / f"area_{a}.h5"
            if not h5_path.exists():
                print(f"[warn] {h5_path} not found, skipping area {a}.")
                continue
            with h5py.File(h5_path, "r") as f:
                all_pts.append(f["points"][:])
                all_lbl.append(f["labels"][:])

        if not all_pts:
            raise FileNotFoundError(
                f"No HDF5 files found in {processed_dir} for areas {areas}. "
                f"Run preprocessing first."
            )

        self.points = np.concatenate(all_pts)  # (N, P, 9)
        self.labels = np.concatenate(all_lbl)  # (N, P)

    def __len__(self) -> int:
        return self.points.shape[0]

    def __getitem__(self, idx: int):
        pts = self.points[idx].copy()   # (P, 9)
        lbl = self.labels[idx].copy()   # (P,)

        if self.augment:
            pts = self._augment(pts)

        return (
            torch.from_numpy(pts).float(),   # (P, 9)
            torch.from_numpy(lbl).long(),    # (P,)
        )

    @staticmethod
    def _augment(pts: np.ndarray) -> np.ndarray:
        """Random rotation around Z-axis + small jitter."""
        theta = np.random.uniform(0, 2 * np.pi)
        cos, sin = np.cos(theta), np.sin(theta)
        rot = np.array([[cos, -sin, 0],
                        [sin,  cos, 0],
                        [0,    0,   1]], dtype=np.float32)
        pts[:, :3] = pts[:, :3] @ rot.T
        # Also rotate the normalised-XYZ channels (6:9)
        pts[:, 6:9] = pts[:, 6:9] @ rot.T
        # Small XYZ jitter
        pts[:, :3] += np.random.normal(0, 0.01, pts[:, :3].shape).astype(
            np.float32
        )
        return pts


# ──────────────────────────────────────────────
#  Whole-room loader (for inference / floor plan)
# ──────────────────────────────────────────────

def load_room_for_inference(
    room_path: str | Path,
    block_size: float = 1.0,
    stride: float = 0.5,
    num_points: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and block-decompose a single room for inference.

    Returns
    -------
    raw_points : (N, 6) original XYZ + RGB
    block_points : (B, num_points, 9)
    block_indices : (B, num_points) indices into raw_points
    """
    room_path = Path(room_path)

    # Detect format: directory (S3DIS), .txt, .ply, .npy
    if room_path.is_dir():
        raw_points, _ = parse_room(room_path)
    elif room_path.suffix == ".npy":
        raw_points = np.load(room_path).astype(np.float32)
    elif room_path.suffix == ".txt":
        raw_points = np.loadtxt(room_path, dtype=np.float32)
    elif room_path.suffix == ".ply":
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError(
                "open3d is required to load .ply files but is not installed. "
                "Install it with 'pip install open3d' (requires Python <=3.12). "
                "Alternatively, convert your .ply to .txt (XYZRGB) or .npy."
            )
        pcd = o3d.io.read_point_cloud(str(room_path))
        xyz = np.asarray(pcd.points, dtype=np.float32)
        rgb = np.asarray(pcd.colors, dtype=np.float32) * 255.0
        raw_points = np.hstack([xyz, rgb])
    else:
        raise ValueError(
            f"Unsupported point cloud format: {room_path.suffix}. "
            f"Supported: directory (S3DIS), .npy, .txt, .ply"
        )

    if raw_points.shape[1] < 6:
        # No colour – pad with zeros
        raw_points = np.hstack([
            raw_points[:, :3],
            np.zeros((len(raw_points), 3), dtype=np.float32),
        ])

    result = _decompose_blocks(
        raw_points, num_points, block_size, stride, return_indices=True
    )
    block_points, block_indices = result
    return raw_points, block_points, block_indices
