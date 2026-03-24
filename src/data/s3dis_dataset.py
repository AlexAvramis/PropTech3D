"""
S3DIS dataset loader and preprocessor.

Handles the Stanford Large-Scale 3D Indoor Spaces Dataset:
  - 6 areas, 272 rooms, 13 semantic classes
  - Classes: ceiling, floor, wall, beam, column, window, door,
             table, chair, sofa, bookcase, board, clutter

The raw dataset is expected at `{root}/Area_X/room_name/*.txt` where each
txt file contains XYZ RGB per line for one object instance.

Preprocessing creates per-area HDF5 caches that store **full rooms** (no
sub-sampling).  Block decomposition and random point sampling happen
on-the-fly in the Dataset so that every epoch sees different subsets.
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
    full_coverage: bool = False,
) -> tuple[np.ndarray, ...]:
    """Shared block decomposition used by training and inference paths.

    Parameters
    ----------
    points : (N, >=6) XYZ + RGB
    return_indices : if True, also return indices into the original array.
    full_coverage  : if True (inference mode), ALL points in each block
                     are covered by splitting into multiple sub-blocks of
                     ``num_points`` so no point is left without a vote.
                     If False (training / preprocessing), a single random
                     sample per block is drawn (standard training approach).

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

            if full_coverage:
                # Inference: cover ALL points by splitting into sub-blocks
                # so every point gets at least one vote.
                order = np.arange(n)
                np.random.shuffle(order)
                for start in range(0, n, num_points):
                    end = start + num_points
                    if end <= n:
                        choice = order[start:end]
                    else:
                        # Last chunk: pad with random repeats to fill
                        remainder = order[start:]
                        pad = np.random.choice(n, num_points - len(remainder),
                                               replace=True)
                        choice = np.concatenate([remainder, pad])
                    block_pts_list.append(bfeat[choice])
                    if return_indices:
                        block_idx_list.append(indices[choice])
            else:
                # Training / preprocessing: single random sample per block
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
    """Preprocess all S3DIS rooms into per-area HDF5 caches.

    Stores **full room point clouds** (no sub-sampling) so that the
    Dataset can draw different random subsets each epoch.
    """
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

        total_pts = 0
        with h5py.File(h5_path, "w") as f:
            rooms_grp = f.create_group("rooms")
            for room in tqdm(area_rooms, desc=f"Area {area_idx}"):
                try:
                    points, labels = parse_room(room["path"])
                except Exception as e:
                    print(f"  [warn] {room['name']}: {e}")
                    continue
                if len(points) == 0:
                    continue
                rg = rooms_grp.create_group(room["name"])
                rg.create_dataset("points", data=points, compression="gzip")
                rg.create_dataset("labels", data=labels, compression="gzip")
                total_pts += len(points)

        print(f"[done] {h5_path}  ({total_pts:,} points in "
              f"{len(area_rooms)} rooms)")


# ──────────────────────────────────────────────
#  PyTorch Dataset
# ──────────────────────────────────────────────

class S3DISDataset(Dataset):
    """PyTorch dataset with **on-the-fly** block sampling.

    New-format HDF5 files store full rooms.  At each ``__getitem__``
    call the dataset randomly sub-samples ``num_points`` from the
    selected block, so every epoch sees different point subsets.

    Falls back to legacy format (pre-sampled static blocks) when
    the ``rooms`` group is absent.

    Parameters
    ----------
    processed_dir : path to the directory containing ``area_X.h5`` files.
    test_area     : area index (1-6) held out for testing.
    split         : ``"train"`` or ``"test"``.
    augment       : apply data augmentation at training time.
    num_points    : points per block sample.
    block_size    : spatial block size in metres.
    stride        : block stride in metres.
    """

    def __init__(
        self,
        processed_dir: str | Path,
        test_area: int = 5,
        split: str = "train",
        augment: bool = True,
        num_points: int = 4096,
        block_size: float = 1.0,
        stride: float = 0.5,
    ):
        super().__init__()
        self.split = split
        self.augment = augment and (split == "train")
        self.num_points = num_points
        self.block_size = block_size

        processed_dir = Path(processed_dir)
        areas = list(range(1, 7))
        if split == "train":
            areas = [a for a in areas if a != test_area]
        else:
            areas = [test_area]

        # Try to load new-format (full rooms) first
        self._dynamic = False
        self.rooms: list[tuple[np.ndarray, np.ndarray, np.ndarray,
                               np.ndarray]] = []
        # Each entry: (points (N,6), labels (N,), coord_min (3,), room_range (3,))
        self.blocks: list[tuple[int, np.ndarray, float, float]] = []
        # Each entry: (room_idx, indices_into_room, gx, gy)

        loaded_any = False
        for a in areas:
            h5_path = processed_dir / f"area_{a}.h5"
            if not h5_path.exists():
                print(f"[warn] {h5_path} not found, skipping area {a}.")
                continue
            loaded_any = True
            with h5py.File(h5_path, "r") as f:
                if "rooms" in f:
                    self._dynamic = True
                    self._load_rooms(f, block_size, stride)
                else:
                    # Legacy static-block format
                    if not hasattr(self, "_legacy_pts"):
                        self._legacy_pts: list[np.ndarray] = []
                        self._legacy_lbl: list[np.ndarray] = []
                    self._legacy_pts.append(f["points"][:])
                    self._legacy_lbl.append(f["labels"][:])

        if not loaded_any:
            raise FileNotFoundError(
                f"No HDF5 files found in {processed_dir} for areas {areas}. "
                f"Run preprocessing first."
            )

        if not self._dynamic:
            # Legacy path
            self.points = np.concatenate(self._legacy_pts)
            self.labels = np.concatenate(self._legacy_lbl)
            del self._legacy_pts, self._legacy_lbl

    # ---- new format helpers ----

    def _load_rooms(self, h5: h5py.File, block_size: float, stride: float):
        rooms_grp = h5["rooms"]
        for room_key in rooms_grp:
            pts = rooms_grp[room_key]["points"][:].astype(np.float32)
            lbl = rooms_grp[room_key]["labels"][:].astype(np.int64)
            if len(pts) == 0:
                continue
            xyz = pts[:, :3]
            coord_min = xyz.min(axis=0)
            coord_max = xyz.max(axis=0)
            room_range = coord_max - coord_min
            room_range[room_range == 0] = 1.0

            room_idx = len(self.rooms)
            self.rooms.append((pts, lbl, coord_min, room_range))

            # Pre-compute block membership
            for gx in np.arange(coord_min[0], coord_max[0], stride):
                for gy in np.arange(coord_min[1], coord_max[1], stride):
                    mask = (
                        (xyz[:, 0] >= gx) & (xyz[:, 0] < gx + block_size) &
                        (xyz[:, 1] >= gy) & (xyz[:, 1] < gy + block_size)
                    )
                    indices = np.where(mask)[0]
                    if len(indices) < 100:
                        continue
                    self.blocks.append((room_idx, indices, gx, gy))

    # ---- interface ----

    def __len__(self) -> int:
        if self._dynamic:
            return len(self.blocks)
        return self.points.shape[0]

    def __getitem__(self, idx: int):
        if self._dynamic:
            return self._getitem_dynamic(idx)
        return self._getitem_legacy(idx)

    def _getitem_dynamic(self, idx: int):
        room_idx, indices, gx, gy = self.blocks[idx]
        pts, lbl, coord_min, room_range = self.rooms[room_idx]

        # Random sub-sample — different every epoch
        n = len(indices)
        if n >= self.num_points:
            choice = np.random.choice(n, self.num_points, replace=False)
        else:
            choice = np.random.choice(n, self.num_points, replace=True)
        sel = indices[choice]

        xyz = pts[sel, :3]
        rgb = pts[sel, 3:6] / 255.0
        xyz_block = xyz - np.array([gx, gy, coord_min[2]], dtype=np.float32)
        xyz_norm = (xyz - coord_min) / room_range

        feat = np.concatenate(
            [xyz_block, rgb, xyz_norm], axis=1
        ).astype(np.float32)
        labels = lbl[sel].copy()

        if self.augment:
            feat = self._augment(feat)

        return (
            torch.from_numpy(feat).float(),
            torch.from_numpy(labels).long(),
        )

    def _getitem_legacy(self, idx: int):
        pts = self.points[idx].copy()
        lbl = self.labels[idx].copy()
        if self.augment:
            pts = self._augment(pts)
        return (
            torch.from_numpy(pts).float(),
            torch.from_numpy(lbl).long(),
        )

    @staticmethod
    def _augment(pts: np.ndarray) -> np.ndarray:
        """Data augmentation: rotation, scale, jitter, colour noise."""
        # Random rotation around Z-axis
        theta = np.random.uniform(0, 2 * np.pi)
        cos, sin = np.cos(theta), np.sin(theta)
        rot = np.array([[cos, -sin, 0],
                        [sin,  cos, 0],
                        [0,    0,   1]], dtype=np.float32)
        pts[:, :3] = pts[:, :3] @ rot.T
        pts[:, 6:9] = pts[:, 6:9] @ rot.T

        # Random scale (0.9 – 1.1)
        scale = np.random.uniform(0.9, 1.1)
        pts[:, :3] *= scale

        # XYZ jitter
        pts[:, :3] += np.random.normal(0, 0.02, pts[:, :3].shape).astype(
            np.float32
        )

        # Colour jitter on RGB channels (3:6), clamp to [0, 1]
        pts[:, 3:6] += np.random.normal(0, 0.05, pts[:, 3:6].shape).astype(
            np.float32
        )
        np.clip(pts[:, 3:6], 0.0, 1.0, out=pts[:, 3:6])

        # Random colour drop (10% chance — set RGB to 0)
        if np.random.random() < 0.1:
            pts[:, 3:6] = 0.0

        return pts


def compute_class_weights(
    processed_dir: str | Path,
    test_area: int = 5,
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    """Compute sqrt-inverse-frequency class weights from the training set.

    Returns
    -------
    weights : (num_classes,) float32 array, normalised so mean = 1.
    """
    processed_dir = Path(processed_dir)
    counts = np.zeros(num_classes, dtype=np.float64)
    areas = [a for a in range(1, 7) if a != test_area]

    for a in areas:
        h5_path = processed_dir / f"area_{a}.h5"
        if not h5_path.exists():
            continue
        with h5py.File(h5_path, "r") as f:
            if "rooms" in f:
                for rk in f["rooms"]:
                    lbl = f[f"rooms/{rk}/labels"][:]
                    for c in range(num_classes):
                        counts[c] += (lbl == c).sum()
            else:
                lbl = f["labels"][:]
                for c in range(num_classes):
                    counts[c] += (lbl == c).sum()

    # Avoid division by zero for absent classes
    counts = np.maximum(counts, 1.0)
    total = counts.sum()
    freq = counts / total
    weights = 1.0 / np.sqrt(freq)
    weights /= weights.mean()  # normalise so mean weight = 1
    return weights.astype(np.float32)


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
        raw_points, num_points, block_size, stride,
        return_indices=True, full_coverage=True,
    )
    block_points, block_indices = result
    return raw_points, block_points, block_indices
