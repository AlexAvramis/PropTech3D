"""
Generate synthetic S3DIS-style room data for testing the pipeline.

Creates fake rooms with floor, walls, ceiling, and furniture-like point
clusters so the full train → segment → floor-plan pipeline can be verified
without downloading the real dataset.

Usage
-----
    python -m scripts.generate_synthetic --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.s3dis_dataset import (
    S3DIS_CLASSES, CLASS_TO_IDX, preprocess_and_cache,
)

# Typical RGB colours for each class (for realism in visualisation)
_CLASS_RGB = {
    "ceiling":  (220, 220, 220),
    "floor":    (150, 120, 90),
    "wall":     (200, 200, 180),
    "beam":     (180, 160, 100),
    "column":   (160, 160, 170),
    "window":   (140, 200, 255),
    "door":     (139, 90, 43),
    "table":    (120, 80, 60),
    "chair":    (200, 50, 50),
    "sofa":     (80, 120, 180),
    "bookcase": (100, 70, 40),
    "board":    (240, 240, 240),
    "clutter":  (60, 60, 60),
}


def _make_box_points(
    x_range: tuple, y_range: tuple, z_range: tuple,
    density: float = 500.0,
    surface_only: bool = True,
) -> np.ndarray:
    """Generate points on (or filling) an axis-aligned box."""
    xmin, xmax = x_range
    ymin, ymax = y_range
    zmin, zmax = z_range
    sx, sy, sz = xmax - xmin, ymax - ymin, zmax - zmin

    if surface_only:
        n_total = int(density * 2 * (sx * sy + sy * sz + sx * sz))
        n_total = max(n_total, 50)

        pts = []
        # XY faces (floor / ceiling of box)
        n_xy = max(int(n_total * sx * sy / (sx * sy + sy * sz + sx * sz)), 10)
        for z in [zmin, zmax]:
            xy = np.column_stack([
                np.random.uniform(xmin, xmax, n_xy),
                np.random.uniform(ymin, ymax, n_xy),
                np.full(n_xy, z),
            ])
            pts.append(xy)
        # XZ faces
        n_xz = max(int(n_total * sx * sz / (sx * sy + sy * sz + sx * sz)), 10)
        for y in [ymin, ymax]:
            xz = np.column_stack([
                np.random.uniform(xmin, xmax, n_xz),
                np.full(n_xz, y),
                np.random.uniform(zmin, zmax, n_xz),
            ])
            pts.append(xz)
        # YZ faces
        n_yz = max(int(n_total * sy * sz / (sx * sy + sy * sz + sx * sz)), 10)
        for x in [xmin, xmax]:
            yz = np.column_stack([
                np.full(n_yz, x),
                np.random.uniform(ymin, ymax, n_yz),
                np.random.uniform(zmin, zmax, n_yz),
            ])
            pts.append(yz)
        return np.concatenate(pts).astype(np.float32)
    else:
        n = int(density * sx * sy * sz)
        n = max(n, 50)
        return np.column_stack([
            np.random.uniform(xmin, xmax, n),
            np.random.uniform(ymin, ymax, n),
            np.random.uniform(zmin, zmax, n),
        ]).astype(np.float32)


def _add_noise(pts: np.ndarray, sigma: float = 0.005) -> np.ndarray:
    return pts + np.random.normal(0, sigma, pts.shape).astype(np.float32)


def generate_room(
    room_width: float = 6.0,
    room_depth: float = 5.0,
    room_height: float = 3.0,
    wall_thickness: float = 0.12,
    density: float = 800.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic room and return (points, labels).

    Returns (N, 6) points [XYZ RGB] and (N,) int labels.
    """
    rng = np.random.default_rng()
    all_pts = []
    all_lbl = []

    def _emit(xyz: np.ndarray, class_name: str):
        xyz = _add_noise(xyz)
        rgb_base = np.array(_CLASS_RGB[class_name], dtype=np.float32)
        rgb = np.tile(rgb_base, (len(xyz), 1))
        rgb += np.random.normal(0, 8, rgb.shape).astype(np.float32)
        rgb = np.clip(rgb, 0, 255)
        all_pts.append(np.hstack([xyz, rgb]))
        all_lbl.append(np.full(len(xyz), CLASS_TO_IDX[class_name], dtype=np.int64))

    W, D, H = room_width, room_depth, room_height
    t = wall_thickness

    # Floor
    _emit(_make_box_points((0, W), (0, D), (0, t), density), "floor")

    # Ceiling
    _emit(_make_box_points((0, W), (0, D), (H - t, H), density), "ceiling")

    # 4 Walls
    _emit(_make_box_points((0, t), (0, D), (0, H), density), "wall")       # left
    _emit(_make_box_points((W - t, W), (0, D), (0, H), density), "wall")   # right
    _emit(_make_box_points((0, W), (0, t), (0, H), density), "wall")       # front
    _emit(_make_box_points((0, W), (D - t, D), (0, H), density), "wall")   # back

    # Door (gap in front wall, with door frame points)
    door_x = rng.uniform(1.0, W - 1.5)
    door_w, door_h = 0.9, 2.1
    _emit(_make_box_points(
        (door_x, door_x + door_w), (0, t * 2), (0, min(door_h, H - 0.1)),
        density * 0.5,
    ), "door")

    # Window (on back wall)
    win_x = rng.uniform(0.5, W - 1.5)
    win_w, win_h = 1.2, 1.0
    win_z = 1.0
    _emit(_make_box_points(
        (win_x, win_x + win_w), (D - t * 2, D), (win_z, win_z + win_h),
        density * 0.5,
    ), "window")

    # Table
    tx = rng.uniform(1.5, W - 2.0)
    ty = rng.uniform(1.5, D - 2.0)
    _emit(_make_box_points((tx, tx + 1.2), (ty, ty + 0.6), (0.6, 0.75), density * 0.4), "table")
    # Table legs
    for lx, ly in [(tx, ty), (tx + 1.1, ty), (tx, ty + 0.5), (tx + 1.1, ty + 0.5)]:
        _emit(_make_box_points((lx, lx + 0.05), (ly, ly + 0.05), (0, 0.6), density * 0.2), "table")

    # Chairs around table
    for cx, cy in [(tx - 0.5, ty + 0.15), (tx + 1.4, ty + 0.15)]:
        if 0.3 < cx < W - 0.6 and 0.3 < cy < D - 0.6:
            _emit(_make_box_points(
                (cx, cx + 0.45), (cy, cy + 0.45), (0.3, 0.47), density * 0.3
            ), "chair")
            # Chair back
            _emit(_make_box_points(
                (cx, cx + 0.45), (cy, cy + 0.05), (0.47, 0.9), density * 0.2
            ), "chair")

    # Bookcase against a wall
    bx = rng.uniform(0.2, 0.5)
    _emit(_make_box_points(
        (W - bx - 0.4, W - bx), (0.3, 1.5), (0, 1.8), density * 0.3
    ), "bookcase")

    # Some clutter
    for _ in range(rng.integers(2, 6)):
        cx = rng.uniform(0.5, W - 0.5)
        cy = rng.uniform(0.5, D - 0.5)
        cs = rng.uniform(0.1, 0.3)
        _emit(_make_box_points(
            (cx, cx + cs), (cy, cy + cs), (0, rng.uniform(0.1, 0.5)),
            density * 0.2
        ), "clutter")

    points = np.concatenate(all_pts).astype(np.float32)
    labels = np.concatenate(all_lbl)
    return points, labels


def write_s3dis_room(out_dir: Path, points: np.ndarray, labels: np.ndarray):
    """Write points in S3DIS directory format: Annotations/<Class>_<id>.txt"""
    ann_dir = out_dir / "Annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    class_counts: dict[str, int] = {}
    for cls_idx in np.unique(labels):
        cls_name = S3DIS_CLASSES[cls_idx]
        mask = labels == cls_idx
        cls_pts = points[mask]

        count = class_counts.get(cls_name, 0) + 1
        class_counts[cls_name] = count
        fname = f"{cls_name}_{count}.txt"
        np.savetxt(ann_dir / fname, cls_pts, fmt="%.4f")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic S3DIS-style data for testing"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--num-rooms", type=int, default=5,
                        help="Rooms per area")
    parser.add_argument("--areas", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6],
                        help="Which area indices to create (default: all 6)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds_cfg = cfg["dataset"]
    s3dis_root = Path(ds_cfg["root"])

    print("=" * 60)
    print("  PropTech3D  –  Synthetic Data Generation")
    print("=" * 60)
    print(f"  Output      : {s3dis_root}")
    print(f"  Areas       : {args.areas}")
    print(f"  Rooms/area  : {args.num_rooms}")
    print()

    room_types = ["office", "conferenceRoom", "hallway", "storage", "lounge"]

    for area_idx in args.areas:
        area_dir = s3dis_root / f"Area_{area_idx}"
        area_dir.mkdir(parents=True, exist_ok=True)
        for i in range(args.num_rooms):
            rtype = room_types[i % len(room_types)]
            room_name = f"{rtype}_{i + 1}"
            room_dir = area_dir / room_name

            w = np.random.uniform(4.0, 8.0)
            d = np.random.uniform(3.5, 7.0)
            h = np.random.uniform(2.7, 3.5)
            points, labels = generate_room(w, d, h)
            write_s3dis_room(room_dir, points, labels)
            print(f"  Area_{area_idx}/{room_name}  "
                  f"({points.shape[0]:,} points, {w:.1f}x{d:.1f}x{h:.1f}m)")

    # Run preprocessing to create HDF5 blocks
    print()
    print("Preprocessing into HDF5 blocks ...")
    preprocess_and_cache(
        root=str(s3dis_root),
        out_dir=ds_cfg["processed"],
        num_points=ds_cfg["num_points"],
        block_size=ds_cfg["block_size"],
        stride=ds_cfg["stride"],
    )

    print()
    print("Done. You can now run:")
    print("  python -m scripts.train --config configs/default.yaml")


if __name__ == "__main__":
    main()
