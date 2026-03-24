"""
End-to-end pipeline: point cloud → semantic segmentation → 2D floor plan.

Usage
-----
    python -m scripts.generate_floorplan \
        --room  data/S3DIS/Area_5/office_1 \
        --checkpoint checkpoints/best.pth \
        --config configs/default.yaml \
        --output outputs/

    # Or pass a .ply / .txt / .npy file directly:
    python -m scripts.generate_floorplan \
        --room  my_scan.ply \
        --checkpoint checkpoints/best.pth
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.segmentation import Segmentor
from src.pipeline.floorplan import FloorPlanGenerator, FloorPlanConfig
from src.utils.visualization import (
    plot_point_cloud_topdown,
    plot_point_cloud_3d,
    plot_floorplan,
)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a 2D floor plan from a 3D point cloud"
    )
    parser.add_argument("--room", required=True,
                        help="Path to room directory, .ply, .txt, or .npy")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="outputs/")
    parser.add_argument("--model", default=None,
                        help="Override model name (pointnet / pointnet2)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model or cfg["model"]["name"]
    room_name = Path(args.room).stem

    print("=" * 60)
    print("  PropTech3D  –  Floor Plan Generation")
    print("=" * 60)
    print(f"  Room       : {args.room}")
    print(f"  Model      : {model_name}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Output     : {output_dir}")
    print()

    # ── Step 1: Semantic segmentation ───────────
    print("[1/3]  Running semantic segmentation …")
    t0 = time.time()

    segmentor = Segmentor(
        checkpoint=args.checkpoint,
        model_name=model_name,
        input_channels=cfg["model"]["input_channels"],
        num_classes=cfg["dataset"]["num_classes"],
        device=args.device,
    )

    ds_cfg = cfg["dataset"]
    raw_points, labels = segmentor.predict_room(
        room_path=args.room,
        block_size=ds_cfg["block_size"],
        stride=ds_cfg["stride"],
        num_points=ds_cfg["num_points"],
    )
    print(f"     {raw_points.shape[0]:,} points segmented in {time.time()-t0:.1f}s")

    # ── Step 2: Generate floor plan ─────────────
    print("[2/3]  Generating 2D floor plan …")
    t1 = time.time()

    fp_cfg_dict = cfg.get("floorplan", {})
    fp_cfg = FloorPlanConfig(
        resolution=fp_cfg_dict.get("resolution", 0.02),
        wall_thickness=fp_cfg_dict.get("wall_thickness", 3),
        min_wall_length=fp_cfg_dict.get("min_wall_length", 0.3),
        classes_structural=fp_cfg_dict.get("classes_structural", [1, 2]),
        classes_openings=fp_cfg_dict.get("classes_openings", [5, 6]),
        morphology_kernel=fp_cfg_dict.get("morphology_kernel", 5),
        morph_dilate_iterations=fp_cfg_dict.get("morph_dilate_iterations", 1),
        morph_erode_iterations=fp_cfg_dict.get("morph_erode_iterations", 1),
        hough_threshold=fp_cfg_dict.get("hough_threshold", 30),
        hough_max_gap=fp_cfg_dict.get("hough_max_gap", 0.10),
        dbscan_eps=fp_cfg_dict.get("dbscan_eps", 0.15),
        dbscan_min_samples=fp_cfg_dict.get("dbscan_min_samples", 5),
        min_opening_length=fp_cfg_dict.get("min_opening_length", 0.3),
        max_opening_length=fp_cfg_dict.get("max_opening_length", 2.5),
        output_dpi=fp_cfg_dict.get("output_dpi", 150),
        snap_angle_tolerance=fp_cfg_dict.get("snap_angle_tolerance", 10.0),
        snap_distance=fp_cfg_dict.get("snap_distance", 0.15),
    )
    generator = FloorPlanGenerator(fp_cfg)
    floorplan = generator.generate(raw_points, labels)

    saved = generator.save(floorplan, output_dir, name=room_name)
    print(f"     Floor plan generated in {time.time()-t1:.1f}s")
    print(f"     Walls: {len(floorplan.walls)}  |  "
          f"Doors: {len(floorplan.doors)}  |  Windows: {len(floorplan.windows)}")

    # ── Step 3: Visualisations ──────────────────
    print("[3/3]  Saving visualisations …")

    plot_point_cloud_topdown(
        raw_points, labels,
        save_path=output_dir / f"{room_name}_segmented_topdown.png",
    )
    plot_point_cloud_3d(
        raw_points, labels,
        save_path=output_dir / f"{room_name}_segmented_3d.png",
    )
    plot_floorplan(
        floorplan.image,
        save_path=output_dir / f"{room_name}_floorplan_plot.png",
    )

    print()
    print("  Outputs saved to:")
    print(f"    Floor plan  : {saved['image']}")
    print(f"    Segments    : {saved['metadata']}")
    print(f"    Seg. top-down: {output_dir / f'{room_name}_segmented_topdown.png'}")
    print(f"    Seg. 3D     : {output_dir / f'{room_name}_segmented_3d.png'}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
