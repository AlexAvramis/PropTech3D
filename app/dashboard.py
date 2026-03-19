"""
PropTech3D – Visualizer Dashboard
==================================

Streamlit app that lets the user upload a point cloud file, runs semantic
segmentation with a trained PointNet / PointNet++ model, and displays:
  - Left:  Interactive 3D color-coded segmented room  (Plotly)
  - Right: Generated 2D floor plan blueprint

Launch
------
    streamlit run app/dashboard.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
import yaml

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.s3dis_dataset import (
    S3DIS_CLASSES,
    CLASS_COLOURS,
    NUM_CLASSES,
)
from src.pipeline.floorplan import FloorPlan, FloorPlanConfig, FloorPlanGenerator
from src.pipeline.segmentation import Segmentor

# ──────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────

CONFIG_PATH = ROOT / "configs" / "default.yaml"

# CSS colours matching CLASS_COLOURS for Plotly
_CLASS_CSS = [
    f"rgb({r},{g},{b})" for r, g, b in CLASS_COLOURS.tolist()
]


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

@st.cache_data
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_segmentor(
    checkpoint: str,
    model_name: str,
    input_channels: int,
    num_classes: int,
) -> Segmentor:
    """Cache the model so it is only loaded once per session."""
    return Segmentor(
        checkpoint=checkpoint,
        model_name=model_name,
        input_channels=input_channels,
        num_classes=num_classes,
    )


def run_segmentation(
    segmentor: Segmentor,
    file_path: Path,
    block_size: float,
    stride: float,
    num_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on the uploaded file and return (points, labels)."""
    return segmentor.predict_room(
        room_path=file_path,
        block_size=block_size,
        stride=stride,
        num_points=num_points,
    )


def build_3d_figure(
    points: np.ndarray,
    labels: np.ndarray,
    max_display: int = 200_000,
) -> go.Figure:
    """Build an interactive Plotly 3D scatter of the segmented point cloud."""
    n = len(points)
    if n > max_display:
        idx = np.random.choice(n, max_display, replace=False)
        points = points[idx]
        labels = labels[idx]

    fig = go.Figure()

    for cls_id in range(NUM_CLASSES):
        mask = labels == cls_id
        if not mask.any():
            continue
        pts = points[mask]
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers",
            marker=dict(size=1.5, color=_CLASS_CSS[cls_id]),
            name=S3DIS_CLASSES[cls_id],
            hovertemplate=(
                f"<b>{S3DIS_CLASSES[cls_id]}</b><br>"
                "x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
        ),
        legend=dict(
            itemsizing="constant",
            font=dict(size=11),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=650,
    )
    return fig


def build_floorplan(
    points: np.ndarray,
    labels: np.ndarray,
    fp_cfg_dict: dict,
) -> tuple[np.ndarray, FloorPlan]:
    """Generate a 2D floor plan image (RGB) from segmented points."""
    fp_cfg = FloorPlanConfig(
        resolution=fp_cfg_dict.get("resolution", 0.02),
        wall_thickness=fp_cfg_dict.get("wall_thickness", 3),
        min_wall_length=fp_cfg_dict.get("min_wall_length", 0.3),
        classes_structural=fp_cfg_dict.get("classes_structural", [1, 2]),
        classes_openings=fp_cfg_dict.get("classes_openings", [5, 6]),
        wall_class_id=fp_cfg_dict.get("wall_class_id"),
        window_class_id=fp_cfg_dict.get("window_class_id"),
        door_class_id=fp_cfg_dict.get("door_class_id"),
        morphology_kernel=fp_cfg_dict.get("morphology_kernel", 5),
        morph_dilate_iterations=fp_cfg_dict.get("morph_dilate_iterations", 2),
        morph_erode_iterations=fp_cfg_dict.get("morph_erode_iterations", 1),
        hough_threshold=fp_cfg_dict.get("hough_threshold", 50),
        hough_max_gap=fp_cfg_dict.get("hough_max_gap", 0.15),
        dbscan_eps=fp_cfg_dict.get("dbscan_eps", 0.3),
        dbscan_min_samples=fp_cfg_dict.get("dbscan_min_samples", 5),
        canvas_margin=fp_cfg_dict.get("canvas_margin", 0.1),
        output_dpi=fp_cfg_dict.get("output_dpi", 150),
        scale_bar_length_m=fp_cfg_dict.get("scale_bar_length_m", 1.0),
        scale_bar_margin_px=fp_cfg_dict.get("scale_bar_margin_px", 20),
        scale_bar_font_scale=fp_cfg_dict.get("scale_bar_font_scale", 0.4),
        snap_angle_tolerance=fp_cfg_dict.get("snap_angle_tolerance", 5.0),
        snap_distance=fp_cfg_dict.get("snap_distance", 0.05),
    )
    generator = FloorPlanGenerator(fp_cfg)
    floorplan = generator.generate(points, labels)

    # BGR → RGB for Streamlit
    image_rgb = cv2.cvtColor(floorplan.image, cv2.COLOR_BGR2RGB)
    return image_rgb, floorplan


# ──────────────────────────────────────────────
#  Streamlit UI
# ──────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="PropTech3D – Visualizer",
        page_icon="house",
        layout="wide",
    )

    st.title("PropTech3D – Visualizer Dashboard")
    st.markdown(
        "Upload a point cloud file, run semantic segmentation, and view "
        "the **interactive 3D room** alongside the **2D floor plan**."
    )

    # ── Sidebar controls ────────────────────────
    with st.sidebar:
        st.header("Settings")

        cfg = load_config(str(CONFIG_PATH))
        ds_cfg = cfg["dataset"]
        model_cfg = cfg["model"]

        # Checkpoint selector
        ckpt_dir = ROOT / cfg["paths"]["checkpoints"]
        available_ckpts = sorted(ckpt_dir.glob("*.pth")) if ckpt_dir.exists() else []

        if not available_ckpts:
            st.warning(
                "No checkpoints found in `checkpoints/`. "
                "Train a model first with `python -m scripts.train`."
            )
            st.stop()

        ckpt_path = st.selectbox(
            "Checkpoint",
            available_ckpts,
            format_func=lambda p: p.name,
        )

        # Auto-detect model architecture from checkpoint when available.
        try:
            _ckpt_meta = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            model_name = _ckpt_meta.get("model_name", model_cfg["name"])
            if "model_name" in _ckpt_meta:
                st.info(f"Model: **{model_name}** (from checkpoint)")
            else:
                st.warning(
                    "This checkpoint does not store model metadata. "
                    f"Using config fallback: {model_name}."
                )
        except Exception as e:
            st.error(f"Failed to read checkpoint metadata: {e}")
            st.stop()

        st.divider()

        max_points = st.slider(
            "Max points to render (3D)",
            min_value=10_000,
            max_value=500_000,
            value=150_000,
            step=10_000,
            help="Subsample for smoother interaction. Does not affect segmentation.",
        )

    # ── File uploader ───────────────────────────
    uploaded = st.file_uploader(
        "Upload a point cloud",
        type=["txt", "npy", "ply"],
        help="Supported formats: .txt (XYZRGB per line), .npy (N×6), .ply",
    )

    if uploaded is None:
        st.info("Upload a point cloud file above to get started.")
        st.stop()

    # Save upload to a temp file so existing loaders can read it by path
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = Path(tmp.name)

    # ── Run pipeline ────────────────────────────
    with st.spinner("Loading model…"):
        try:
            segmentor = load_segmentor(
                checkpoint=str(ckpt_path),
                model_name=model_name,
                input_channels=model_cfg["input_channels"],
                num_classes=ds_cfg["num_classes"],
            )
        except Exception as e:
            st.error(
                "Model loading failed. This usually means the selected checkpoint "
                "does not match the expected architecture or classes. "
                f"Details: {e}"
            )
            st.stop()

    with st.spinner("Running semantic segmentation…"):
        raw_points, labels = run_segmentation(
            segmentor,
            tmp_path,
            block_size=ds_cfg["block_size"],
            stride=ds_cfg["stride"],
            num_points=ds_cfg["num_points"],
        )

    st.success(
        f"Segmented **{raw_points.shape[0]:,}** points into "
        f"**{len(np.unique(labels))}** classes."
    )

    # ── Split view ──────────────────────────────
    col_3d, col_fp = st.columns(2)

    with col_3d:
        st.subheader("3D Segmented Room")
        fig_3d = build_3d_figure(raw_points, labels, max_display=max_points)
        st.plotly_chart(fig_3d, use_container_width=True)

    with col_fp:
        st.subheader("2D Floor Plan")
        fp_cfg_dict = cfg.get("floorplan", {})

        with st.spinner("Generating floor plan…"):
            fp_image, floorplan = build_floorplan(raw_points, labels, fp_cfg_dict)

        st.image(fp_image, use_container_width=True)

        st.caption(
            f"Walls: {len(floorplan.walls)} · "
            f"Doors: {len(floorplan.doors)} · "
            f"Windows: {len(floorplan.windows)}"
        )

    # ── Summary metrics ─────────────────────────
    st.divider()
    metric_cols = st.columns(4)
    unique, counts = np.unique(labels, return_counts=True)
    metric_cols[0].metric("Total Points", f"{raw_points.shape[0]:,}")
    metric_cols[1].metric("Classes Found", str(len(unique)))
    metric_cols[2].metric("Wall Segments", str(len(floorplan.walls)))
    metric_cols[3].metric("Openings", str(len(floorplan.doors) + len(floorplan.windows)))

    # ── Per-class breakdown ─────────────────────
    with st.expander("Per-class point distribution"):
        class_data = {
            S3DIS_CLASSES[c]: int(counts[i])
            for i, c in enumerate(unique)
        }
        st.bar_chart(class_data)

    # Clean up temp file
    try:
        tmp_path.unlink()
    except OSError:
        pass


if __name__ == "__main__":
    main()
