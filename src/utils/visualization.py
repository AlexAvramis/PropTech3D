"""
Visualisation utilities for point clouds and floor plans.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.s3dis_dataset import CLASS_COLOURS, S3DIS_CLASSES, NUM_CLASSES


def plot_point_cloud_topdown(
    points: np.ndarray,
    labels: np.ndarray,
    save_path: str | Path | None = None,
    title: str = "Segmented Point Cloud (top-down)",
    figsize: tuple = (10, 10),
    point_size: float = 0.5,
) -> None:
    """Scatter plot of a point cloud coloured by semantic label (XY view)."""
    fig, ax = plt.subplots(figsize=figsize)
    colours = CLASS_COLOURS[labels] / 255.0

    ax.scatter(points[:, 0], points[:, 1], c=colours, s=point_size, edgecolors="none")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    # Legend
    handles = []
    for i, name in enumerate(S3DIS_CLASSES):
        if (labels == i).any():
            handles.append(
                plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=CLASS_COLOURS[i] / 255.0,
                           markersize=8, label=name)
            )
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_point_cloud_3d(
    points: np.ndarray,
    labels: np.ndarray,
    save_path: str | Path | None = None,
    title: str = "Segmented Point Cloud",
    figsize: tuple = (12, 8),
    point_size: float = 0.3,
    elev: float = 30,
    azim: float = -60,
) -> None:
    """3-D scatter plot coloured by semantic label."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    colours = CLASS_COLOURS[labels] / 255.0

    # Sub-sample for speed if very large
    n = len(points)
    if n > 200_000:
        idx = np.random.choice(n, 200_000, replace=False)
        points = points[idx]
        colours = colours[idx]

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=colours, s=point_size, depthshade=True)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_floorplan(
    floorplan_image: np.ndarray,
    save_path: str | Path | None = None,
    title: str = "Generated Floor Plan",
    figsize: tuple = (10, 10),
    dpi: int = 150,
) -> None:
    """Display / save a floor plan image."""
    fig, ax = plt.subplots(figsize=figsize)
    # OpenCV BGR → RGB
    if floorplan_image.ndim == 3 and floorplan_image.shape[2] == 3:
        img = floorplan_image[:, :, ::-1]
    else:
        img = floorplan_image
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_confusion_matrix(
    pred: np.ndarray,
    target: np.ndarray,
    save_path: str | Path | None = None,
    figsize: tuple = (10, 8),
) -> None:
    """Plot a normalised confusion matrix."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(target, pred, labels=list(range(NUM_CLASSES)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(S3DIS_CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(S3DIS_CLASSES, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalised Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
