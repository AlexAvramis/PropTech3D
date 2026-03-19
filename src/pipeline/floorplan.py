"""
Floor Plan Generator
====================

Converts a semantically-segmented 3D point cloud into a clean 2D floor plan.

Pipeline
--------
1. Extract structural classes (walls, floor) and openings (doors, windows).
2. Project wall points onto the XY-plane (top-down view).
3. Rasterise into a binary occupancy image.
4. Apply morphological operations to clean noise.
5. Detect wall line segments with the Hough transform.
6. Snap lines to axis-aligned grid.
7. Render doors / windows as gaps or special markers.
8. Output a publication-quality PNG + optional SVG.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString
from shapely.ops import unary_union


# ──────────────────────────────────────────────
#  Data classes
# ──────────────────────────────────────────────

@dataclass
class FloorPlanConfig:
    resolution: float = 0.02         # m/px
    wall_thickness: int = 3          # px
    min_wall_length: float = 0.3     # metres
    classes_structural: list[int] = field(default_factory=lambda: [1, 2])
    classes_openings: list[int] = field(default_factory=lambda: [5, 6])
    morphology_kernel: int = 5
    output_dpi: int = 150
    snap_angle_tolerance: float = 5.0   # degrees
    snap_distance: float = 0.05         # metres – merge nearby endpoints


@dataclass
class FloorPlan:
    image: np.ndarray                    # (H, W, 3) uint8 BGR
    walls: list[tuple]                   # list of ((x1,y1),(x2,y2)) in metres
    doors: list[tuple]
    windows: list[tuple]
    origin: np.ndarray                   # world XY of pixel (0,0)
    resolution: float


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

def _world_to_pixel(
    xy: np.ndarray, origin: np.ndarray, resolution: float,
) -> np.ndarray:
    """Convert world XY (N,2) → pixel coords (N,2) int."""
    px = ((xy - origin) / resolution).astype(np.int32)
    return px


def _pixel_to_world(
    px: np.ndarray, origin: np.ndarray, resolution: float,
) -> np.ndarray:
    return px.astype(np.float64) * resolution + origin


# ──────────────────────────────────────────────
#  Core generator
# ──────────────────────────────────────────────

class FloorPlanGenerator:
    """Generate a 2D floor plan from segmented 3D points."""

    def __init__(self, cfg: FloorPlanConfig | None = None):
        self.cfg = cfg or FloorPlanConfig()

    def generate(
        self,
        points: np.ndarray,
        labels: np.ndarray,
    ) -> FloorPlan:
        """
        Parameters
        ----------
        points : (N, >=3) XYZ (+ optional RGB)
        labels : (N,) class indices

        Returns
        -------
        FloorPlan with image, wall / door / window segments, and geo-referencing.
        """
        cfg = self.cfg
        xyz = points[:, :3].astype(np.float64)

        # 1. Extract structural points (walls + floor for outline)
        struct_mask = np.isin(labels, cfg.classes_structural)
        # Use the last structural class as wall (convention: [floor, wall])
        wall_class = cfg.classes_structural[-1] if cfg.classes_structural else 2
        wall_mask = (labels == wall_class)
        # classes_openings convention: [window, door]
        window_class = cfg.classes_openings[0] if len(cfg.classes_openings) > 0 else None
        door_class = cfg.classes_openings[1] if len(cfg.classes_openings) > 1 else None
        door_mask = (labels == door_class) if door_class is not None else np.zeros(len(labels), dtype=bool)
        window_mask = (labels == window_class) if window_class is not None else np.zeros(len(labels), dtype=bool)

        # 2. Project to XY (top-down)
        wall_xy = xyz[wall_mask, :2]
        door_xy = xyz[door_mask, :2] if door_mask.any() else np.empty((0, 2))
        window_xy = xyz[window_mask, :2] if window_mask.any() else np.empty((0, 2))

        if wall_xy.shape[0] < 10:
            # Fallback: use all structural points if no wall-specific class
            wall_xy = xyz[struct_mask, :2]

        # 3. Set up canvas
        all_xy = xyz[:, :2]
        origin = all_xy.min(axis=0) - 0.1  # small margin
        extent = all_xy.max(axis=0) + 0.1
        canvas_size = ((extent - origin) / cfg.resolution).astype(int) + 1
        H, W = int(canvas_size[1]), int(canvas_size[0])

        # 4. Rasterise wall points
        wall_img = np.zeros((H, W), dtype=np.uint8)
        if wall_xy.shape[0] > 0:
            px = _world_to_pixel(wall_xy, origin, cfg.resolution)
            valid = (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
            px = px[valid]
            wall_img[px[:, 1], px[:, 0]] = 255

        # 5. Morphological clean-up
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (cfg.morphology_kernel, cfg.morphology_kernel),
        )
        wall_img = cv2.dilate(wall_img, kernel, iterations=2)
        wall_img = cv2.erode(wall_img, kernel, iterations=1)
        wall_img = cv2.morphologyEx(wall_img, cv2.MORPH_CLOSE, kernel)

        # 6. Detect line segments (Probabilistic Hough)
        min_line_px = int(cfg.min_wall_length / cfg.resolution)
        lines_raw = cv2.HoughLinesP(
            wall_img,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=min_line_px,
            maxLineGap=int(0.15 / cfg.resolution),
        )

        wall_segments = []
        if lines_raw is not None:
            for line in lines_raw:
                x1, y1, x2, y2 = line[0]
                p1 = _pixel_to_world(np.array([x1, y1]), origin, cfg.resolution)
                p2 = _pixel_to_world(np.array([x2, y2]), origin, cfg.resolution)
                wall_segments.append((tuple(p1), tuple(p2)))

        # 7. Snap to axis-aligned directions
        wall_segments = self._snap_segments(wall_segments)

        # 8. Merge collinear / overlapping segments
        wall_segments = self._merge_segments(wall_segments)

        # 9. Detect door / window positions (projected centroids)
        door_segments = self._opening_segments(door_xy) if door_xy.shape[0] > 0 else []
        window_segments = self._opening_segments(window_xy) if window_xy.shape[0] > 0 else []

        # 10. Render final image
        image = self._render(
            H, W, origin, wall_segments, door_segments, window_segments,
        )

        return FloorPlan(
            image=image,
            walls=wall_segments,
            doors=door_segments,
            windows=window_segments,
            origin=origin,
            resolution=cfg.resolution,
        )

    # ─── segment processing ──────────────────

    def _snap_segments(
        self, segments: list[tuple],
    ) -> list[tuple]:
        """Snap near-axis-aligned segments to perfect 0° / 90°."""
        tol = self.cfg.snap_angle_tolerance
        snapped = []
        for (x1, y1), (x2, y2) in segments:
            dx, dy = x2 - x1, y2 - y1
            angle = np.degrees(np.arctan2(abs(dy), abs(dx)))
            if angle < tol:
                # Horizontal – average Y
                ym = (y1 + y2) / 2
                snapped.append(((x1, ym), (x2, ym)))
            elif abs(angle - 90) < tol:
                # Vertical – average X
                xm = (x1 + x2) / 2
                snapped.append(((xm, y1), (xm, y2)))
            else:
                snapped.append(((x1, y1), (x2, y2)))
        return snapped

    def _merge_segments(
        self, segments: list[tuple],
    ) -> list[tuple]:
        """Merge collinear and overlapping wall segments with Shapely."""
        if not segments:
            return []
        lines = [LineString([p1, p2]) for p1, p2 in segments]
        # Buffer slightly to merge nearby collinear lines
        buffered = unary_union(
            [l.buffer(self.cfg.snap_distance) for l in lines]
        )
        # Extract centrelines back
        merged = []
        if buffered.is_empty:
            return segments

        # Use skeleton-like extraction: get boundary and simplify
        try:
            boundary = buffered.boundary
            if boundary.is_empty:
                return segments
            if hasattr(boundary, "geoms"):
                parts = list(boundary.geoms)
            else:
                parts = [boundary]
            for part in parts:
                coords = list(part.coords)
                simplified = LineString(coords).simplify(self.cfg.snap_distance * 2)
                sc = list(simplified.coords)
                for i in range(len(sc) - 1):
                    seg_len = np.hypot(sc[i+1][0]-sc[i][0], sc[i+1][1]-sc[i][1])
                    if seg_len >= self.cfg.min_wall_length:
                        merged.append((sc[i], sc[i+1]))
        except Exception:
            return segments

        return merged if merged else segments

    def _opening_segments(self, xy: np.ndarray) -> list[tuple]:
        """Cluster opening points into segments via PCA per cluster."""
        from sklearn.cluster import DBSCAN

        if xy.shape[0] < 3:
            return []

        clustering = DBSCAN(eps=0.3, min_samples=5).fit(xy)
        segments = []
        for cid in set(clustering.labels_):
            if cid == -1:
                continue
            cluster = xy[clustering.labels_ == cid]
            # PCA main axis → line segment
            mean = cluster.mean(axis=0)
            centered = cluster - mean
            cov = np.cov(centered.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            main_axis = eigvecs[:, eigvals.argmax()]
            proj = centered @ main_axis
            p1 = mean + main_axis * proj.min()
            p2 = mean + main_axis * proj.max()
            segments.append((tuple(p1), tuple(p2)))

        return segments

    # ─── rendering ───────────────────────────

    def _render(
        self,
        H: int, W: int,
        origin: np.ndarray,
        walls: list[tuple],
        doors: list[tuple],
        windows: list[tuple],
    ) -> np.ndarray:
        """Render a clean architectural floor plan image."""
        # White background
        img = np.full((H, W, 3), 255, dtype=np.uint8)
        cfg = self.cfg

        # Draw walls (black)
        for (x1, y1), (x2, y2) in walls:
            p1 = _world_to_pixel(np.array([x1, y1]), origin, cfg.resolution)
            p2 = _world_to_pixel(np.array([x2, y2]), origin, cfg.resolution)
            cv2.line(img, (int(p1[0]), int(p1[1])),
                     (int(p2[0]), int(p2[1])),
                     color=(0, 0, 0), thickness=cfg.wall_thickness)

        # Draw doors (blue arcs)
        for (x1, y1), (x2, y2) in doors:
            p1 = _world_to_pixel(np.array([x1, y1]), origin, cfg.resolution)
            p2 = _world_to_pixel(np.array([x2, y2]), origin, cfg.resolution)
            mid = ((p1 + p2) / 2).astype(int)
            radius = int(np.linalg.norm(p2 - p1) / 2)
            angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
            cv2.ellipse(img, tuple(mid), (radius, radius),
                        angle, 0, 90, color=(200, 100, 0), thickness=2)
            # White gap over wall to indicate opening
            cv2.line(img, tuple(p1), tuple(p2),
                     color=(255, 255, 255), thickness=cfg.wall_thickness + 2)

        # Draw windows (cyan double-line)
        for (x1, y1), (x2, y2) in windows:
            p1 = _world_to_pixel(np.array([x1, y1]), origin, cfg.resolution)
            p2 = _world_to_pixel(np.array([x2, y2]), origin, cfg.resolution)
            # White gap
            cv2.line(img, tuple(p1), tuple(p2),
                     color=(255, 255, 255), thickness=cfg.wall_thickness + 2)
            # Cyan lines
            cv2.line(img, tuple(p1), tuple(p2),
                     color=(200, 200, 0), thickness=1)
            dx = int(2 * (p2[1] - p1[1]) / max(np.linalg.norm(p2 - p1), 1))
            dy = int(-2 * (p2[0] - p1[0]) / max(np.linalg.norm(p2 - p1), 1))
            cv2.line(img, (p1[0]+dx, p1[1]+dy), (p2[0]+dx, p2[1]+dy),
                     color=(200, 200, 0), thickness=1)

        # Draw scale bar (bottom-right)
        bar_length_m = 1.0
        bar_length_px = int(bar_length_m / cfg.resolution)
        margin = 20
        if W > bar_length_px + 2 * margin and H > 40:
            bx2 = W - margin
            bx1 = bx2 - bar_length_px
            by = H - margin
            cv2.line(img, (bx1, by), (bx2, by), (0, 0, 0), 2)
            cv2.line(img, (bx1, by - 5), (bx1, by + 5), (0, 0, 0), 2)
            cv2.line(img, (bx2, by - 5), (bx2, by + 5), (0, 0, 0), 2)
            cv2.putText(img, f"{bar_length_m:.0f} m",
                        (bx1, by - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 0, 0), 1, cv2.LINE_AA)

        return img

    # ─── I/O ─────────────────────────────────

    def save(
        self, floorplan: FloorPlan, output_dir: str | Path, name: str = "floorplan",
    ) -> dict[str, Path]:
        """Save floor plan image and metadata."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        img_path = output_dir / f"{name}.png"
        cv2.imwrite(str(img_path), floorplan.image)

        # Save wall / opening data as plain text
        meta_path = output_dir / f"{name}_segments.txt"
        with open(meta_path, "w") as f:
            f.write("# Wall segments (x1 y1 x2 y2) in metres\n")
            for (x1, y1), (x2, y2) in floorplan.walls:
                f.write(f"WALL {x1:.4f} {y1:.4f} {x2:.4f} {y2:.4f}\n")
            for (x1, y1), (x2, y2) in floorplan.doors:
                f.write(f"DOOR {x1:.4f} {y1:.4f} {x2:.4f} {y2:.4f}\n")
            for (x1, y1), (x2, y2) in floorplan.windows:
                f.write(f"WINDOW {x1:.4f} {y1:.4f} {x2:.4f} {y2:.4f}\n")

        return {"image": img_path, "metadata": meta_path}
