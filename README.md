# PropTech3D – 3D Point Cloud to 2D Floor Plan Pipeline

End-to-end deep learning pipeline that takes a **raw 3D point cloud** of an indoor space, performs **semantic segmentation** (PointNet / PointNet++), and automatically generates a **2D architectural floor plan**.


---

## Architecture Overview

```
┌──────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  Raw Point   │ ───▶ │  PointNet /      │ ───▶ │  Floor Plan      │
│  Cloud       │      │  PointNet++ Seg. │      │  Generator       │
│  (.ply .txt) │      │  (13 classes)    │      │  (2D image)      │
└──────────────┘      └──────────────────┘      └──────────────────┘
     XYZ+RGB            per-point labels          walls / doors /
                                                  windows extracted
```

## Semantic Classes (S3DIS)

| ID | Class    | ID | Class    |
|----|----------|----|----------|
| 0  | ceiling  | 7  | table    |
| 1  | floor    | 8  | chair    |
| 2  | wall     | 9  | sofa     |
| 3  | beam     | 10 | bookcase |
| 4  | column   | 11 | board    |
| 5  | window   | 12 | clutter  |
| 6  | door     |    |          |

## Project Structure

```
PropTech3D/
├── app/
│   └── dashboard.py           # Streamlit visualizer (3D + floor plan)
├── configs/
│   └── default.yaml           # All hyper-parameters
├── scripts/
│   ├── preprocess.py          # S3DIS → HDF5 blocks
│   ├── train.py               # Train segmentation model
│   └── generate_floorplan.py  # End-to-end inference + floor plan
├── src/
│   ├── data/
│   │   └── s3dis_dataset.py   # Dataset loader & block sampler
│   ├── models/
│   │   └── pointnet.py        # PointNet & PointNet++ SSG
│   ├── pipeline/
│   │   ├── train.py           # Training loop & metrics
│   │   ├── segmentation.py    # Inference with vote aggregation
│   │   └── floorplan.py       # Wall detection → 2D plan
│   └── utils/
│       └── visualization.py   # Plotting helpers
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download & prepare S3DIS

Download the Stanford S3DIS dataset (aligned version) from the official source and place it at `data/S3DIS/` so the directory tree looks like:

```
data/S3DIS/
├── Area_1/
│   ├── conferenceRoom_1/
│   │   └── Annotations/
│   │       ├── ceiling_1.txt
│   │       ├── wall_1.txt
│   │       └── ...
│   └── ...
├── Area_2/
│   └── ...
└── Area_6/
```

Then preprocess into HDF5 blocks:

```bash
python -m scripts.preprocess --config configs/default.yaml
```

### 3. Train

```bash
python -m scripts.train --config configs/default.yaml
```

Quick smoke-test with a single epoch:

```bash
python -m scripts.train --config configs/default.yaml --epochs 1
```

Resume from a checkpoint:

```bash
python -m scripts.train --config configs/default.yaml --resume checkpoints/latest.pth
```

Training uses **Area 5** as the test set by default (standard S3DIS protocol). Key hyper-parameters:

| Parameter       | Default |
|-----------------|---------|
| Points / block  | 4096    |
| Block size      | 1.0 m   |
| Batch size      | 32      |
| Learning rate   | 0.001   |
| Epochs          | 100     |
| Scheduler       | Cosine  |

Best checkpoint → `checkpoints/best.pth`.

### 4. Generate a Floor Plan

```bash
python -m scripts.generate_floorplan \
    --room  data/S3DIS/Area_5/office_1 \
    --checkpoint checkpoints/best.pth \
    --output outputs/

# Also supports .ply, .txt (XYZRGB), or .npy files:
python -m scripts.generate_floorplan \
    --room my_scan.txt \
    --checkpoint checkpoints/best.pth

# Override model architecture or device:
python -m scripts.generate_floorplan \
    --room my_scan.npy \
    --checkpoint checkpoints/best.pth \
    --model pointnet2 --device cuda
```

**Outputs** (in `outputs/`):

| File | Content |
|------|---------|
| `<room>.png` | Clean 2D floor plan with walls, doors, windows |
| `<room>_segments.txt` | Wall / door / window line segments in metres |
| `<room>_segmented_topdown.png` | Colour-coded semantic segmentation (XY) |
| `<room>_segmented_3d.png` | Colour-coded 3D view |
| `<room>_floorplan_plot.png` | Matplotlib figure of the floor plan |

### 5. Visualizer Dashboard

Interactive Streamlit app with a split-screen view: 3D segmented room on the left, 2D floor plan on the right.

```bash
streamlit run app/dashboard.py
```

The dashboard lets you:

- Upload a point cloud file (`.txt`, `.npy`, or `.ply`)
- Select a trained checkpoint; model architecture is auto-detected from checkpoint metadata
- View an interactive, colour-coded 3D Plotly scatter of the segmented room
- View the generated 2D floor plan blueprint side-by-side
- Inspect per-class point distributions and summary metrics

A trained checkpoint must exist in `checkpoints/` before launching.

---

## Script Reference

### `scripts/preprocess.py`

Converts raw S3DIS room directories into per-area HDF5 block caches.

| Argument    | Required | Default                  | Description                 |
|-------------|----------|--------------------------|-----------------------------|
| `--config`  | No       | `configs/default.yaml`   | Path to YAML config file    |

### `scripts/train.py`

Trains the PointNet / PointNet++ segmentation model.

| Argument    | Required | Default                  | Description                              |
|-------------|----------|--------------------------|------------------------------------------|
| `--config`  | No       | `configs/default.yaml`   | Path to YAML config file                 |
| `--resume`  | No       | —                        | Path to checkpoint to resume training    |
| `--epochs`  | No       | from config (100)        | Override number of epochs (also updates cosine scheduler `T_max`) |

### `scripts/generate_floorplan.py`

Runs inference on a point cloud and generates the 2D floor plan.

| Argument       | Required | Default                  | Description                                    |
|----------------|----------|--------------------------|------------------------------------------------|
| `--room`       | **Yes**  | —                        | Path to room dir, `.ply`, `.txt`, or `.npy`    |
| `--checkpoint` | **Yes**  | —                        | Path to model checkpoint (`.pth`)              |
| `--config`     | No       | `configs/default.yaml`   | Path to YAML config file                       |
| `--output`     | No       | `outputs/`               | Output directory                               |
| `--model`      | No       | from config              | Override model name (`pointnet` / `pointnet2`) |
| `--device`     | No       | auto                     | Force device (`cuda` / `cpu`)                  |

### `app/dashboard.py`

Streamlit visualizer dashboard. No CLI arguments — all settings are controlled via the sidebar in the browser.

```bash
streamlit run app/dashboard.py
```

## Floor Plan Generation Pipeline

1. **Extract** wall, door, and window points from the segmented cloud.
2. **Project** wall points onto the XY plane (top-down view).
3. **Rasterise** into a binary occupancy image.
4. **Morphological cleanup** (dilate → erode → close) to fill gaps.
5. **Hough line detection** to extract wall line segments.
6. **Axis-alignment snapping** – near-horizontal/vertical lines are straightened.
7. **Segment merging** via Shapely to combine collinear and overlapping walls.
8. **Door/window detection** – DBSCAN clusters opening points, PCA extracts line segments.
9. **Render** final architectural drawing with walls (black), doors (blue arcs), windows (cyan), and a scale bar.

## Configuration

All parameters are in [`configs/default.yaml`](configs/default.yaml). Key sections:

- **dataset** – paths, block geometry, number of classes
- **model** – architecture (`pointnet` / `pointnet2`), dropout
- **train** – epochs, batch size, LR, scheduler
- **floorplan** – resolution (m/px), wall thickness, structural/opening class IDs, morphology kernel
- **paths** – checkpoint, output, and log directories

## Models

| Model | Description |
|-------|-------------|
| **PointNet** | Shared MLPs + T-Net transforms + global max-pool. Lightweight and fast. |
| **PointNet++ SSG** | Hierarchical set abstraction with farthest-point sampling and ball query. Better accuracy at higher compute cost. |

Switch models in the config:

```yaml
model:
  name: pointnet2   # or "pointnet"
```

## Supported Input Formats

| Format | Details |
|--------|---------|
| S3DIS directory | Room folder with `Annotations/*.txt` |
| `.ply` | Requires [open3d](https://pypi.org/project/open3d/) (Python <=3.12). Convert to `.txt` or `.npy` if unavailable. |
| `.txt` | Space-separated `X Y Z R G B` per line |
| `.npy` | NumPy array of shape `(N, 6)` |
