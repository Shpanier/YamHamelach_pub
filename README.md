# YamHamelach — Dead Sea Scroll Fragment Matching Pipeline

A computer-vision pipeline for detecting, matching, and ranking Dead Sea Scroll
fragment images. The system extracts patches from high-resolution scroll
photographs, computes SIFT feature matches between every pair of patches,
refines matches with homography error estimation, and exposes the results
through an interactive Streamlit viewer.

---

## Pipeline Overview

The pipeline consists of **six sequential stages** (`f01`–`f06`), each
implemented as a standalone script that reads from and writes to a shared
SQLite database under the active **database profile** directory.

```
f01  Patch Extraction
 │   (YOLO / Faster R-CNN)
 ▼
f02  SIFT Feature Matching
 │   (pairwise brute-force, ratio test, SQLite storage)
 ▼
f03  Homography Error Estimation
 │   (RANSAC homography, per-pair projection error stats)
 ▼
f04  Interactive Visualization   ← Streamlit app (local)
 │
f05  Database & Image Pruning
 │   (keep top-N matches, copy referenced images)
 ▼
f06  Deployment
     (copy pruned output to Streamlit Cloud folder)
```

`fragment_viewer.py` is the **production Streamlit viewer** used for both
local exploration and Streamlit Cloud deployment (with rotation, zoom, and
multi-profile support). `f04_plot_matching_pipeline.py` is a simpler local-only
viewer intended for quick analysis during development.

---

## File Descriptions

| File | Purpose |
|---|---|
| `f01_scrollPatchExtractor.py` | Detects and extracts patches from scroll images using YOLO or Faster R-CNN. Saves cropped patches, bounding-box visualizations, and JSON metadata. |
| `f02_sift_matching_pipeline.py` | Computes pairwise SIFT matches across all extracted patches. Stores results in an SQLite database (`matches.db`) with resume support and ground-truth (PAM) validation. |
| `f03_Homography_matching_pipeline.py` | Estimates homography between matched pairs and computes projection-error statistics. Supports Numba JIT, Apple Silicon MPS, and early pruning of poor matches. |
| `f04_plot_matching_pipeline.py` | Streamlit app for local interactive exploration of match results (table, scatter plots, error distributions). |
| `f05_purnning_db_pipeline.py` | Creates a pruned copy of the database and images, keeping only the top-N matches by homography error for lightweight deployment. |
| `f06_deploy_pipeline.py` | Copies the pruned database and images to the Streamlit Cloud deployment folder. Supports dry-run, `--execute`, and `--clean` modes. |
| `fragment_viewer.py` | Full-featured Streamlit viewer with database profile selection, rotation/zoom, and Streamlit Cloud support. This is the deployment entry point. |
| `db_profile.py` | Resolves database profiles from `.env` (e.g. `180`, `354`). Shared by all scripts. |
| `env_arguments_loader.py` | Loads `.env` variables into a namespace for `f01`. |
| `system_state_report.py` | Prints a diagnostic report of the current project state (file counts, DB stats, folder sizes). |
| `backup_and_cleanup.py` | Archives deprecated/old files into a timestamped ZIP. |
| `packages.txt` | System-level dependencies for Streamlit Cloud (OpenGL, X11 libs). |
| `requirements.txt` | Python package dependencies. |

---

## Requirements

- **Python 3.9+**
- A trained object-detection model (YOLO `.pt` or Faster R-CNN `.pth`)
- Input scroll images (`.jpg`) in the directory specified by the active profile

Install Python dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** `torch` / `torchvision` installation may vary by platform.
> See <https://pytorch.org/get-started/locally/> for CUDA/MPS wheels.

---

## Configuration (`.env`)

All scripts read their configuration from a `.env` file in the project root.
Create one based on the template below and adjust the paths:

```dotenv
# ── Base data directory ──────────────────────────────────
BASE_PATH = "/path/to/YamHamelach_data_n_model/"

# ── Model ────────────────────────────────────────────────
MODEL_NN_WEIGHTS = "/path/to/model/frcnn_r50fpn_epoch_2299.pth"
MODEL_TYPE = faster_rcnn          # or "yolo"
CONFIDENCE_THRESHOLD = 0.5

# ── Database Profiles ────────────────────────────────────
# Each profile isolates an image set with its own output directory.
# Scripts accept  --db <name>  to select a profile.
DEFAULT_DB_PROFILE = "180"

DB_PROFILE_180_IMAGES_IN   = "all_180_images/"
DB_PROFILE_180_OUTPUT_DIR  = "OUTPUT_faster_rcnn"
DB_PROFILE_180_LABEL       = "180 Images (Original)"

DB_PROFILE_354_IMAGES_IN   = "all_354_images/"
DB_PROFILE_354_OUTPUT_DIR  = "OUTPUT_faster_rcnn_354"
DB_PROFILE_354_LABEL       = "354 Images (Extended)"

# ── Pipeline settings ────────────────────────────────────
PATCHES_DIR        = output_patches
BBOX_DIR           = output_bbox
PATCHES_CACHE      = patches_key_dec_cache/
HOMOGRAPHY_CACHE   = homography_cache/
MIN_MATCH_COUNT    = 50
DEBUG              = True

# Early pruning (0 = disabled).
# f02 marks matches below MIN_MATCHES as pruned; f03 marks those above MAX_HOMO_ERROR.
EARLY_PRUNE_MIN_MATCHES    = 10
EARLY_PRUNE_MAX_HOMO_ERROR = 200

# ── Pruning & Deployment (f05 / f06) ────────────────────
PRUNED_DB           = matches_pruned.db
PRUNED_PATCHES_DIR  = output_patches_pruned
PRUNED_BBOX_DIR     = output_bbox_pruned
DEPLOY_PATH         = /path/to/streamlit-cloud/repo
```

---

## Running the Pipeline

All pipeline scripts support `--db <profile>` to override the default profile.

### Step 1 — Patch Extraction

```bash
python f01_scrollPatchExtractor.py [--db 180] [--model_type faster_rcnn]
```

Reads images from `BASE_PATH/<DB_PROFILE_*_IMAGES_IN>/`, runs object detection, and writes:
- `output_patches/<image_stem>/` — individual patch crops (`.jpg`)
- `output_bbox/<image_stem>.jpg` — annotated bounding-box image
- `*_patch_info.json` — metadata per image

### Step 2 — SIFT Feature Matching

```bash
python f02_sift_matching_pipeline.py [--db 180] [--stage complete]
```

| `--stage` | Description |
|---|---|
| `matching` | Run pairwise SIFT matching only |
| `validation` | Cross-reference with PAM ground truth |
| `info` | Print database statistics |
| `complete` | Run all stages sequentially (default) |

Results are stored in `matches.db` (SQLite). The run is **resumable** —
already-processed pairs are skipped automatically.

### Step 3 — Homography Error Estimation

```bash
python f03_Homography_matching_pipeline.py [--db 180] [--mode all]
```

| `--mode` | Description |
|---|---|
| `process` | Compute homography errors for pending pairs |
| `stats` | Print completion statistics |
| `all` | All of the above (default) |

Adds a `homography_errors` table to the same `matches.db`. Also resumable.

### Step 4 — Interactive Visualization (local)

```bash
streamlit run f04_plot_matching_pipeline.py
# or the full-featured viewer:
streamlit run fragment_viewer.py
```

Opens a browser-based UI to browse, filter, and visualize matches with
scatter plots, error distributions, and side-by-side fragment images.

### Step 5 — Pruning

```bash
python f05_purnning_db_pipeline.py [--db 180]
```

Creates a smaller `matches_pruned.db` keeping only the top matches, and copies
only the referenced patch and bounding-box images into `*_pruned` directories.

### Step 6 — Deployment

```bash
python f06_deploy_pipeline.py --status          # check current state
python f06_deploy_pipeline.py                    # dry run
python f06_deploy_pipeline.py --execute          # copy files
python f06_deploy_pipeline.py --execute --clean  # wipe target, then copy
```

Copies the pruned database and images to the Streamlit Cloud repo folder
defined by `DEPLOY_PATH`.

---

## Utility Scripts

| Script | Usage |
|---|---|
| `python system_state_report.py` | Print a full diagnostic of folders, databases, and image counts. |
| `python backup_and_cleanup.py [--execute]` | Archive deprecated files into a timestamped ZIP. |

---

## Directory Structure (at runtime)

```
BASE_PATH/
├── all_180_images/                  ← input scroll photos
├── OUTPUT_faster_rcnn/              ← profile "180" output
│   ├── output_patches/              ← extracted patches (f01)
│   ├── output_bbox/                 ← bbox visualizations (f01)
│   ├── patches_key_dec_cache/       ← SIFT descriptor cache (f02)
│   ├── homography_cache/            ← homography cache (f03)
│   ├── matches.db                   ← full results database (f02+f03)
│   ├── matches_pruned.db            ← pruned database (f05)
│   ├── output_patches_pruned/       ← pruned patches (f05)
│   └── output_bbox_pruned/          ← pruned bbox images (f05)
└── OUTPUT_faster_rcnn_354/          ← profile "354" output (same structure)
```

---

## License

*TBD*
