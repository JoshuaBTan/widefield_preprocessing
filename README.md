# Widefield Calcium Imaging Preprocessing Pipeline

A Python pipeline for preprocessing widefield calcium imaging data, including channel separation, motion correction, hemodynamic correction, atlas registration, and ROI extraction.

## Features

- Multi-channel separation from interleaved TIFF stacks (green / red / blue)
- Rigid motion correction
- ΔF/F normalization
- Hemodynamic artifact correction (regression-based)
- Temporal filtering (Butterworth bandpass for resting-state data)
- Atlas registration via landmark-based alignment (supports both `atlas_to_mouse` and `mouse_to_atlas` directions)
- Brain masking and ROI time series extraction using the Allen Atlas

## Requirements

- Python 3.8+
- See `requirements.txt` for package dependencies

Install dependencies:

In terminal/command prompt

```bash
pip install -r requirements.txt
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate widefield
```

## Usage

1. Copy `config_template.yaml` to `config.yaml` and fill in your paths and parameters:

```bash
cp config_template.yaml config.yaml
```

2. Edit `config.yaml` to point to your data, atlas, and desired output directory. Also choose parameter values appropriate for the data.

3. Run the pipeline:

```bash
python preprocess_calcium.py
```

Or import and call programmatically:

```python
from preprocess_calcium import run_pipeline
run_pipeline("config.yaml")
```

## Configuration

All pipeline parameters are controlled via `config.yaml`. Key sections:

| Section | Description |
|---|---|
| `data.filepath` | Path to raw `.tif` / `.tiff` stack |
| `experiment.type` | `"rest"` (applies bandpass filter) or `"task"` (no temporal filtering) |
| `downsampling.factor` | Spatial downsampling factor (e.g. `0.5` → 2048×2048 to 1024×1024) |
| `channel_separation` | Frames per cycle, channel order, and sampling rate |
| `normalization` | Baseline frames, method (`divide`), rotation, and filter cutoffs |
| `hemodynamic_correction` | Method (`regression`) and QC flag |
| `atlas` | Paths to Allen Atlas `.mat` files, registration method and direction |
| `brain_mask` | Method (`simple`) and QC flag |
| `roi_extraction` | Minimum ROI overlap threshold |
| `output` | Paths for all output `.pkl` files |

See `config_template.yaml` for full documentation of each parameter.

## Registration Modes

The pipeline supports two registration directions:

- **`atlas_to_mouse`** (default): The atlas is warped to match the native mouse FOV. Data stays in mouse space.
- **`mouse_to_atlas`**: The mouse data is transformed into standardized atlas space. Recommended for group-level analyses.

## Project Structure

```
widefield_pipeline/
├── __init__.py
├── calcium_io.py          # TIFF loading
├── isolate_calcium.py     # Hemodynamic artifact correction
├── normalization.py       # ΔF/F, detrending, filtering
├── preprocessing.py       # Downsampling, motion correction, masking
├── qc.py                  # Quality control plots
├── registration_new.py    # Atlas registration and transforms
├── roi_extraction.py      # ROI time series extraction
└── utils.py               # Shared utilities
preprocess_calcium.py      # Main pipeline entry point
config_template.yaml       # Template configuration file
requirements.txt
environment.yml
```

## Output Files

All outputs are saved as `.pkl` files at paths specified in `config.yaml`:

| Key | Contents |
|---|---|
| `pixel_ts` | Corrected calcium pixel time series |
| `roi_ts` | Calcium ROI-averaged time series |
| `green_pixel` | ΔF/F green channel (pixel-level) |
| `green_ts` | Green ROI-averaged time series |
| `hemo_ts` | Hemodynamic ROI-averaged time series |
| `hemopixel_ts` | Hemodynamic pixel time series |
| `roi_id` | Allen Brain Region (ROI) labels corresponding to data |
| `atlas_mask` | Atlas restricted to brain FOV |
| `brain_mask` | Binary brain mask |
| `green_ref` / `blue_ref` / `red_ref` | Median reference frames per channel |
| `transform` | Registration transform (mouse_to_atlas mode only) |

