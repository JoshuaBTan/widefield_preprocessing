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

```bash
pip install -r requirements.txt
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate widefield
```

## Usage

There are two ways to run the pipeline: a **single-run** mode for processing one file at a time, and a **batch/discovery** mode for automatically preprocessing an entire BIDS-organized dataset. Batch mode is now the recommended way to run the pipeline for anything beyond a single test file, since it automatically handles run-1 vs. follow-up logic, split files, and per-session output organization.

### Single-run mode

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

Use `preprocess_calcium.py` for data with green + red + blue channels, or `preprocess_calciumonly.py` for green + blue only. These are always treated as a "first run" — they perform interactive atlas registration and brain-mask drawing from scratch.

### Batch / discovery mode

`discover_and_run.py` scans a BIDS-organized data folder, automatically groups files into logical runs (transparently reassembling split acquisitions), and runs the appropriate pipeline for each one — the first-run pipeline (with interactive registration) for each session's run-1, and the follow-up pipeline (`*_nf.py`, which reuses run-1's registration, brain mask, and reference frames) for every other run in that session.

Expected input folder structure:

```
data_root/
    sub-01/
        ses-1/
            func/
                sub-01_ses-1_task-rest_run-1_gb.tiff
                sub-01_ses-1_task-rest_run-1_gb_X1.tiff   # split part
                sub-01_ses-1_task-rest_run-1_gb_X2.tiff   # split part
                sub-01_ses-1_task-rest_run-2_gb.tiff
        ses-2/
            func/
                sub-01_ses-2_task-rest_run-1_grb.tiff
```

The channel suffix on the filename (`_gb`, `_grb`, etc.) determines whether the green+blue-only pipeline or the full green+red+blue pipeline is used for that file. Output mirrors the input `sub-XX/ses-XX/` structure under the chosen output root, with filenames automatically derived from each raw file's BIDS prefix (via `pipeline_utils.build_output_paths`) — there's no need to list individual output filenames in the config.

Run a full batch:

```bash
python discover_and_run.py --data /data --output /output --config config.yaml
```

Common filters for reprocessing a subset without touching the rest of the dataset:

```bash
# One subject
python discover_and_run.py --data /data --output /output --config config.yaml --subject sub-01

# One session
python discover_and_run.py --data /data --output /output --config config.yaml --subject sub-01 --session ses-2

# One run, standalone (its own registration, ignoring any existing run-1)
python discover_and_run.py --data /data --output /output --config config.yaml \
    --subject sub-01 --session ses-1 --run run-2

# One run, as a follow-up (reuse an already-preprocessed run's brain mask/atlas/references)
python discover_and_run.py --data /data --output /output --config config.yaml \
    --subject sub-01 --session ses-1 --run run-2 --ref-run run-1
```

Or from a script / Spyder:

```python
from discover_and_run import discover_and_run
discover_and_run(
    data_root   = "/data",
    output_root = "/output",
    config_file = "config.yaml",
    subject     = "sub-01",      # optional
    session     = "ses-1",       # optional
    run         = "run-2",       # optional
    ref_run     = "run-1",       # optional — reuse this run's references
    overrides_file = "overrides.yaml",  # optional
)
```

By default, run-1 for each session is the run with the lowest run number, and all other runs in that session reuse its brain mask, atlas registration, and reference frames — so only run-1 requires interactive input.

#### Overriding which run is treated as run-1

Sometimes the default run-1 (lowest run number) isn't the one you want to base a session's registration on — e.g. it was corrupted, aborted, or had poor registration compared to another run. `overrides.yaml` lets you specify a different file to use as run-1, per session:

```yaml
sub-01/ses-1: sub-01_ses-1_task-rest_run-2_gb.tiff
sub-02/ses-1: sub-02_ses-1_task-rest_run-3_gb.tiff
```

Each key is a `sub-XX/ses-XX` session label; each value is the base filename (not the full path, and without any `_Xn` split suffix) of the file to treat as run-1 for that session. Sessions not listed use the default behavior. Pass it with `--overrides overrides.yaml` on the command line, or `overrides_file="overrides.yaml"` when calling `discover_and_run()` directly.

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
preprocess_calcium.py          # First-run entry point (green+red+blue)
preprocess_calciumonly.py      # First-run entry point (green+blue only)
preprocess_calcium_nf.py       # Follow-up run entry point (green+red+blue)
preprocess_calciumonly_nf.py   # Follow-up run entry point (green+blue only)
discover_and_run.py            # Batch preprocessor: scans a BIDS dataset,
                                #   dispatches each run to the right pipeline
pipeline_utils.py              # Derives output file paths from BIDS filenames;
                                #   reconstructs a run's reference-file paths
config_template.yaml           # Template configuration file (first run)
config_nf.yaml                 # Template configuration file (follow-up run)
overrides.yaml                 # Optional per-session run-1 overrides for batch mode
requirements.txt
environment.yml
```

## Output Files

Output filenames are derived automatically from each raw file's BIDS prefix (e.g. `sub-04_ses-1_task-rest_run-1`) via `pipeline_utils.build_output_paths`, so you never need to list individual output filenames — only the output directory in `config.yaml` / `config_nf.yaml`. All outputs are saved as `.pkl` files:

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
| `brain_mask` | Binary brain mask (atlas-space for `mouse_to_atlas` mode, mouse-space for `atlas_to_mouse` mode) |
| `brain_mask_mouse` | Mouse-space brain mask (`mouse_to_atlas` mode only) — used for pre-warp masking and as the motion-correction reference in follow-up runs |
| `green_ref` / `blue_ref` / `red_ref` | Median reference frames per channel, used to motion-correct follow-up runs against run-1 |
| `transform` | Registration transform (`mouse_to_atlas` mode only) — reused by follow-up runs to warp data into the same atlas space as run-1 |

Follow-up runs (`preprocess_calcium_nf.py` / `preprocess_calciumonly_nf.py`) read the `green_ref`, `blue_ref`, `red_ref`, `brain_mask`, `brain_mask_mouse`, `atlas_mask`, and `transform` files saved by their session's run-1 instead of regenerating them, which is what lets every run in a session share identical registration without repeating the interactive steps.

