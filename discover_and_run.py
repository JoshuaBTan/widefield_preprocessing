    # -*- coding: utf-8 -*-
"""
discover_and_run.py  —  Automatic batch preprocessor for widefield calcium imaging.

Scans a BIDS-organised data folder, groups files into logical runs (handling
split files transparently), determines which pipeline to use per session, and
runs preprocessing sequentially — first-run pipeline for the designated run-1,
follow-up pipeline for all other runs within the same session.

Folder structure expected
-------------------------
data_root/
    sub-01/
        ses-1/
            func/
                sub-01_ses-1_task-rest_run-1_gb.tiff
                sub-01_ses-1_task-rest_run-1_gb_X1.tiff   <- split part
                sub-01_ses-1_task-rest_run-1_gb_X2.tiff   <- split part
                sub-01_ses-1_task-rest_run-2_gb.tiff
        ses-2/
            func/
                sub-01_ses-2_task-rest_run-1_grb.tiff
    sub-02/
        ses-1/
            func/
                sub-02_ses-1_task-rest_run-1_gb.tiff

Output mirrors input structure under output_root:
output_root/
    sub-01/
        ses-1/
            sub-01_ses-1_task-rest_run-1_pixel.pkl
            sub-01_ses-1_task-rest_run-1_roi.pkl
            ...
        ses-2/
            sub-01_ses-2_task-rest_run-1_pixel.pkl
            ...
    sub-02/
        ses-1/
            ...

Key behaviours
--------------
- Each session is processed independently. The brain mask, atlas registration,
  and reference frames from a session's run-1 are reused for all subsequent
  runs in that same session only.
- Split files (_X1, _X2, …) are loaded and downsampled independently, then
  concatenated in memory before processing (safe at 8× downsampling).
- Run-1 is normally the file with the lowest run number, but can be overridden
  per session via an overrides YAML (--overrides).
- Filtering by --subject, --session, or --run lets you reprocess a single run
  without touching the rest of the dataset.
- Filtering by --task lets you restrict a batch to runs of one task label
  (e.g. "rest" vs "whisker"), which is useful when different tasks need
  different config.yaml settings (e.g. experiment.type: "rest" applies a
  bandpass filter, "task" does not). Run discover_and_run.py once per task,
  each with its own config file. Within a session, "run-1" (the run used for
  atlas registration and brain masking) is chosen from among the
  matching-task runs only.

Usage
-----
From the command line:
    # Full batch
    python discover_and_run.py --data /data --output /output --config config.yaml

    # Single subject
    python discover_and_run.py --data /data --output /output --config config.yaml \\
        --subject sub-01

    # Single session
    python discover_and_run.py --data /data --output /output --config config.yaml \\
        --subject sub-01 --session ses-2

    # Single run — standalone (own registration, ignores any existing run-1)
    python discover_and_run.py --data /data --output /output --config config.yaml \\
        --subject sub-01 --session ses-1 --run run-2

    # Single run — follow-up (reuse an already-preprocessed run's references)
    python discover_and_run.py --data /data --output /output --config config.yaml \\
        --subject sub-01 --session ses-1 --run run-2 --ref-run run-1

    # Custom run-1 overrides
    python discover_and_run.py --data /data --output /output --config config.yaml \\
        --overrides overrides.yaml

    # Only one task across the whole dataset — e.g. rest data needs its own
    # config (experiment.type: "rest") separate from a whisker-stim config
    python discover_and_run.py --data /data --output /output \\
        --config config_rest.yaml --task rest

    python discover_and_run.py --data /data --output /output \\
        --config config_whisker.yaml --task whisker

    # Task filter combined with subject/session/run
    python discover_and_run.py --data /data --output /output --config config.yaml \\
        --subject sub-01 --session ses-1 --task rest --run run-2

From Spyder / a script:
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

--run / --ref-run behaviour
---------------------------
--run only        : targeted run is standalone — gets its own interactive atlas
                    registration and brain mask. Use when settings differ.

--run + --ref-run : targeted run is a follow-up — skips registration and reuses
                    brain mask, atlas, and reference frames from ref-run's
                    already-saved outputs. Use when run-1 was preprocessed
                    earlier and you now want to add a new run without
                    reprocessing run-1.

Overrides YAML format
---------------------
Map each session (as "sub-XX/ses-XX") to the filename (not full path) of the
file that should be treated as run-1 for that session:

    sub-01/ses-1: sub-01_ses-1_task-rest_run-2_gb.tiff
    sub-02/ses-1: sub-02_ses-1_task-rest_run-3_gb.tiff

Only sessions listed here are affected; all others use the default (lowest
run number).
"""

import argparse
import copy
import re
import traceback
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Channel suffix immediately before the extension
# e.g. _gb  _grb  _GRB  in  sub-01_ses-1_task-rest_run-1_gb.tiff
_CHANNEL_RE = re.compile(
    r"_(g(?:r(?:b)?)?|r(?:b)?|b|gb|grb|gr|rb|green|blue|red|GRB|GB|RB)$",
    re.IGNORECASE,
)

# Split-part suffix: _X1, _X2, _X10 …
_SPLIT_RE = re.compile(r"_X(\d+)$", re.IGNORECASE)

# Channels that include a red channel
_HAS_RED = {"grb", "gr", "rb", "red"}


# ---------------------------------------------------------------------------
# File discovery helpers
# ---------------------------------------------------------------------------

def _iter_func_dirs(data_root: Path):
    """Yield every func/ directory at depth sub-*/ses-*/func/ that has TIFFs."""
    for sub_dir in sorted(data_root.iterdir()):
        if not sub_dir.is_dir() or not sub_dir.name.startswith("sub-"):
            continue
        for ses_dir in sorted(sub_dir.iterdir()):
            if not ses_dir.is_dir() or not ses_dir.name.startswith("ses-"):
                continue
            func_dir = ses_dir / "func"
            if func_dir.is_dir():
                tiffs = (list(func_dir.glob("*.tif")) +
                         list(func_dir.glob("*.tiff")))
                if tiffs:
                    yield func_dir


def _stem_without_split(path: Path) -> str:
    """Strip _Xn suffix from stem. e.g. run-1_gb_X2 -> run-1_gb"""
    return _SPLIT_RE.sub("", path.stem)


def _channel_suffix(stem: str):
    """Extract channel suffix from stem, or None if not recognised."""
    m = _CHANNEL_RE.search(stem)
    return m.group(1).lower() if m else None


def _bids_prefix(stem: str) -> str:
    """Strip channel suffix from stem. e.g. run-1_gb -> run-1"""
    return _CHANNEL_RE.sub("", stem)


def _run_number(prefix: str) -> int:
    """Extract run number from BIDS prefix for sorting. Defaults to 0."""
    m = re.search(r"_run-(\d+)", prefix)
    return int(m.group(1)) if m else 0


def _task_name(prefix: str):
    """
    Extract the task label from a BIDS run prefix, e.g.
    'sub-01_ses-1_task-rest_run-1' -> 'rest'.
    Returns None if no task-XXX segment is found.
    """
    m = re.search(r"_task-([A-Za-z0-9]+)_run", prefix)
    return m.group(1).lower() if m else None


def _split_number(path: Path) -> int:
    """Return split part number: base file -> 0, _X1 -> 1, _X2 -> 2, …"""
    m = _SPLIT_RE.search(path.stem)
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# Session-level grouping
# ---------------------------------------------------------------------------

def _group_session(func_dir: Path) -> list:
    """
    Return an ordered list of logical run descriptors for a func/ directory.

    Each descriptor dict contains:
      prefix   : BIDS run prefix, e.g. "sub-01_ses-1_task-rest_run-1"
      channel  : channel suffix string, e.g. "gb"
      has_red  : bool
      base     : Path to the base file (no _Xn suffix)
      parts    : [base, X1, X2, …] sorted numerically
      is_split : bool
    """
    tiffs = sorted(
        [p for p in func_dir.iterdir()
         if p.suffix.lower() in {".tif", ".tiff"}],
        key=lambda p: p.name,
    )

    # Group by base stem (stem without _Xn)
    groups: dict = {}
    for t in tiffs:
        base_stem = _stem_without_split(t)
        groups.setdefault(base_stem, []).append(t)

    runs = []
    for base_stem, files in groups.items():
        channel = _channel_suffix(base_stem)
        if channel is None:
            print(f"  WARNING: unrecognised channel suffix in '{base_stem}' — skipping.")
            continue

        prefix       = _bids_prefix(base_stem)
        files_sorted = sorted(files, key=_split_number)

        runs.append({
            "prefix":   prefix,
            "channel":  channel,
            "has_red":  channel in _HAS_RED,
            "base":     files_sorted[0],   # base file (split number 0)
            "parts":    files_sorted,
            "is_split": len(files_sorted) > 1,
        })

    runs.sort(key=lambda r: _run_number(r["prefix"]))
    return runs


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_and_downsample(filepath: Path, scale: float) -> np.ndarray:
    """Load one TIFF and downsample it immediately to free raw memory."""
    from widefield_pipeline.calcium_io import load_tiff_stack
    from widefield_pipeline.preprocessing import downsample_stack

    print(f"    Loading {filepath.name} …")
    stack = load_tiff_stack(str(filepath))
    print(f"      Raw shape : {stack.shape}  →  downsampling ×{scale}")
    ds = downsample_stack(stack, scale=scale)
    print(f"      DS shape  : {ds.shape}")
    del stack
    return ds


def _load_run(run_desc: dict, scale: float) -> np.ndarray:
    """
    Load and downsample all parts of a logical run, concatenating along
    the time axis.  For non-split runs this is a single file load.
    """
    parts_data = [_load_and_downsample(p, scale) for p in run_desc["parts"]]

    if len(parts_data) == 1:
        return parts_data[0]

    print(f"    Concatenating {len(parts_data)} parts along time axis …")
    concat = np.concatenate(parts_data, axis=0)
    print(f"      Concatenated shape: {concat.shape}")
    del parts_data
    return concat


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_temp_config(cfg: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    return path


def _make_config(shared: dict,
                 filepath: Path,
                 out_dir: Path,
                 run1_filepath: Path = None,
                 run1_out_dir: Path = None) -> dict:
    """Build a per-run config from shared config, injecting run-specific paths."""
    cfg = copy.deepcopy(shared)
    cfg.setdefault("data",   {})["filepath"] = str(filepath)
    cfg.setdefault("output", {})["dir"]      = str(out_dir)

    if run1_filepath is not None:
        cfg.setdefault("reference", {})["run1_filepath"] = str(run1_filepath)
    if run1_out_dir is not None:
        cfg.setdefault("reference", {})["run1_out_dir"]  = str(run1_out_dir)

    return cfg


# ---------------------------------------------------------------------------
# Pipeline dispatch
# ---------------------------------------------------------------------------

def _get_pipeline(has_red: bool, is_first_run: bool):
    """Return the appropriate run_pipeline function."""
    if is_first_run:
        if has_red:
            from preprocess_calcium import run_pipeline
        else:
            from preprocess_calciumonly import run_pipeline
    else:
        if has_red:
            from preprocess_calcium_nf import run_pipeline
        else:
            from preprocess_calciumonly_nf import run_pipeline
    return run_pipeline


# ---------------------------------------------------------------------------
# Session processor
# ---------------------------------------------------------------------------

def _process_session(func_dir: Path,
                     output_root: Path,
                     shared_config: dict,
                     tmp_root: Path,
                     run1_override: str = None,
                     run_filter: str = None,
                     ref_run: str = None,
                     task_filter: str = None) -> tuple:
    """
    Process all logical runs in one session func/ directory.

    Parameters
    ----------
    func_dir : Path
        The session's func/ directory.
    output_root : Path
        Root output folder (session outputs go to output_root/sub-XX/ses-XX/).
    shared_config : dict
        Shared processing parameters from config.yaml.
    tmp_root : Path
        Directory for temporary per-run config files.
    run1_override : str or None
        Filename (not full path) of the file to treat as run-1 for this session.
        If None, the run with the lowest run number is used.
    run_filter : str or None
        If set (e.g. "run-2"), process only that run.  When ref_run is also
        set the targeted run is treated as a follow-up; otherwise standalone.
    ref_run : str or None
        If set alongside run_filter (e.g. "run-1"), the targeted run is
        processed as a follow-up using the already-saved outputs of ref_run
        as the reference (brain mask, atlas, reference frames).  If None and
        run_filter is set, the targeted run is treated as standalone.
    task_filter : str or None
        If set (e.g. "rest" or "whisker"), only runs whose filename contains
        "task-<task_filter>" are processed for this session; all other runs
        are ignored as if they didn't exist. This is applied before run1
        selection, so "run-1" (lowest run number) is chosen from within the
        matching task only. Useful when different tasks need different
        preprocessing config (e.g. experiment.type: "rest" vs "task"), since
        a single discover_and_run.py invocation uses one shared config.

    Returns
    -------
    (n_ok, n_fail) : tuple of int
    """
    # Mirror: data_root/sub-01/ses-1/func -> output_root/sub-01/ses-1
    rel     = func_dir.relative_to(func_dir.parent.parent.parent)
    out_dir = output_root / rel.parent

    sub_label = func_dir.parent.parent.name
    ses_label = func_dir.parent.name
    print(f"\n{'='*70}")
    print(f"Session : {sub_label}/{ses_label}")
    print(f"Input   : {func_dir}")
    print(f"Output  : {out_dir}")
    print(f"{'='*70}")

    all_runs = _group_session(func_dir)
    if not all_runs:
        print("  No recognised TIFF files found — skipping.")
        return 0, 0

    # ------------------------------------------------------------------
    # Apply task_filter: restrict to runs matching the requested task
    # (e.g. "rest" vs "whisker") before any run-1 selection happens, so
    # run-1 is chosen from within the matching task only.
    # ------------------------------------------------------------------
    if task_filter is not None:
        task_filter_lc = task_filter.lower()
        all_runs = [r for r in all_runs if _task_name(r["prefix"]) == task_filter_lc]
        if not all_runs:
            print(f"  No runs found matching task '{task_filter}' — skipping.")
            return 0, 0

    # ------------------------------------------------------------------
    # Apply run_filter: process a single run, standalone or follow-up
    # ------------------------------------------------------------------
    if run_filter is not None:
        target = [r for r in all_runs
                  if re.search(rf"_{re.escape(run_filter)}(?:_|$)", r["prefix"])]
        if not target:
            print(f"  WARNING: run '{run_filter}' not found in this session — skipping.")
            return 0, 0

        if ref_run is not None:
            # Follow-up mode: find the reference run in the full run list
            ref_matches = [r for r in all_runs
                           if re.search(rf"_{re.escape(ref_run)}(?:_|$)", r["prefix"])]
            if not ref_matches:
                print(f"  WARNING: ref-run '{ref_run}' not found in this session — "
                      f"falling back to standalone mode.")
                runs       = target
                run1_index = 0
                run1_desc  = runs[0]
                is_followup_single = False
            else:
                runs              = target
                run1_index        = 0
                run1_desc         = ref_matches[0]
                is_followup_single = True
                print(f"  Follow-up mode: '{run_filter}' will reuse references "
                      f"from '{ref_run}'.")
        else:
            # Standalone mode: targeted run gets its own registration
            runs              = target
            run1_index        = 0
            run1_desc         = runs[0]
            is_followup_single = False
            print(f"  Single-run mode: processing '{run_filter}' as standalone.")
    else:
        is_followup_single = False   # batch mode uses index-based logic below
        runs = all_runs

        # ------------------------------------------------------------------
        # Identify run-1 (lowest run number, or overridden)
        # ------------------------------------------------------------------
        if run1_override is not None:
            # Find the run whose base filename matches the override
            override_match = [
                i for i, r in enumerate(runs)
                if r["base"].name == run1_override
            ]
            if not override_match:
                print(f"  WARNING: run-1 override '{run1_override}' not found — "
                      f"falling back to default (lowest run number).")
                run1_index = 0
            else:
                run1_index = override_match[0]
                print(f"  run-1 override: using '{run1_override}' as run-1.")
        else:
            run1_index = 0   # already sorted by run number; lowest is first

    # Log what we found
    print(f"  Found {len(runs)} logical run(s) to process:")
    for i, r in enumerate(runs):
        parts_str  = f" + {len(r['parts'])-1} split part(s)" if r["is_split"] else ""
        role       = "[run-1]" if (run_filter is None and i == run1_index) else ""
        standalone = "[standalone]" if run_filter is not None else ""
        print(f"    {r['prefix']}  [{r['channel']}]{parts_str}  {role}{standalone}")

    scale  = shared_config.get("downsampling", {}).get("factor", 0.125)
    n_ok   = 0
    n_fail = 0

    # In batch mode run1_desc is set by run1_index; in single-run modes it
    # was already set in the run_filter block above.
    if run_filter is None:
        run1_desc = runs[run1_index]

    for i, run_desc in enumerate(runs):
        if run_filter is not None:
            # Single-run modes: follow-up if ref_run was resolved, else standalone
            is_first = not is_followup_single
        else:
            # Batch mode: only the designated run-1 index uses the first-run pipeline
            is_first = (i == run1_index)
        prefix    = run_desc["prefix"]
        p_label   = ("calcium" if run_desc["has_red"] else "calciumonly")
        p_label  += ("" if is_first else "_nf")

        print(f"\n  --- {prefix}  [{p_label}] ---")

        try:
            data = _load_run(run_desc, scale)

            if is_first:
                cfg = _make_config(shared_config, run_desc["base"], out_dir)
            else:
                cfg = _make_config(
                    shared_config,
                    run_desc["base"],
                    out_dir,
                    run1_filepath=run1_desc["base"],
                    run1_out_dir=out_dir,
                )

            tmp_cfg = _write_temp_config(
                cfg, tmp_root / f"{prefix}_config.yaml"
            )

            run_pipeline = _get_pipeline(run_desc["has_red"], is_first)
            run_pipeline(config_file=str(tmp_cfg), data=data)

            tmp_cfg.unlink(missing_ok=True)
            del data

            n_ok += 1
            print(f"  ✓ {prefix} complete")

        except Exception:
            n_fail += 1
            print(f"  ✗ {prefix} FAILED:")
            traceback.print_exc()

    try:
        tmp_root.rmdir()
    except OSError:
        pass

    return n_ok, n_fail


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def discover_and_run(data_root: str,
                     output_root: str,
                     config_file: str = "config.yaml",
                     subject: str = None,
                     session: str = None,
                     run: str = None,
                     ref_run: str = None,
                     overrides_file: str = None,
                     task: str = None) -> None:
    """
    Scan data_root for BIDS-organised sessions and preprocess all runs.

    Parameters
    ----------
    data_root : str or Path
        Root folder containing sub-XX/ subject directories.
    output_root : str or Path
        Root folder for outputs (mirrored sub-XX/ses-XX structure).
    config_file : str or Path
        Shared parameter config YAML (atlas paths, filter settings, etc.).
    subject : str, optional
        If set (e.g. "sub-01"), process only this subject.
    session : str, optional
        If set (e.g. "ses-2"), process only this session.
        Requires subject to also be set.
    run : str, optional
        If set (e.g. "run-2"), process only this run.
        Requires subject and session to also be set.
    ref_run : str, optional
        If set alongside run (e.g. "run-1"), the targeted run is processed as
        a follow-up using the already-saved outputs of ref_run as references
        (brain mask, atlas, reference frames).  If omitted, the targeted run
        is treated as standalone with its own registration.
        Requires run, subject, and session to all be set.
    overrides_file : str or Path, optional
        Path to an overrides YAML specifying a custom run-1 per session.
        Format:
            sub-01/ses-1: sub-01_ses-1_task-rest_run-2_gb.tiff
            sub-02/ses-1: sub-02_ses-1_task-rest_run-3_gb.tiff
    task : str, optional
        If set (e.g. "rest" or "whisker"), only process runs whose filename
        contains "task-<task>" — matching the task label used in your BIDS
        filenames (e.g. sub-01_ses-1_task-rest_run-1_gb.tiff -> "rest").
        Can be combined with subject/session/run, or used alone to filter
        across the whole dataset. Within each session, "run-1" is chosen
        from among the matching-task runs only, so different tasks (which
        may need different config.yaml settings, e.g. experiment.type)
        can each be run as a separate discover_and_run.py invocation.
    """
    data_root   = Path(data_root)
    output_root = Path(output_root)
    config_file = Path(config_file)
    tmp_root    = output_root / ".tmp"

    print(f"Data root   : {data_root}")
    print(f"Output root : {output_root}")
    print(f"Config      : {config_file}")
    if subject or task:
        filter_bits = []
        if subject:
            filter_bits.append(f"subject={subject}")
        if session:
            filter_bits.append(f"session={session}")
        if run:
            filter_bits.append(f"run={run}")
        if ref_run:
            filter_bits.append(f"ref-run={ref_run}")
        if task:
            filter_bits.append(f"task={task}")
        print(f"Filter      : " + "  ".join(filter_bits))

    shared_config = _load_yaml(config_file)

    # Load per-session run-1 overrides if provided
    overrides: dict = {}
    if overrides_file is not None:
        overrides = _load_yaml(Path(overrides_file)) or {}
        print(f"Overrides   : {overrides_file}  ({len(overrides)} session(s) overridden)")

    # Validate filter combination
    if run is not None and (subject is None or session is None):
        raise ValueError("--run requires both --subject and --session to be specified.")
    if ref_run is not None and run is None:
        raise ValueError("--ref-run requires --run to also be specified.")
    if session is not None and subject is None:
        raise ValueError("--session requires --subject to be specified.")

    # Collect func/ directories and apply subject/session filters
    func_dirs = []
    for func_dir in _iter_func_dirs(data_root):
        sub = func_dir.parent.parent.name
        ses = func_dir.parent.name
        if subject is not None and sub != subject:
            continue
        if session is not None and ses != session:
            continue
        func_dirs.append(func_dir)

    if not func_dirs:
        print("\nNo matching sessions found — check folder structure and filters.")
        return

    print(f"\nFound {len(func_dirs)} session(s) to process.\n")

    total_ok   = 0
    total_fail = 0
    failed_sessions = []

    for func_dir in func_dirs:
        sub_name      = func_dir.parent.parent.name
        ses_name      = func_dir.parent.name
        session_label = f"{sub_name}/{ses_name}"

        # Look up per-session run-1 override
        run1_override = overrides.get(session_label)

        try:
            ok, fail = _process_session(
                func_dir,
                output_root,
                shared_config,
                tmp_root,
                run1_override=run1_override,
                run_filter=run,
                ref_run=ref_run,
                task_filter=task,
            )
            total_ok   += ok
            total_fail += fail
            if fail:
                failed_sessions.append(session_label)
        except Exception:
            total_fail += 1
            failed_sessions.append(session_label)
            print(f"\n✗ Session {session_label} failed entirely:")
            traceback.print_exc()

    # Summary
    print(f"\n{'='*70}")
    print("BATCH COMPLETE")
    print(f"{'='*70}")
    print(f"  Sessions processed : {len(func_dirs)}")
    print(f"  Runs succeeded     : {total_ok}")
    print(f"  Runs failed        : {total_fail}")
    if failed_sessions:
        print("\n  Sessions with failures:")
        for s in failed_sessions:
            print(f"    - {s}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Discover and preprocess widefield imaging runs in a BIDS folder."
    )
    parser.add_argument("--data",      required=True,
                        help="Root folder with sub-XX/ subject directories.")
    parser.add_argument("--output",    required=True,
                        help="Root folder for outputs.")
    parser.add_argument("--config",    default="config.yaml",
                        help="Shared parameter config YAML (default: config.yaml).")
    parser.add_argument("--subject",   default=None,
                        help="Process only this subject, e.g. sub-01.")
    parser.add_argument("--session",   default=None,
                        help="Process only this session, e.g. ses-2. Requires --subject.")
    parser.add_argument("--run",       default=None,
                        help="Process only this run, e.g. run-2. "
                             "Requires --subject and --session.")
    parser.add_argument("--ref-run",   default=None, dest="ref_run",
                        help="When --run is set, reuse this already-preprocessed "
                             "run's brain mask/atlas/references instead of running "
                             "a fresh registration. E.g. --ref-run run-1. "
                             "Requires --run.")
    parser.add_argument("--overrides", default=None,
                        help="YAML file specifying a custom run-1 per session.")
    parser.add_argument("--task",      default=None,
                        help="Process only runs with this task label, e.g. "
                             "'rest' or 'whisker' (matches 'task-<label>' in "
                             "the BIDS filename). Can be combined with "
                             "--subject/--session/--run, or used alone to "
                             "filter across the whole dataset. Handy when "
                             "different tasks need different config.yaml "
                             "settings (e.g. experiment.type).")
    args = parser.parse_args()

    discover_and_run(
        data_root      = args.data,
        output_root    = args.output,
        config_file    = args.config,
        subject        = args.subject,
        session        = args.session,
        run            = args.run,
        ref_run        = args.ref_run,
        overrides_file = args.overrides,
        task           = args.task,
    )
