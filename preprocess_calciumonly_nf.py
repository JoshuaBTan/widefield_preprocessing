# -*- coding: utf-8 -*-
"""
preprocess_calciumonly_nf.py  —  Follow-up run pipeline (green + blue channels only)

Use this script for runs 2, 3, … of the same session.
It reuses the reference frames, brain mask, registered atlas (and, for
mouse_to_atlas mode, the spatial transform) that were saved by the first run,
so that all runs within a session share exactly the same spatial registration.

Steps performed
---------------
1.  Load raw TIFF, downsample, separate channels (green / blue only).
2.  Motion-correct each channel against the run-1 reference frames.
3.  Compute ΔF/F, reorient, correct hemodynamic contamination, filter.
4.  Load run-1 brain mask and registered atlas  (no registration or brain-mask
    drawing needed).
5.  For mouse_to_atlas: apply the saved spatial transform to warp data to
    atlas space before extraction.
6.  Extract calcium and green ROI timecourses.
7.  Save outputs.

Config
------
Use config_nf_template.yaml (calcium-only variant) as your starting point.
The key difference from the first-run config is the `reference` section,
which points to the pkl files produced by run 1.
"""

from pathlib import Path
import yaml
import pickle
import numpy as np
from skimage.transform import warp
from widefield_pipeline.calcium_io import load_tiff_stack
from widefield_pipeline.preprocessing import (
    downsample_stack,
    separate_channels_from_interleaved,
    motion_correction_rigid,
)
from widefield_pipeline.normalization import compute_dff, butter_filter
from widefield_pipeline.isolate_calcium import correct_hemodynamic_artifacts
from widefield_pipeline.registration_new import apply_transform_to_stack
from widefield_pipeline.roi_extraction import extract_timecourses_from_atlas_fixed, convert_to_hbt_single_wavelength
from pipeline_utils import build_output_paths, reference_paths_from_run1


def get_project_root():
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    else:
        return Path.cwd()


def run_pipeline(config_file="config_nf.yaml", data=None):
    """
    Run the follow-up calcium-only preprocessing pipeline.

    Parameters
    ----------
    config_file : str
        Path to the YAML config file.
    data : np.ndarray, optional
        Pre-loaded and pre-downsampled stack (T, H, W). When provided by the
        batch/discovery script (e.g. for split runs concatenated after
        downsampling), load_tiff_stack and downsample_stack are skipped.
        config's data.filepath is still used to derive the output filename prefix.
    """
    config_path = Path(config_file)
    if not config_path.is_absolute():
        config_path = get_project_root() / config_path

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Build full output paths from BIDS filename + output dir
    config["output"] = build_output_paths(
        config["data"]["filepath"],
        config["output"]["dir"],
    )

    # Reconstruct run-1 reference paths automatically from run-1 BIDS filename
    # and its output directory. Any paths already listed explicitly in config
    # under reference: will override these (setdefault only fills missing keys).
    run1_ref = reference_paths_from_run1(
        config["reference"]["run1_filepath"],
        config["reference"]["run1_out_dir"],
        has_red=False,
    )
    for key, path in run1_ref.items():
        config["reference"].setdefault(key, path)

    registration_direction = config["atlas"].get(
        "registration_direction", "atlas_to_mouse"
    )
    print(f"Registration direction (inherited from run 1): {registration_direction}")

    # ------------------------------------------------------------------
    # 1. Load & preprocess raw data
    # ------------------------------------------------------------------
    if data is not None:
        # Pre-loaded stack passed in by batch runner (e.g. concatenated split run)
        stack_ds = data
        print(f"Using pre-loaded stack: shape {stack_ds.shape}")
    else:
        stack = load_tiff_stack(config["data"]["filepath"])
        print("Data load successful")
        stack_ds = downsample_stack(stack, scale=config["downsampling"]["factor"])
        print("Downsample successful")

    ch = separate_channels_from_interleaved(
        stack_ds,
        frames_per_cycle=config["channel_separation"]["frames_per_cycle"],
        order=config["channel_separation"]["order"],
    )
    print("Channel separation successful")

    # ------------------------------------------------------------------
    # 2. Load run-1 reference frames and motion-correct against them
    # ------------------------------------------------------------------
    with open(config["reference"]["green_ref"], "rb") as f:
        green_ref = pickle.load(f)
    with open(config["reference"]["blue_ref"], "rb") as f:
        blue_ref = pickle.load(f)
    print("Run-1 reference frames loaded")

    green_mc, _ = motion_correction_rigid(
        ch["green"],
        reference=green_ref,
        upsample_factor=config["upsample_factor"]["green"],
        return_shifts=True,
    )
    blue_mc, _ = motion_correction_rigid(
        ch["blue"],
        reference=blue_ref,
        upsample_factor=config["upsample_factor"]["blue"],
        return_shifts=True,
    )
    print("Motion correction successful")

    # ------------------------------------------------------------------
    # 3. Normalise, reorient, hemodynamic correction, temporal filter
    # ------------------------------------------------------------------
    k = config["normalization"]["rotate"]

    dff_blue, _ = compute_dff(
        blue_mc,
        baseline_frames=slice(*config["normalization"]["baseline_frames"]),
        method=config["normalization"]["method"],
    )
    dff_green, _ = compute_dff(
        green_mc,
        baseline_frames=slice(*config["normalization"]["baseline_frames"]),
        method=config["normalization"]["method"],
    )
    print("Normalization successful")

    dff_blue_oriented  = np.rot90(dff_blue,  k=k, axes=(1, 2))
    dff_green_oriented = np.rot90(dff_green, k=k, axes=(1, 2))
    # Raw (non-dF/F) reoriented green — needed for the HbT conversion, which
    # does its own baseline normalization internally.
    green_mc_oriented  = np.rot90(green_mc,  k=k, axes=(1, 2))
    print("Data reorientation successful")

    corrected_blue, _ = correct_hemodynamic_artifacts(
        dff_blue_oriented,
        dff_green_oriented,
        method=config["hemodynamic_correction"]["method"],
        qc=config["hemodynamic_correction"]["qc"],
    )
    print("Calcium correction successful")

    fs = config["channel_separation"]["fs"]
    if config["experiment"]["type"] == "rest":
        blue_hp = butter_filter(
            corrected_blue,
            fs,
            lowcut=config["normalization"]["highpass"],
            highcut=config["normalization"]["lowpass"],
        )
        green_hp = butter_filter(
            dff_green_oriented,
            fs,
            lowcut=config["normalization"]["highpass"],
            highcut=config["normalization"]["lowpass"],
        )
        print("Temporal filtering applied")
    else:
        blue_hp = corrected_blue
        green_hp = dff_green_oriented
        print("No temporal filtering")

    # ------------------------------------------------------------------
    # 4. Load run-1 atlas mask and brain mask
    #
    #  Both directions load brain_mask from run-1:
    #    atlas_to_mouse : brain_mask is in mouse space.
    #    mouse_to_atlas : brain_mask is in atlas space (drawn on the
    #                     warped mouse image during run-1 registration).
    # ------------------------------------------------------------------
    with open(config["reference"]["atlas_mask"], "rb") as f:
        atlas_masked = pickle.load(f)
    with open(config["reference"]["brain_mask"], "rb") as f:
        brain_mask = pickle.load(f)
    print("Run-1 atlas mask and brain mask loaded")

    if registration_direction == "mouse_to_atlas":
        # Rename for clarity in step 5 — brain_mask is in atlas space.
        brain_mask_atlas = brain_mask

    # ------------------------------------------------------------------
    # 5. Branch on registration direction
    # ------------------------------------------------------------------
    if registration_direction == "atlas_to_mouse":
        # --------------------------------------------------------------
        # Data stays in mouse space — apply brain mask and extract
        # --------------------------------------------------------------
        print("\n--- ATLAS → MOUSE follow-up workflow ---")

        corrected_blue_masked = blue_hp.copy()
        corrected_blue_masked[:, ~brain_mask] = np.nan

        green_masked = green_hp.copy()
        green_masked[:, ~brain_mask] = np.nan

        print("\n--- Extracting ROI timecourses ---")

        roi_timecourses, valid_rois = extract_timecourses_from_atlas_fixed(
            data=blue_hp,
            atlas=atlas_masked,
            brain_mask=brain_mask,
            min_overlap=config["roi_extraction"]["min_overlap"],
            qc=config["roi_extraction"]["qc"],
        )
        print("Calcium ROI time series extracted")

        green_timecourses, _ = extract_timecourses_from_atlas_fixed(
            data=green_hp,
            atlas=atlas_masked,
            brain_mask=brain_mask,
            min_overlap=config["roi_extraction"]["min_overlap"],
            qc=config["roi_extraction"]["qc"],
        )
        print("Green ROI time series extracted")

        # Single-wavelength HbT: calibrated modified Beer-Lambert estimate (μM),
        # using tabulated extinction coefficients (isosbestic approximation).
        print("Computing single-wavelength HbT from green channel...")
        hbt_signal = convert_to_hbt_single_wavelength(
            green_mc_oriented,
            baseline_frames=slice(*config["hemodynamic_extraction"]["baseline_frames"]),
            green_wavelength=config["wavelength"]["green"],
            pathlength_green=config["hemodynamic_extraction"].get("pathlength_green", 0.057),
            extinction_filepath=config["hemodynamic_extraction"]["extinction_filepath"]
        )
        hbt_masked = hbt_signal.copy()
        hbt_masked[:, ~brain_mask] = np.nan

        hbt_timecourses, _ = extract_timecourses_from_atlas_fixed(
            data=hbt_signal,
            atlas=atlas_masked,
            brain_mask=brain_mask,
            min_overlap=config["roi_extraction"]["min_overlap"],
            qc=config["roi_extraction"]["qc"],
        )
        hemo_roi_timecourses = {'hbt': hbt_timecourses}
        hemo_pixel_timecourses = {'hbt': hbt_masked}
        print("HbT (single-wavelength) ROI time series extracted")

        print("\n--- Saving outputs ---")
        with open(config["output"]["pixel_ts"], "wb") as f:
            pickle.dump(corrected_blue_masked, f)
        with open(config["output"]["roi_ts"], "wb") as f:
            pickle.dump(roi_timecourses, f)
        with open(config["output"]["green_pixel"], "wb") as f:
            pickle.dump(green_masked, f)
        with open(config["output"]["green_ts"], "wb") as f:
            pickle.dump(green_timecourses, f)
        with open(config["output"]["hemo_ts"], "wb") as f:
            pickle.dump(hemo_roi_timecourses, f)
        with open(config["output"]["hemopixel_ts"], "wb") as f:
            pickle.dump(hemo_pixel_timecourses, f)
        with open(config["output"]["roi_id"], "wb") as f:
            pickle.dump(valid_rois, f)

    else:
        # --------------------------------------------------------------
        # mouse_to_atlas: warp data to atlas space using the saved transform.
        # No pre-warp masking — pixels outside atlas ROIs are excluded at
        # extraction time by extract_timecourses_from_atlas_fixed.
        # --------------------------------------------------------------
        print("\n--- MOUSE → ATLAS follow-up workflow ---")

        with open(config["reference"]["transform"], "rb") as f:
            transform_data = pickle.load(f)
        tform          = transform_data["tform"]
        template_shape = transform_data["template_shape"]
        print("Run-1 spatial transform loaded")

        print("Transforming calcium data to atlas space...")
        corrected_blue_atlas = apply_transform_to_stack(
            blue_hp, tform, output_shape=template_shape, order=1
        )

        print("Transforming green data to atlas space...")
        dff_green_atlas = apply_transform_to_stack(
            green_hp, tform, output_shape=template_shape, order=1
        )

        print("\n--- Extracting ROI timecourses ---")

        roi_timecourses, valid_rois = extract_timecourses_from_atlas_fixed(
            data=corrected_blue_atlas,
            atlas=atlas_masked,
            brain_mask=brain_mask_atlas,
            min_overlap=config["roi_extraction"]["min_overlap"],
            qc=config["roi_extraction"]["qc"],
        )
        print("Calcium ROI time series extracted")

        green_timecourses, _ = extract_timecourses_from_atlas_fixed(
            data=dff_green_atlas,
            atlas=atlas_masked,
            brain_mask=brain_mask_atlas,
            min_overlap=config["roi_extraction"]["min_overlap"],
            qc=config["roi_extraction"]["qc"],
        )
        print("Green ROI time series extracted")

        # Single-wavelength HbT: calibrated modified Beer-Lambert estimate (μM).
        # Computed in mouse space, then warped to atlas space like calcium/green.
        print("Computing single-wavelength HbT from green channel...")
        hbt_signal = convert_to_hbt_single_wavelength(
            green_mc_oriented,
            baseline_frames=slice(*config["hemodynamic_extraction"]["baseline_frames"]),
            green_wavelength=config["wavelength"]["green"],
            pathlength_green=config["hemodynamic_extraction"].get("pathlength_green", 0.057),
            extinction_filepath=config["hemodynamic_extraction"]["extinction_filepath"]
        )
        print("Transforming HbT data to atlas space...")
        hbt_atlas = apply_transform_to_stack(
            hbt_signal, tform, output_shape=template_shape, order=1
        )
        hbt_timecourses, _ = extract_timecourses_from_atlas_fixed(
            data=hbt_atlas,
            atlas=atlas_masked,
            brain_mask=brain_mask_atlas,
            min_overlap=config["roi_extraction"]["min_overlap"],
            qc=config["roi_extraction"]["qc"],
        )
        hemo_roi_timecourses = {'hbt': hbt_timecourses}
        hemo_pixel_timecourses = {'hbt': hbt_atlas}
        print("HbT (single-wavelength) ROI time series extracted")

        print("\n--- Saving outputs ---")
        with open(config["output"]["pixel_ts"], "wb") as f:
            pickle.dump(corrected_blue_atlas, f)
        with open(config["output"]["roi_ts"], "wb") as f:
            pickle.dump(roi_timecourses, f)
        with open(config["output"]["green_pixel"], "wb") as f:
            pickle.dump(dff_green_atlas, f)
        with open(config["output"]["green_ts"], "wb") as f:
            pickle.dump(green_timecourses, f)
        with open(config["output"]["hemo_ts"], "wb") as f:
            pickle.dump(hemo_roi_timecourses, f)
        with open(config["output"]["hemopixel_ts"], "wb") as f:
            pickle.dump(hemo_pixel_timecourses, f)
        with open(config["output"]["roi_id"], "wb") as f:
            pickle.dump(valid_rois, f)

    print("\nAll files saved")
    print("Preprocessing was successful")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY  (follow-up run)")
    print("=" * 60)
    print(f"Registration direction : {registration_direction}")
    print(f"Number of ROIs extracted: {len(valid_rois)}")
    print("Hemodynamic signals    : ['hbt'] (single-wavelength, calibrated \u03bcM)")
    if registration_direction == "mouse_to_atlas":
        print("  → Data is in standardized atlas space (matches run 1)")
    else:
        print("  → Data is in native mouse space (matches run 1)")
    print("=" * 60)

    return


if __name__ == "__main__":
    run_pipeline("config_nf.yaml")
