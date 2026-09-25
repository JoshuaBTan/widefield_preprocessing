# -*- coding: utf-8 -*-
"""
Updated main_pipeline.py with flexible registration direction

Key changes:
- Supports both "atlas_to_mouse" and "mouse_to_atlas" registration
- When using mouse_to_atlas, data is transformed to standardized atlas space
- Atlas is cropped to FOV in mouse_to_atlas mode
"""

from pathlib import Path
import yaml
import pickle
import numpy as np
from skimage.transform import resize, warp
from widefield_pipeline.calcium_io import load_tiff_stack
from widefield_pipeline.preprocessing import (downsample_stack, separate_channels_from_interleaved, 
                                              motion_correction_rigid, pad_to_size, create_vasculature_mask,
                                              apply_spatial_mask, create_vasculature_mask_percentile,
                                              downsample_stack_nanmean)
from widefield_pipeline.normalization import compute_dff, detrend_quadratic, butter_filter
from widefield_pipeline.isolate_calcium import correct_hemodynamic_artifacts
from widefield_pipeline.registration_new import (load_allen_atlas, register_atlas_landmarks,
                         apply_transform_to_stack, apply_transform_to_mask,
                         crop_atlas_to_fov, make_brain_mask_from_atlas, resample_timeseries,
                         resample_frame, make_brain_mask_fixed)
from widefield_pipeline.roi_extraction import (extract_timecourses_from_atlas_fixed, extract_hemodynamic_signals,
                         convert_to_hbt_single_wavelength)
from pipeline_utils import build_output_paths

def get_project_root():
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    else:
        # Spyder / interactive fallback
        return Path.cwd()
    
def run_pipeline(config_file="config.yaml", data=None):
    """
    Run the first-run calcium-only preprocessing pipeline.

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
        project_root = get_project_root()
        config_path = project_root / config_path
    
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Build full output paths from BIDS filename + output dir
    config["output"] = build_output_paths(
        config["data"]["filepath"],
        config["output"]["dir"],
    )

    if data is not None:
        # Pre-loaded stack passed in by batch runner (e.g. concatenated split run)
        stack_ds = data
        print(f"Using pre-loaded stack: shape {stack_ds.shape}")
    else:
        # Load data
        stack = load_tiff_stack(config["data"]["filepath"])
        print("Data load successful")
        # Downsample
        stack_ds = downsample_stack(stack, scale=config["downsampling"]["factor"])
        print("Downsample successful")
    
    # Separate Channels
    ch = separate_channels_from_interleaved(stack_ds, 
                                           frames_per_cycle=config["channel_separation"]["frames_per_cycle"], 
                                           order=config["channel_separation"]["order"])
    print("Channel separation successful")
    
    # Motion Correction
    green_mc, g_shifts = motion_correction_rigid(ch['green'], 
                                                upsample_factor=config["upsample_factor"]["green"], 
                                                return_shifts=True)
    blue_mc, b_shifts = motion_correction_rigid(ch['blue'], 
                                               upsample_factor=config["upsample_factor"]["blue"], 
                                               return_shifts=True)
    print("Motion correction successful")
    
    # Calculate median reference frame for future trials
    green_ref = np.median(green_mc.astype(np.float32), axis=0)
    blue_ref = np.median(blue_mc.astype(np.float32), axis=0)

    # Normalization
    dff_blue, Fb = compute_dff(blue_mc, 
                              baseline_frames=slice(*config["normalization"]["baseline_frames"]), 
                              method=config["normalization"]["method"])
    dff_green, Fg = compute_dff(green_mc, 
                               baseline_frames=slice(*config["normalization"]["baseline_frames"]), 
                               method=config["normalization"]["method"])
    print("Normalization successful")
    
    
    # Reorientate data    
    # Flip raw image for atlas registration  
    mean_blue = np.mean(blue_mc, axis=0)
    mean_blue_oriented = np.rot90(mean_blue, k=config["normalization"]["rotate"])
    
    # Flip corrected data
    blue_mc_oriented = np.rot90(blue_mc, k=config["normalization"]["rotate"], axes=(1,2))
    green_mc_oriented = np.rot90(green_mc, k=config["normalization"]["rotate"], axes=(1,2))
    
    # Flip normalised data
    dff_blue_oriented = np.rot90(dff_blue, k=config["normalization"]["rotate"], axes=(1,2))
    dff_green_oriented = np.rot90(dff_green, k=config["normalization"]["rotate"], axes=(1,2))
    print("Data reorientation successful")
    
    # Correct signal using green (hemodynamics)
    corrected_blue, correction_params = correct_hemodynamic_artifacts(
        dff_blue_oriented, dff_green_oriented, 
        method=config["hemodynamic_correction"]["method"], 
        qc=config["hemodynamic_correction"]["qc"])
    print("Calcium correction successful")
    
    # Temporal filter
    fs = config["channel_separation"]["fs"]  # e.g. 10 Hz (per wavelength!)

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
    elif config["experiment"]["type"] == "task":
        blue_hp = corrected_blue
        green_hp = dff_green_oriented
        print("No temporal filtering")

    # Load in and resize atlas
    atlas, labels = load_allen_atlas(config["atlas"]["filepath"], config["atlas"]["labels"])
    atlas_resized = atlas
    #atlas_resized = resize(atlas, (144,157), order=0, preserve_range=True, anti_aliasing=False).astype(np.int16)
    
    # Load in average template
    template, labels = load_allen_atlas(config["template"]["filepath"], config["atlas"]["labels"])
    print("Loaded in atlas and average template")
    
    
    # Get registration direction from config (default to atlas_to_mouse for backward compatibility)
    registration_direction = config["atlas"].get("registration_direction", "atlas_to_mouse")
    print(f"\nUsing registration direction: {registration_direction}")
    
    # =================================================================
    # REGISTRATION - Two different workflows based on direction
    # =================================================================
    
    if registration_direction == "atlas_to_mouse":
        # ============ ATLAS → MOUSE workflow ============
        print("\n--- ATLAS → MOUSE workflow ---")

        # Register atlas to mouse space via landmarks.
        # Returns: atlas labels warped to mouse space, transform, atlas-footprint
        # brain mask, and the landmark coordinates for both images.
        template_reg, tform, brain_mask, pts_mouse, pts_atlas = register_atlas_landmarks(
            mean_blue_oriented,
            template,
            config["atlas"]["num_points"],
            mode=config["atlas"]["method"],
            registration_direction="atlas_to_mouse"
        )
        print("Template registration successful")

        # Apply the same transform to the full-resolution atlas labels
        print("Transforming atlas labels to mouse space...")
        atlas_reg = warp(
            atlas,
            inverse_map=tform.inverse,
            output_shape=mean_blue_oriented.shape,
            order=0,              # nearest-neighbour preserves integer region IDs
            preserve_range=True,
            cval=0
        ).astype(np.int32)

        # Brain mask step 1: atlas footprint in mouse space
        atlas_footprint = make_brain_mask_from_atlas(atlas_reg)
        print("Atlas footprint mask derived from registered atlas")

        # Brain mask step 2: user draws a mask over the actual brain FOV.
        # This removes non-brain pixels that fall inside the atlas footprint
        # (e.g. due to imperfect registration or imaging artefacts at the edges).
        print("\nDraw a brain mask over the mean image to exclude non-brain pixels.")
        print("This mask will be intersected with the atlas footprint.")
        user_brain_mask = make_brain_mask_fixed(
            mean_blue_oriented,
            method=config["brain_mask"]["method"],
            qc=config["brain_mask"]["qc"],
            interactive_correction=False   # one polygon only; no add/subtract loop
        )

        # Intersect: keep only pixels inside both the atlas footprint and the drawn region
        brain_mask = atlas_footprint & user_brain_mask
        print("Brain mask created (atlas footprint intersected with user-drawn mask)")
        print("Brain mask creation successful")

        # Restrict atlas to brain mask
        atlas_masked = atlas_reg * brain_mask
        atlas_masked = np.nan_to_num(atlas_masked, nan=0)
        print("Atlas restricted to brain mask")
        
        # Data stays in mouse space
        data_for_extraction = blue_hp
        #green_for_hemo = green_mc_oriented
        
        # Save brain-masked pixel data
        corrected_blue_masked = blue_hp.copy()
        corrected_blue_masked[:, ~brain_mask] = np.nan  # or 0
        
        print("\n--- Extracting ROI timecourses ---")
        
        # Extract calcium time series
        roi_timecourses, valid_rois = extract_timecourses_from_atlas_fixed(
            data=data_for_extraction,
            atlas=atlas_masked, 
            brain_mask=brain_mask,
            min_overlap=config["roi_extraction"]["min_overlap"],
            qc=config["roi_extraction"]["qc"]
        )
        print("Calcium ROI time series extracted")
        
        valid_regions = valid_rois
        
        # Extract green signal
        green_masked = green_hp.copy()
        green_masked[:, ~brain_mask] = np.nan
        
        green_timecourses = extract_timecourses_from_atlas_fixed(
            data=green_hp,
            atlas=atlas_masked, 
            brain_mask=brain_mask,
            min_overlap=config["roi_extraction"]["min_overlap"],
            qc=config["roi_extraction"]["qc"]
        )
        print("Green ROI time series extracted")

        # Single-wavelength HbT: calibrated modified Beer-Lambert estimate (μM),
        # using the tabulated extinction coefficients (isosbestic approximation).
        # Uses raw motion-corrected green (not dF/F) since the conversion does
        # its own baseline normalization internally.
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
            qc=config["roi_extraction"]["qc"]
        )
        # Wrapped in a dict (keyed 'hbt') to match the dual-wavelength pipeline's
        # hemo_ts / hemopixel_ts format, even though only one signal is available.
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
            
        with open(config["output"]["atlas_mask"], "wb") as f:
            pickle.dump(atlas_masked, f)
            
        with open(config["output"]["brain_mask"], "wb") as f:
            pickle.dump(brain_mask, f)
            
        with open(config["output"]["green_ref"], "wb") as f:
            pickle.dump(green_ref, f)

        with open(config["output"]["blue_ref"], "wb") as f:
            pickle.dump(blue_ref, f)

        with open(config["output"]["transform"], "wb") as f:
            pickle.dump({'tform': tform.inverse, 'template_shape': template.shape}, f)       
        
    else:  # mouse_to_atlas
    # ============ MOUSE → ATLAS workflow ============
        print("\n--- MOUSE → ATLAS workflow ---")

        # Register mouse image to atlas space via landmarks.
        # Returns: mouse image warped to atlas space, transform, atlas-footprint
        # brain mask (atlas space), and landmark coordinates for both images.
        mouse_in_atlas, tform, brain_mask_atlas, pts_mouse, pts_atlas = register_atlas_landmarks(
            mean_blue_oriented,
            template,
            config["atlas"]["num_points"],
            mode=config["atlas"]["method"],
            registration_direction="mouse_to_atlas"
        )
        print("Mouse → Atlas registration successful")

        # Brain mask step 1: atlas footprint in atlas space (from registration).
        atlas_footprint_atlas = brain_mask_atlas.copy()

        # Brain mask step 2: user draws a mask on the warped mouse image
        # (already in atlas space) to exclude non-brain pixels such as
        # imaging artefacts or areas outside the cranial window.
        # This mask is then intersected with the atlas footprint so that
        # the final brain_mask stays within the registered atlas boundary.
        print("\nDraw a brain mask over the registered mouse image (atlas space).")
        print("This mask will be intersected with the atlas footprint.")
        user_brain_mask_atlas = make_brain_mask_fixed(
            mouse_in_atlas,
            method=config["brain_mask"]["method"],
            qc=config["brain_mask"]["qc"],
            interactive_correction=False
        )

        # Intersect: keep only pixels inside both the atlas footprint and
        # the user-drawn region — identical logic to atlas_to_mouse.
        brain_mask_atlas = atlas_footprint_atlas & user_brain_mask_atlas
        print("Brain mask created (atlas footprint intersected with user-drawn mask)")
        print("Brain mask creation successful")

        # --- Warp data to atlas space ---
        print("Transforming calcium data to atlas space...")
        corrected_blue_atlas = apply_transform_to_stack(
            blue_hp, tform, output_shape=template.shape, order=1
        )

        print("Transforming green (dff) data to atlas space...")
        dff_green_atlas = apply_transform_to_stack(
            green_hp, tform, output_shape=template.shape, order=1
        )

        # --- Crop atlas to the registered FOV ---
        # Uses the refined brain_mask_atlas (atlas footprint ∩ user mask)
        # so ROI definitions respect both the atlas boundary and the drawn FOV.
        atlas_masked, valid_regions = crop_atlas_to_fov(
            atlas_resized, brain_mask_atlas,
            min_overlap=config["roi_extraction"]["min_overlap"]
        )
        print(f"Atlas cropped to FOV ({len(valid_regions)} regions)")

        # --- Extract ROI timecourses (atlas space) ---
        print("\n--- Extracting ROI timecourses ---")
        roi_timecourses, valid_rois = extract_timecourses_from_atlas_fixed(
            data=corrected_blue_atlas,
            atlas=atlas_masked,
            brain_mask=brain_mask_atlas,
            min_overlap=config["roi_extraction"]["min_overlap"],
            qc=config["roi_extraction"]["qc"]
        )
        print("Calcium ROI time series extracted")
        valid_regions = valid_rois

        green_timecourses, _ = extract_timecourses_from_atlas_fixed(
            data=dff_green_atlas,
            atlas=atlas_masked,
            brain_mask=brain_mask_atlas,
            min_overlap=config["roi_extraction"]["min_overlap"],
            qc=config["roi_extraction"]["qc"]
        )
        print("Green ROI time series extracted")

        # Single-wavelength HbT: calibrated modified Beer-Lambert estimate (μM),
        # using the tabulated extinction coefficients (isosbestic approximation).
        # Computed in mouse space (raw motion-corrected green), then warped to atlas
        # space the same way calcium/green data are, and extracted with the same
        # crop_atlas_to_fov-derived atlas/mask.
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
            hbt_signal, tform, output_shape=template.shape, order=1
        )
        hbt_timecourses, _ = extract_timecourses_from_atlas_fixed(
            data=hbt_atlas,
            atlas=atlas_masked,
            brain_mask=brain_mask_atlas,
            min_overlap=config["roi_extraction"]["min_overlap"],
            qc=config["roi_extraction"]["qc"]
        )
        hemo_roi_timecourses = {'hbt': hbt_timecourses}
        hemo_pixel_timecourses = {'hbt': hbt_atlas}
        print("HbT (single-wavelength) ROI time series extracted")

        # --- Save outputs ---
        # transform: used by nf pipelines to warp subsequent runs to atlas space.
        # atlas_masked: the atlas label image cropped to this session's FOV;
        #               shared with all nf runs so ROI definitions are identical.
        # brain_mask: atlas-space footprint used for ROI extraction.
        # green_ref / blue_ref: median frames in mouse space (pre-rotation)
        #               used by nf pipelines for motion correction.
        with open(config["output"]["transform"], "wb") as f:
            pickle.dump({'tform': tform, 'template_shape': template.shape}, f)
        print("Transform saved")

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
            pickle.dump(valid_regions, f)

        with open(config["output"]["atlas_mask"], "wb") as f:
            pickle.dump(atlas_masked, f)

        with open(config["output"]["brain_mask"], "wb") as f:
            pickle.dump(brain_mask_atlas, f)

        with open(config["output"]["green_ref"], "wb") as f:
            pickle.dump(green_ref, f)

        with open(config["output"]["blue_ref"], "wb") as f:
            pickle.dump(blue_ref, f)

        print("\nAll files saved")
        print("Preprocessing was successful")
    
    
    #==================================================================
    # SUMMARY
    #==================================================================
    
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    print(f"Registration direction: {registration_direction}")
    if registration_direction == "mouse_to_atlas":
        print("  → Data is in standardized atlas space")
        print("  → Atlas is perfectly vertical/aligned")
        print("  → Ready for group-level analyses")
    else:
        print("  → Data is in native mouse space")
        print("  → Atlas warped to match your FOV")
    print(f"Number of ROIs extracted: {len(valid_regions)}")
    print("Hemodynamic signals: ['hbt'] (single-wavelength, calibrated \u03bcM, isosbestic approximation)")
    print("="*60)

    return

if __name__ == "__main__":
    run_pipeline("config.yaml")