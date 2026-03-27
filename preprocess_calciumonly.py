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
                         resample_frame)
from widefield_pipeline.roi_extraction import extract_timecourses_from_atlas_fixed, extract_hemodynamic_signals

def get_project_root():
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    else:
        # Spyder / interactive fallback
        return Path.cwd()
    
def run_pipeline(config_file="config.yaml"):
    
    config_path = Path(config_file)
    
    if not config_path.is_absolute():
        project_root = get_project_root()
        config_path = project_root / config_path
    
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Build full output paths from dir + filename
    out_dir = Path(config["output"]["dir"])
    for key, filename in config["output"].items():
        if key != "dir":
            config["output"][key] = str(out_dir / filename)

    # Load data
    stack = load_tiff_stack(config["data"]["filepath"])
    print("Data load successful")


    # Preprocess
    
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
        print("Temporal filtering applied")
    elif config["experiment"]["type"] == "task":
        blue_hp = corrected_blue
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

        # Brain mask = atlas footprint in mouse space (no manual drawing needed)
        brain_mask = make_brain_mask_from_atlas(atlas_reg)
        print("Brain mask derived from atlas footprint")
        print("Brain mask creation successful")
        
        # Restrict atlas to brain mask
        atlas_masked = atlas_reg * brain_mask
        atlas_masked = np.nan_to_num(atlas_masked, nan=0)
        print("Atlas restricted to brain mask")
        
        # Data stays in mouse space
        data_for_extraction = blue_hp
        green_for_hemo = green_mc_oriented
        
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
        green_masked = dff_green_oriented.copy()
        green_masked[:, ~brain_mask] = np.nan
        
        green_timecourses = extract_timecourses_from_atlas_fixed(
            data=dff_green_oriented,
            atlas=atlas_masked, 
            brain_mask=brain_mask,
            min_overlap=config["roi_extraction"]["min_overlap"],
            qc=config["roi_extraction"]["qc"]
        )
        print("Green ROI time series extracted")        
        print("\n--- Saving outputs ---")
        
        with open(config["output"]["pixel_ts"], "wb") as f:
            pickle.dump(corrected_blue_masked, f)
        
        with open(config["output"]["roi_ts"], "wb") as f:
            pickle.dump(roi_timecourses, f)
            
        with open(config["output"]["green_pixel"], "wb") as f:
            pickle.dump(green_masked, f)
            
        with open(config["output"]["green_ts"], "wb") as f:
            pickle.dump(green_timecourses, f)
            
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
        
    else:  # mouse_to_atlas
    # ============ MOUSE → ATLAS workflow ============
        print("\n--- MOUSE → ATLAS workflow ---")

        # Register mouse image to atlas space via landmarks.
        # brain_mask_atlas is the atlas footprint in atlas space.
        # brain_mask_mouse is its inverse-warp back into mouse space.
        mouse_in_atlas, tform, brain_mask_atlas, pts_mouse, pts_atlas = register_atlas_landmarks(
            mean_blue_oriented,
            template,
            config["atlas"]["num_points"],
            mode=config["atlas"]["method"],
            registration_direction="mouse_to_atlas"
        )
        print("Mouse → Atlas registration successful")

        # Brain mask in mouse space = inverse-warp of the atlas footprint
        brain_mask_mouse = apply_transform_to_mask(
            brain_mask_atlas, tform, output_shape=mean_blue_oriented.shape
        )
        print("Brain mask derived from atlas footprint (mouse space)")
    
        # --- Apply brain mask in mouse space before transforming ---
        corrected_blue_masked = blue_hp.copy()
        corrected_blue_masked[:, ~brain_mask_mouse] = np.nan
    
        dff_green_masked = dff_green_oriented.copy()
        dff_green_masked[:, ~brain_mask_mouse] = np.nan
    
        # --- Transform all stacks to atlas space in one pass ---
        print("Transforming calcium data to atlas space...")
        corrected_blue_atlas = apply_transform_to_stack(
            corrected_blue_masked, tform, output_shape=template.shape, order=1
        )
    
        print("Transforming green (dff) data to atlas space...")
        dff_green_atlas = apply_transform_to_stack(
            dff_green_masked, tform, output_shape=template.shape, order=1
        )
    
        # --- Transform reference frames ---
        print("Transforming reference frames...")
        for ref, name in [(red_ref, "red"), (green_ref, "green"), (blue_ref, "blue")]:
            pass  # done below with named variables
        
        green_atlas = warp(green_ref, inverse_map=tform.inverse,
                           output_shape=template.shape, order=1,
                           preserve_range=True, cval=np.nan)
        blue_atlas = warp(blue_ref, inverse_map=tform.inverse,
                          output_shape=template.shape, order=1,
                          preserve_range=True, cval=np.nan)
    
        # --- Crop atlas to transformed FOV ---
        atlas_masked, valid_regions = crop_atlas_to_fov(
            atlas_resized, brain_mask_atlas,
            min_overlap=config["roi_extraction"]["min_overlap"]
        )
        print(f"Atlas cropped to FOV ({len(valid_regions)} regions)")
        
        # --- Extract calcium ROI timecourses (in atlas space) ---
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
    
        # --- Extract green ROI timecourses (in atlas space) ---
        green_timecourses, _ = extract_timecourses_from_atlas_fixed(
            data=dff_green_atlas,
            atlas=atlas_masked,
            brain_mask=brain_mask_atlas,
            min_overlap=config["roi_extraction"]["min_overlap"],
            qc=config["roi_extraction"]["qc"]
        )
        print("Green ROI time series extracted")
    
        # --- Save transform ---
        with open(config["output"]["transform"], "wb") as f:
            pickle.dump({'tform': tform, 'template_shape': template.shape}, f)
        print("Transform saved")
    
        # --- Save all outputs ---
        print("\n--- Saving outputs ---")
    
        with open(config["output"]["pixel_ts"], "wb") as f:
            pickle.dump(corrected_blue_atlas, f)
    
        with open(config["output"]["roi_ts"], "wb") as f:
            pickle.dump(roi_timecourses, f)
    
        with open(config["output"]["green_pixel"], "wb") as f:
            pickle.dump(dff_green_atlas, f)
    
        with open(config["output"]["green_ts"], "wb") as f:
            pickle.dump(green_timecourses, f)
    
        with open(config["output"]["roi_id"], "wb") as f:
            pickle.dump(valid_regions, f)
    
        with open(config["output"]["atlas_mask"], "wb") as f:
            pickle.dump(atlas_masked, f)
    
        with open(config["output"]["brain_mask"], "wb") as f:
            pickle.dump(brain_mask_atlas, f)
    
        with open(config["output"]["green_ref"], "wb") as f:
            pickle.dump(green_atlas, f)
    
        with open(config["output"]["blue_ref"], "wb") as f:
            pickle.dump(blue_atlas, f)
    
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

    return

if __name__ == "__main__":
    run_pipeline("config.yaml")