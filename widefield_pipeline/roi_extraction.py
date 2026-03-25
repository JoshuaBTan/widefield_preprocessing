# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 16:29:49 2025

@author: User
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import warnings

# ---------------------
# ROI Extraction using brain mask and atlas
# ---------------------
def extract_timecourses_from_atlas_fixed(
    data, atlas, brain_mask=None, min_overlap=50, qc=True, qc_mode="subsample"
):
    """
    Extract ROI time series from atlas, restricted to brain mask.

    Parameters
    ----------
    data : 3D numpy array
        Imaging data. Can be (T, H, W) or (H, W, T).
    atlas : 2D numpy array
        Registered atlas labels (H x W).
    brain_mask : 2D numpy array, optional
        Binary mask (same H x W). If None, whole atlas is used.
    min_overlap : int
        Minimum number of valid pixels required to keep an ROI.
    qc : bool
        Show QC plots.
    qc_mode : str
        How to display QC results. Options:
        - "subsample": (default) show ~10 ROIs overlaid + 5 timecourses
        - "all_overlay": overlay ALL ROIs on atlas, plot all timecourses in one panel
        - "grid": grid of subplots with all ROI timecourses
        - "per_roi": one figure per ROI with atlas mask + timecourse
    """

    # Detect data format and reshape
    if data.ndim != 3:
        raise ValueError(f"Data must be 3D, got shape {data.shape}")
    #if data.shape[0] > max(data.shape[1], data.shape[2]):  # (T, H, W)
        #data = np.transpose(data, (1, 2, 0))  # -> (H, W, T)
        #print(f"Detected (T,H,W), transposed to (H,W,T): {data.shape}")
    #else:
        #print(f"Detected (H,W,T): {data.shape}")
    
    # Force transpose since data is typically (T, H, W)
    data = np.transpose(data, (1, 2, 0))  # -> (H, W, T)
    H, W, T = data.shape

    if atlas.shape != (H, W):
        raise ValueError(f"Atlas shape {atlas.shape} doesn't match data spatial dimensions ({H},{W})")

    # Apply brain mask
    atlas_work = atlas.copy().astype(int)
    if brain_mask is not None:
        if brain_mask.shape != (H, W):
            raise ValueError(f"Brain mask shape {brain_mask.shape} doesn't match ({H},{W})")
        atlas_work = (atlas_work * brain_mask.astype(int)).astype(int)

    roi_timecourses = {}
    valid_labels = []
    labels = np.unique(atlas_work)
    labels = labels[labels > 0]
    labels = [int(l) for l in labels]

    print(f"Found {len(labels)} potential ROIs in atlas")

    for label in labels:
        roi_mask = atlas_work == label
        n_pix = np.sum(roi_mask)
        if n_pix < min_overlap:
            continue
        tc = np.nanmean(data[roi_mask, :], axis=0)
        roi_timecourses[label] = tc
        valid_labels.append(label)

    print(f"Extracted {len(valid_labels)} ROI timecourses (min_overlap={min_overlap} pixels)")

    # === QC plotting ===
    if qc and len(valid_labels) > 0:
        if qc_mode == "subsample":
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            mean_img = np.mean(data, axis=2)
            axes[0].imshow(mean_img, cmap="gray")

            colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(valid_labels))))
            for i, label in enumerate(valid_labels[:10]):
                roi_mask = atlas_work == label
                axes[0].contour(roi_mask, colors=[colors[i]], linewidths=1, levels=[0.5])
            axes[0].set_title(f"Valid ROIs (n={len(valid_labels)}, showing 10)")
            axes[0].axis("off")

            for i, label in enumerate(valid_labels[:5]):
                axes[1].plot(roi_timecourses[label], label=f"ROI {label}", alpha=0.8)
            axes[1].set_title("Sample ROI timecourses")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        elif qc_mode == "all_overlay":
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))

            mean_img = np.mean(data, axis=2)
            axes[0].imshow(mean_img, cmap="gray")

            colors = plt.cm.tab20(np.linspace(0, 1, len(valid_labels)))
            for i, label in enumerate(valid_labels):
                roi_mask = atlas_work == label
                axes[0].contour(roi_mask, colors=[colors[i % 20]], linewidths=1, levels=[0.5])
            axes[0].set_title(f"All {len(valid_labels)} ROIs")
            axes[0].axis("off")

            for label in valid_labels:
                axes[1].plot(roi_timecourses[label], alpha=0.7, label=f"ROI {label}")
            axes[1].set_title("All ROI timecourses")
            axes[1].legend(fontsize=6, ncol=3)
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        elif qc_mode == "grid":
            n = len(valid_labels)
            ncols = 5
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3 * nrows), sharex=True, sharey=True)
            axes = axes.flatten()

            for i, label in enumerate(valid_labels):
                axes[i].plot(roi_timecourses[label], color="blue")
                axes[i].set_title(f"ROI {label}", fontsize=8)
                axes[i].grid(True, alpha=0.3)

            for j in range(i + 1, len(axes)):
                axes[j].axis("off")

            plt.tight_layout()
            plt.show()

        elif qc_mode == "per_roi":
            for label in valid_labels:
                tc = roi_timecourses[label]
                fig, ax = plt.subplots(1, 2, figsize=(10, 4))

                ax[0].imshow(atlas_work, cmap="gray")
                ax[0].contour(atlas_work == label, colors="r", linewidths=1)
                ax[0].set_title(f"ROI {label} mask")
                ax[0].axis("off")

                ax[1].plot(tc, color="blue")
                ax[1].set_title(f"ROI {label} timecourse")
                ax[1].set_xlabel("Frame")
                ax[1].set_ylabel("Signal")
                ax[1].grid(True, alpha=0.3)

                plt.tight_layout()
                plt.show()

        else:
            raise ValueError(f"Unknown qc_mode: {qc_mode}")

    return roi_timecourses, valid_labels


# -----------------------
# Extinction coefficient data
# -----------------------
def get_extinction_coefficients():
    """
    Returns extinction coefficients for HbO and HbR at common wavelengths.
    Based on Ma et al. 2016 tabulated data (cm^-1 M^-1).
    """
    
    # Use values specified by Ma et al., 2016
    NIRS_val = sio.loadmat("C:/Users/User/OneDrive/Documents/Calcium_Imaging/thanh/NIRS_extData.mat")
    wavelengths = NIRS_val['lambda'].ravel()
    hbo_ext = NIRS_val['eHbO2'].ravel()
    hbr_ext = NIRS_val['eHb'].ravel()
    
    return wavelengths, hbo_ext, hbr_ext

def get_extinction_at_wavelength(wavelength, species='hbo'):
    """
    Get extinction coefficient at specific wavelength via interpolation.
    
    Parameters
    ----------
    wavelength : float or array
        Wavelength(s) in nm
    species : str
        'hbo' for oxyhemoglobin, 'hbr' for deoxyhemoglobin
    
    Returns
    -------
    float or array
        Extinction coefficient(s) in cm^-1 M^-1
    """
    wavelengths, hbo_ext, hbr_ext = get_extinction_coefficients()
    
    if species.lower() == 'hbo':
        ext_data = hbo_ext
    elif species.lower() == 'hbr':
        ext_data = hbr_ext
    else:
        raise ValueError("species must be 'hbo' or 'hbr'")
    
    # Interpolate
    interp_func = interp1d(wavelengths, ext_data, kind='cubic', 
                          bounds_error=False, fill_value='extrapolate')
    return interp_func(wavelength)

# -----------------------
# Hemodynamic signal conversion
# -----------------------
def convert_to_hbt_concentration(green_signal, baseline_frames=slice(0, 100)):
    """
    Convert green channel signal to total hemoglobin (HbT) concentration changes.
    This is a simplified single-wavelength approach.
    
    Parameters
    ----------
    green_signal : array
        Green channel data (T,H,W) or (H,W,T)
    baseline_frames : slice or array
        Frames to use for baseline calculation
    
    Returns
    -------
    hbt_signal : array
        Relative HbT concentration changes (same shape as input)
    """
    # Auto-detect format
    # if green_signal.ndim == 3:
    #     if green_signal.shape[0] > max(green_signal.shape[1], green_signal.shape[2]):
    #         # (T,H,W) format
    #         data = green_signal.astype(np.float32)
    #         transposed = False
    #     else:
    #         # (H,W,T) format - transpose to (T,H,W)
    #         data = np.transpose(green_signal, (2, 0, 1)).astype(np.float32)
    #         transposed = True
    # else:
    #     raise ValueError("Input must be 3D array")
    
    # If data is in (T, H, W)
    if green_signal.ndim != 3:
        raise ValueError("Input must be 3D array")
    else:
        data = green_signal.astype(np.float32)
        transposed = False
    
    T, H, W = data.shape
    
    # Calculate baseline
    baseline_data = np.mean(data[baseline_frames], axis=0)  # (H,W)
    
    # Avoid division by zero
    baseline_data[baseline_data <= 0] = 1e-6
    
    # Calculate percentage change
    percent_change = 100 * (data - baseline_data) / baseline_data
    
    # Convert to HbT using simplified relationship
    # Negative because increased HbT reduces transmitted light
    hbt_signal = -percent_change
    
    # Return in original format
    if transposed:
       hbt_signal = np.transpose(hbt_signal, (1, 2, 0))
    
    return hbt_signal
    

def convert_to_hemoglobin_concentrations(green_signal, red_signal, 
                                       green_wavelength=530, red_wavelength=625,
                                       baseline_frames=slice(0, 100),
                                       pathlength_green=0.057, pathlength_red=0.25,
                                       clip_ratio=False):
    """
    Convert dual-wavelength optical data to HbO, HbR, and HbT concentrations.
    
    Parameters
    ----------
    green_signal, red_signal : array
        Raw intensity data (T,H,W) or (H,W,T)
    green_wavelength, red_wavelength : float
        Wavelengths in nm (default: 530, 625)
    baseline_frames : slice or array
        Frames for baseline calculation
    pathlength_green, pathlength_red : float
        Differential pathlength factors in cm
    clip_ratio : bool
        If True, clip intensity ratios to valid range (0.1, 10) to avoid log errors
    
    Returns
    -------
    dict with keys 'hbo', 'hbr', 'hbt' containing concentration changes (μM)
    """
    # Ensure both signals have same shape
    if green_signal.shape != red_signal.shape:
        raise ValueError("Green and red signals must have same shape")
    
    # Auto-detect format and reshape to (T,H,W)
    # if green_signal.ndim == 3:
    #     if green_signal.shape[0] > max(green_signal.shape[1], green_signal.shape[2]):
    #         green_data = green_signal.astype(np.float32)
    #         red_data = red_signal.astype(np.float32)
    #         transposed = False
    #     else:
    #         green_data = np.transpose(green_signal, (2, 0, 1)).astype(np.float32)
    #         red_data = np.transpose(red_signal, (2, 0, 1)).astype(np.float32)
    #         transposed = True
    # else:
    #     raise ValueError("Input must be 3D arrays")
    
    # Input data should be in format (T, H, W)
    if green_signal.ndim != 3:
        raise ValueError("Input must be 3D arrays")
    else:
        green_data = green_signal.astype(np.float32)
        red_data = red_signal.astype(np.float32)
        transposed = False
    
    T, H, W = green_data.shape
    
    # Calculate baselines
    green_baseline = np.mean(green_data[baseline_frames], axis=0)
    red_baseline = np.mean(red_data[baseline_frames], axis=0)
    
    # Handle zero/negative baselines - set to small positive value
    #min_valid = 1e-6
    #green_baseline[green_baseline <= 0] = min_valid
    #red_baseline[red_baseline <= 0] = min_valid
    
    # Also handle any zero/negative values in data
    #green_data = np.maximum(green_data, min_valid)
    #red_data = np.maximum(red_data, min_valid)
    
    # Calculate intensity ratios
    green_ratio = green_data / green_baseline
    red_ratio = red_data / red_baseline
    
    # Optional: Clip ratios to reasonable range to avoid extreme log values
    if clip_ratio:
        ratio_min, ratio_max = 0.1, 10.0  # Allows ±90% change
        green_ratio = np.clip(green_ratio, ratio_min, ratio_max)
        red_ratio = np.clip(red_ratio, ratio_min, ratio_max)
        
        # Count clipped values for warning
        n_clipped_green = np.sum((green_data / green_baseline < ratio_min) | 
                                 (green_data / green_baseline > ratio_max))
        n_clipped_red = np.sum((red_data / red_baseline < ratio_min) | 
                               (red_data / red_baseline > ratio_max))
        
        if n_clipped_green > 0:
            print(f"  Note: Clipped {n_clipped_green} extreme green values ({100*n_clipped_green/(T*H*W):.2f}%)")
        if n_clipped_red > 0:
            print(f"  Note: Clipped {n_clipped_red} extreme red values ({100*n_clipped_red/(T*H*W):.2f}%)")
    
    # Convert to optical density changes
    # ΔOD = -ln(I/I0) where I0 is baseline
    with np.errstate(invalid='ignore', divide='ignore'):  # Suppress warnings, we handle NaNs below
        green_od = -np.log(green_ratio)
        red_od = -np.log(red_ratio)
    
    # Replace any remaining NaN/inf with 0
    green_od = np.nan_to_num(green_od, nan=0.0, posinf=0.0, neginf=0.0)
    red_od = np.nan_to_num(red_od, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Get extinction coefficients
    eps_hbo_green = get_extinction_at_wavelength(green_wavelength, 'hbo')
    eps_hbr_green = get_extinction_at_wavelength(green_wavelength, 'hbr')
    eps_hbo_red = get_extinction_at_wavelength(red_wavelength, 'hbo')
    eps_hbr_red = get_extinction_at_wavelength(red_wavelength, 'hbr')
    
    print(f"Extinction coefficients:")
    print(f"  Green ({green_wavelength}nm): HbO={eps_hbo_green:.0f}, HbR={eps_hbr_green:.0f}")
    print(f"  Red ({red_wavelength}nm): HbO={eps_hbo_red:.0f}, HbR={eps_hbr_red:.0f}")
    
    # Set up extinction matrix and invert
    # [ΔOD_red]   = [εHbR_red  εHbO_red ] [ΔcHbR]
    # [ΔOD_green]   [εHbR_green εHbO_green] [ΔcHbO]
    extinction_matrix = np.array([
        [eps_hbr_red * pathlength_red, eps_hbo_red * pathlength_red],
        [eps_hbr_green * pathlength_green, eps_hbo_green * pathlength_green]
    ])
    
    # Check matrix condition
    condition_num = np.linalg.cond(extinction_matrix)
    if condition_num > 100:
        warnings.warn(f"Extinction matrix is poorly conditioned (cond={condition_num:.1f}). "
                     f"Results may be unreliable.")
    
    extinction_inv = np.linalg.inv(extinction_matrix)
    
    # Apply inversion pixel by pixel
    hbr_conc = np.zeros((T, H, W), dtype=np.float32)
    hbo_conc = np.zeros((T, H, W), dtype=np.float32)
    
    print(f"  Converting optical density to hemoglobin concentrations...")
    for h in range(H):
        if h % 100 == 0:
            print(f"    Processing row {h}/{H}")
        for w in range(W):
            # Stack optical densities: [red_OD, green_OD]
            od_stack = np.column_stack([
                red_od[:, h, w],
                green_od[:, h, w]
            ])  # Shape: (T, 2)
            
            # Apply inverse: (2,2) @ (T,2).T -> (2,T) -> (T,2).T
            conc_changes = (extinction_inv @ od_stack.T).T  # Shape: (T, 2)
            
            hbr_conc[:, h, w] = conc_changes[:, 0] * 1e6  # Convert to μM
            hbo_conc[:, h, w] = conc_changes[:, 1] * 1e6  # Convert to μM
    
    hbt_conc = hbr_conc + hbo_conc
    
    # Check for extreme values and report
    for name, data in [('HbR', hbr_conc), ('HbO', hbo_conc), ('HbT', hbt_conc)]:
        valid_data = data[np.isfinite(data)]
        if len(valid_data) > 0:
            p1, p99 = np.percentile(valid_data, [1, 99])
            print(f"  {name} range (1-99 percentile): [{p1:.2f}, {p99:.2f}] μM")
    
    # Return in original format
    results = {
        'hbr': hbr_conc,
        'hbo': hbo_conc, 
        'hbt': hbt_conc
    }
    
    if transposed:
        for key in results:
            results[key] = np.transpose(results[key], (1, 2, 0))
    
    return results

def calculate_cmro2(hbr_conc, hbt_conc, 
                   baseline_hbt=100e-6,  # 100 μM baseline HbT
                   baseline_so2=0.6,     # 60% oxygen saturation  
                   grubb_exponent=0.38,
                   vascular_weights=(1.0, 1.0)):  # (gR, gT)
    """
    Calculate cerebral metabolic rate of oxygen consumption (CMRO2) changes.
    Based on the model from Bauer et al. 2020.
    
    Parameters
    ----------
    hbr_conc, hbt_conc : array
        HbR and HbT concentration changes (μM)
    baseline_hbt : float
        Baseline total hemoglobin concentration (M)
    baseline_so2 : float
        Baseline oxygen saturation (fraction)
    grubb_exponent : float
        Grubb's law exponent relating CBV to CBF
    vascular_weights : tuple
        (gR, gT) vascular weighting constants
        
    Returns
    -------
    dict with keys 'cbf', 'cmro2' containing relative changes
    """
    gR, gT = vascular_weights
    
    # Calculate baseline HbR concentration
    baseline_hbr = (1 - baseline_so2) * baseline_hbt
    
    # Convert concentration changes to fractional changes
    delta_hbt_frac = hbt_conc * 1e-6 / baseline_hbt  # Convert μM to M
    delta_hbr_frac = hbr_conc * 1e-6 / baseline_hbr
    
    # Clip extreme fractional changes to avoid numerical issues
    # Don't allow HbT to drop below -90% (would be negative blood volume)
    delta_hbt_frac = np.maximum(delta_hbt_frac, -0.9)
    delta_hbr_frac = np.maximum(delta_hbr_frac, -0.9)
    
    # Calculate cerebral blood flow changes using Grubb's law
    # CBF/CBF0 = (CBV/CBV0)^(1/α) ≈ (1 + ΔHbT/HbT0)^(1/α)
    with np.errstate(invalid='ignore', divide='ignore'):
        cbf_ratio = (1 + delta_hbt_frac) ** (1 / grubb_exponent)
    
    # Handle any invalid values
    cbf_ratio = np.nan_to_num(cbf_ratio, nan=1.0, posinf=1.0, neginf=1.0)
    
    # Calculate CMRO2 changes
    # CMRO2/CMRO20 = CBF * (1 + gR*ΔHbR/HbR0) / (1 + gT*ΔHbT/HbT0)
    numerator = 1 + gR * delta_hbr_frac
    denominator = 1 + gT * delta_hbt_frac
    
    # Avoid division by zero
    denominator = np.maximum(denominator, 0.1)
    
    with np.errstate(invalid='ignore', divide='ignore'):
        cmro2_ratio = cbf_ratio * numerator / denominator
    
    # Handle invalid values
    cmro2_ratio = np.nan_to_num(cmro2_ratio, nan=1.0, posinf=1.0, neginf=1.0)
    
    # Convert to relative changes (ΔF/F format)
    cbf_change = cbf_ratio - 1
    cmro2_change = cmro2_ratio - 1
    
    # Clip extreme values to reasonable physiological range
    cbf_change = np.clip(cbf_change, -0.9, 5.0)     # -90% to +500%
    cmro2_change = np.clip(cmro2_change, -0.9, 5.0)  # -90% to +500%
    
    return {
        'cbf': cbf_change,
        'cmro2': cmro2_change
    }

# -----------------------
# Integration with existing pipeline
# -----------------------
def extract_hemodynamic_signals(red_corrected, green_corrected, 
                               method='dual_wavelength',
                               baseline_frames=slice(0, 100),
                               **kwargs):
    """
    Extract hemodynamic signals from red and green channels.
    
    Parameters
    ----------
    red_corrected, green_corrected : array
        Motion-corrected raw intensity data
    method : str
        'dual_wavelength': full HbO/HbR separation
        'single_wavelength': simplified HbT only from green
    baseline_frames : slice
        Frames for baseline calculation
    **kwargs : dict
        Additional parameters for conversion functions
        
    Returns
    -------
    dict containing hemodynamic parameters
    """
    if method == 'dual_wavelength':
        # Full hemoglobin decomposition
        hb_concentrations = convert_to_hemoglobin_concentrations(
            green_corrected, red_corrected,
            baseline_frames=baseline_frames,
            **kwargs
        )
        
        # Calculate metabolic parameters
        metabolism = calculate_cmro2(
            hb_concentrations['hbr'], 
            hb_concentrations['hbt'],
            **{k: v for k, v in kwargs.items() 
               if k in ['baseline_hbt', 'baseline_so2', 'grubb_exponent', 'vascular_weights']}
        )
        
        results = {**hb_concentrations, **metabolism}
        
    elif method == 'single_wavelength':
        # Simplified HbT from green only
        hbt = convert_to_hbt_concentration(
            green_corrected, baseline_frames=baseline_frames
        )
        results = {'hbt': hbt}
        
    else:
        raise ValueError("method must be 'dual_wavelength' or 'single_wavelength'")
    
    return results