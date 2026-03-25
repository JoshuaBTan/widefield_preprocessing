# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 17:13:18 2025

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#-----------------------
# Correct calcium signal using hemodynamics
# ----------------------
def correct_hemodynamic_artifacts(blue_dff, green_dff, method='regression', 
                                 fit_intercept=True, alpha_global=None, qc=True):
    """
    Remove hemodynamic contamination from calcium-sensitive (blue) signal using
    calcium-insensitive (green) signal.
    
    Parameters
    ----------
    blue_dff : array
        Blue channel ΔF/F data (T,H,W) or (nTrials,T,H,W)
    green_dff : array  
        Green channel ΔF/F data (same shape as blue_dff)
    method : str
        'regression' - fit blue = α*green + β per pixel (recommended)
        'global_regression' - fit single α,β for entire FOV
        'simple_subtraction' - blue - α*green (α provided or estimated)
    fit_intercept : bool
        Whether to fit intercept term in regression (recommended: True)
    alpha_global : float, optional
        Fixed scaling factor if using 'simple_subtraction'
    qc : bool
        Show QC plots
        
    Returns
    -------
    corrected_blue : array
        Hemodynamic-corrected blue signal (same shape as input)
    alpha_map : array
        Scaling factors used (H,W) for pixel-wise, scalar for global
    """
    
    # Debug info
    print(f"Input blue_dff shape: {blue_dff.shape}")
    print(f"Input green_dff shape: {green_dff.shape}")
    print(f"Blue dtype: {blue_dff.dtype}, Green dtype: {green_dff.dtype}")
    
    if blue_dff.shape != green_dff.shape:
        raise ValueError(f"Blue and green arrays must have same shape. Got blue: {blue_dff.shape}, green: {green_dff.shape}")
    
    if blue_dff.ndim not in [3, 4]:
        raise ValueError(f"Input arrays must be 3D (T,H,W) or 4D (nTrials,T,H,W). Got {blue_dff.ndim}D")
    
    single_trial = False
    if blue_dff.ndim == 3:
        single_trial = True
        blue_dff = blue_dff[np.newaxis, ...]
        green_dff = green_dff[np.newaxis, ...]
    
    nTrials, T, H, W = blue_dff.shape
    print(f"Processing shape: {nTrials} trials, {T} timepoints, {H}x{W} pixels")
    corrected = np.empty_like(blue_dff, dtype=np.float32)
    
    if method == 'regression':
        # Fit regression per pixel: blue = α*green + β + noise
        alpha_map = np.zeros((H, W), dtype=np.float32)
        beta_map = np.zeros((H, W), dtype=np.float32)
        r2_map = np.zeros((H, W), dtype=np.float32)
        
        for h in range(H):
            for w in range(W):
                # Collect all time points across trials for this pixel
                blue_pixel = blue_dff[:, :, h, w].flatten()  # (nTrials*T,)
                green_pixel = green_dff[:, :, h, w].flatten()
                
                # Remove any NaN/inf values
                valid = np.isfinite(blue_pixel) & np.isfinite(green_pixel)
                if np.sum(valid) < 10:  # Need minimum points for regression
                    corrected[:, :, h, w] = blue_dff[:, :, h, w]  # No correction
                    continue
                
                # Fit regression
                reg = LinearRegression(fit_intercept=fit_intercept)
                X = green_pixel[valid].reshape(-1, 1)
                y = blue_pixel[valid]
                
                if len(np.unique(X)) < 2:  # No variance in green signal
                    corrected[:, :, h, w] = blue_dff[:, :, h, w]
                    continue
                    
                reg.fit(X, y)
                
                alpha_map[h, w] = reg.coef_[0]
                if fit_intercept:
                    beta_map[h, w] = reg.intercept_
                
                # Calculate R²
                y_pred = reg.predict(X)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2_map[h, w] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # Apply correction: corrected = blue - α*green - β
                for n in range(nTrials):
                    corrected[n, :, h, w] = (blue_dff[n, :, h, w] - 
                                           alpha_map[h, w] * green_dff[n, :, h, w] -
                                           (beta_map[h, w] if fit_intercept else 0))
        
        correction_params = {'alpha_map': alpha_map, 'beta_map': beta_map, 'r2_map': r2_map}
        
    elif method == 'global_regression':
        # Single regression for entire FOV
        blue_flat = blue_dff.flatten()
        green_flat = green_dff.flatten()
        
        valid = np.isfinite(blue_flat) & np.isfinite(green_flat)
        
        reg = LinearRegression(fit_intercept=fit_intercept)
        X = green_flat[valid].reshape(-1, 1)
        y = blue_flat[valid]
        reg.fit(X, y)
        
        alpha_global = reg.coef_[0]
        beta_global = reg.intercept_ if fit_intercept else 0
        
        # Apply correction to all pixels
        corrected = blue_dff - alpha_global * green_dff - beta_global
        correction_params = {'alpha': alpha_global, 'beta': beta_global}
        
    elif method == 'simple_subtraction':
        # Simple subtraction with fixed or estimated α
        if alpha_global is None:
            # Estimate global α as correlation-based scaling
            blue_flat = blue_dff.flatten()
            green_flat = green_dff.flatten()
            valid = np.isfinite(blue_flat) & np.isfinite(green_flat)
            
            if np.sum(valid) > 0:
                alpha_global = np.corrcoef(blue_flat[valid], green_flat[valid])[0, 1] * \
                              (np.std(blue_flat[valid]) / np.std(green_flat[valid]))
            else:
                alpha_global = 0.1  # fallback
        
        corrected = blue_dff - alpha_global * green_dff
        correction_params = {'alpha': alpha_global}
        
    else:
        raise ValueError("method must be 'regression', 'global_regression', or 'simple_subtraction'")
    
    if qc:
        _qc_hemodynamic_correction(blue_dff, green_dff, corrected, correction_params, method)
 
    if single_trial:
        corrected = corrected[0]
        
    return corrected, correction_params


def _qc_hemodynamic_correction(blue_dff, green_dff, corrected, params, method):
    """QC plots for hemodynamic correction"""
    
    # Use first trial if multi-trial
    if blue_dff.ndim == 4:
        blue_show = blue_dff[0]
        green_show = green_dff[0] 
        corr_show = corrected[0]
    else:
        blue_show = blue_dff
        green_show = green_dff
        corr_show = corrected
    
    print(f"QC shapes - Blue: {blue_show.shape}, Green: {green_show.shape}, Corrected: {corr_show.shape}")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Mean images
    axes[0,0].imshow(np.mean(blue_show, axis=0), cmap='RdBu_r')
    axes[0,0].set_title('Blue ΔF/F (mean)')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(np.mean(green_show, axis=0), cmap='RdBu_r') 
    axes[0,1].set_title('Green ΔF/F (mean)')
    axes[0,1].axis('off')
    
    axes[0,2].imshow(np.mean(corr_show, axis=0), cmap='RdBu_r')
    axes[0,2].set_title('Corrected Blue (mean)')
    axes[0,2].axis('off')
    
    # Row 2: Method-specific plots
    if method == 'regression':
        # Show alpha map (scaling factors)
        im1 = axes[1,0].imshow(params['alpha_map'], cmap='viridis')
        axes[1,0].set_title('α (scaling) map')
        axes[1,0].axis('off')
        plt.colorbar(im1, ax=axes[1,0])
        
        # Show R² map (fit quality)
        im2 = axes[1,1].imshow(params['r2_map'], cmap='viridis', vmin=0, vmax=1)
        axes[1,1].set_title('R² (fit quality) map')
        axes[1,1].axis('off') 
        plt.colorbar(im2, ax=axes[1,1])
        
        # Sample pixel comparison
        h, w = blue_show.shape[1]//2, blue_show.shape[2]//2  # center pixel
        axes[1,2].plot(blue_show[:, h, w], label='Blue original', alpha=0.7)
        axes[1,2].plot(green_show[:, h, w], label='Green', alpha=0.7)
        axes[1,2].plot(corr_show[:, h, w], label='Blue corrected', alpha=0.7)
        axes[1,2].set_title(f'Pixel ({h},{w}) timecourses')
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)
        
    else:
        # For global methods, show scatter plot and timecourses
        # Subsample for scatter plot (too many points otherwise)
        idx = np.random.choice(blue_show.size, min(10000, blue_show.size), replace=False)
        blue_sub = blue_show.flatten()[idx]
        green_sub = green_show.flatten()[idx]
        
        axes[1,0].scatter(green_sub, blue_sub, alpha=0.3, s=1)
        axes[1,0].set_xlabel('Green ΔF/F')
        axes[1,0].set_ylabel('Blue ΔF/F') 
        axes[1,0].set_title('Blue vs Green correlation')
        axes[1,0].grid(True, alpha=0.3)
        
        # Sample timecourses  
        h, w = blue_show.shape[1]//2, blue_show.shape[2]//2
        axes[1,1].plot(blue_show[:, h, w], label='Blue original')
        axes[1,1].plot(green_show[:, h, w], label='Green') 
        axes[1,1].plot(corr_show[:, h, w], label='Blue corrected')
        axes[1,1].set_title(f'Center pixel timecourses')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # Correction parameters
        if 'alpha' in params:
            axes[1,2].text(0.1, 0.5, f"α = {params['alpha']:.3f}", fontsize=14)
            if 'beta' in params:
                axes[1,2].text(0.1, 0.3, f"β = {params['beta']:.3f}", fontsize=14)
        axes[1,2].set_title('Correction parameters')
        axes[1,2].axis('off')
    
    plt.tight_layout()
    plt.show()