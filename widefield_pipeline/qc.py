# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 16:29:55 2025

@author: User
"""

def qc_hemodynamic_signals(hemo_signals, sample_roi_mask=None, 
                          time_window=None, title_prefix=""):
    """
    Quality control plots for hemodynamic signals.
    
    Parameters
    ----------
    hemo_signals : dict
        Dictionary containing hemodynamic signals
    sample_roi_mask : 2D bool array, optional
        ROI mask for timecourse plotting
    time_window : slice, optional
        Time window to display
    """
    available_signals = list(hemo_signals.keys())
    n_signals = len(available_signals)
    
    # Determine layout
    if n_signals <= 3:
        ncols = n_signals
        nrows = 2
    else:
        ncols = 3
        nrows = int(np.ceil(n_signals / 3)) + 1
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # Plot mean maps
    for i, signal_name in enumerate(available_signals):
        signal = hemo_signals[signal_name]
        
        # Calculate mean (handle different formats)
        if signal.ndim == 3:
            if signal.shape[0] > max(signal.shape[1], signal.shape[2]):
                mean_map = np.mean(signal, axis=0)
            else:
                mean_map = np.mean(signal, axis=2)
        
        # Choose colormap based on signal type
        if 'hbo' in signal_name.lower():
            cmap = 'Reds'
        elif 'hbr' in signal_name.lower():
            cmap = 'Blues'
        elif 'hbt' in signal_name.lower():
            cmap = 'RdBu_r'
        elif 'cmro2' in signal_name.lower():
            cmap = 'viridis'
        elif 'cbf' in signal_name.lower():
            cmap = 'plasma'
        else:
            cmap = 'RdBu_r'
        
        im = axes[i].imshow(mean_map, cmap=cmap)
        axes[i].set_title(f'{title_prefix}{signal_name.upper()} (mean)')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], shrink=0.6)
    
    # Plot sample timecourses if ROI provided
    if sample_roi_mask is not None and n_signals > 0:
        ax_tc = axes[n_signals] if n_signals < len(axes) else plt.gca()
        
        for signal_name in available_signals:
            signal = hemo_signals[signal_name]
            
            # Extract ROI timecourse
            if signal.ndim == 3:
                if signal.shape[0] > max(signal.shape[1], signal.shape[2]):
                    # (T,H,W)
                    roi_tc = np.mean(signal[:, sample_roi_mask], axis=1)
                else:
                    # (H,W,T)
                    roi_tc = np.mean(signal[sample_roi_mask, :], axis=0)
            
            if time_window is not None:
                roi_tc = roi_tc[time_window]
            
            ax_tc.plot(roi_tc, label=signal_name.upper(), alpha=0.8)
        
        ax_tc.set_title(f'{title_prefix}Sample ROI Timecourses')
        ax_tc.set_xlabel('Frame')
        ax_tc.set_ylabel('Signal')
        ax_tc.legend()
        ax_tc.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_signals + (1 if sample_roi_mask is not None else 0), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# -----------------------
# QC Plotting utilities
# -----------------------
def qc_channel_separation(ch_dict, frame_idx=0, vmin=None, vmax=None):
    """
    Visual check: plot representative frame and mean image for each channel.
    ch_dict expected keys 'blue', 'green', 'red'
    """
    keys = list(ch_dict.keys())
    n = len(keys)
    plt.figure(figsize=(4*n, 6))
    for i,k in enumerate(keys):
        arr = ch_dict[k]
        if arr.ndim == 4:
            # (nTrials,T,H,W) -> show first trial
            img_mean = np.mean(arr[0], axis=0)
            rep = arr[0, frame_idx]
        else:
            # (T,H,W)
            img_mean = np.mean(arr, axis=0)
            rep = arr[frame_idx]
        plt.subplot(2, n, i+1)
        plt.imshow(rep, cmap='gray', vmin=vmin, vmax=vmax)
        plt.title(f"{k} rep frame")
        plt.axis('off')
        plt.subplot(2, n, n + i + 1)
        plt.imshow(img_mean, cmap='gray')
        plt.title(f"{k} mean")
        plt.axis('off')
    plt.show()

def qc_motion_correction(before_stack, after_stack, shifts=None):
    """Show mean before/after and shifts trace if provided."""
    # accept (T,H,W) or (nTrials,T,H,W) ; use first trial if multi
    if before_stack.ndim == 4:
        before = np.mean(before_stack[0], axis=0)
    else:
        before = np.mean(before_stack, axis=0)
    if after_stack.ndim == 4:
        after = np.mean(after_stack[0], axis=0)
    else:
        after = np.mean(after_stack, axis=0)

    plt.figure(figsize=(12,5))
    plt.subplot(1,3,1); plt.imshow(before, cmap='gray'); plt.title('Before (mean)'); plt.axis('off')
    plt.subplot(1,3,2); plt.imshow(after, cmap='gray'); plt.title('After (mean)'); plt.axis('off')
    plt.subplot(1,3,3)
    diff = np.abs(after.astype(np.float32) - before.astype(np.float32))
    plt.imshow(diff, cmap='magma'); plt.title('Abs difference'); plt.axis('off')
    plt.show()

    if shifts is not None:
        # shifts shape (T,2) or (nTrials,T,2)
        if shifts.ndim == 3:
            s = shifts[0]
        else:
            s = shifts
        plt.figure(figsize=(8,3))
        plt.plot(s[:,1], label='x shift'); plt.plot(s[:,0], label='y shift')
        plt.legend(); plt.title('Frame shifts (pixels)'); plt.xlabel('frame')
        plt.show()

def qc_normalization(raw_stack, dff_stack, sample_pixel=(10,10)):
    """
    Show raw vs dF/F mean maps and timecourse for sample pixel.
    - raw_stack and dff_stack shape (T,H,W) or (nTrials,T,H,W)
    """
    if raw_stack.ndim==4:
        raw = np.mean(raw_stack[0], axis=0)
        dff = np.mean(dff_stack[0], axis=0)
    else:
        raw = np.mean(raw_stack, axis=0)
        dff = np.mean(dff_stack, axis=0)

    plt.figure(figsize=(10,4))
    plt.subplot(1,3,1); plt.imshow(raw, cmap='gray'); plt.title('raw mean'); plt.axis('off')
    plt.subplot(1,3,2); plt.imshow(dff, cmap='bwr'); plt.title('dF/F mean'); plt.axis('off')

    # pixel trace
    if raw_stack.ndim==4:
        trace_raw = raw_stack[0,:, sample_pixel[0], sample_pixel[1]]
        trace_dff = dff_stack[0,:, sample_pixel[0], sample_pixel[1]]
    else:
        trace_raw = raw_stack[:, sample_pixel[0], sample_pixel[1]]
        trace_dff = dff_stack[:, sample_pixel[0], sample_pixel[1]]

    plt.subplot(1,3,3)
    plt.plot(trace_raw, label='raw'); plt.plot(trace_dff, label='dF/F')
    plt.legend(); plt.title(f"Pixel {sample_pixel} trace")
    plt.show()

def calculate_signal_metrics(timecourse, sampling_rate=None):
    """
    Calculate comprehensive signal quality metrics for a single timecourse.
    
    Parameters
    ----------
    timecourse : 1D array
        Signal timecourse
    sampling_rate : float, optional
        Sampling rate in Hz (for frequency analysis)
        
    Returns
    -------
    dict with quality metrics
    """
    tc = np.array(timecourse)
    
    # Basic statistics
    mean_val = np.mean(tc)
    std_val = np.std(tc)
    snr = np.abs(mean_val) / std_val if std_val > 0 else 0
    
    # Signal range and variability
    signal_range = np.max(tc) - np.min(tc)
    coefficient_of_variation = std_val / np.abs(mean_val) if mean_val != 0 else np.inf
    
    # Drift assessment (linear trend)
    x = np.arange(len(tc))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, tc)
    drift_magnitude = np.abs(slope * len(tc))  # Total drift over recording
    
    # Stability metrics
    # Check for sudden jumps (difference between consecutive points)
    diff = np.diff(tc)
    max_jump = np.max(np.abs(diff))
    mean_abs_diff = np.mean(np.abs(diff))
    
    # Outlier detection (points beyond 3 std from mean)
    outliers = np.abs(tc - mean_val) > 3 * std_val
    outlier_fraction = np.mean(outliers)
    
    # Frequency content (if sampling rate provided)
    freq_metrics = {}
    if sampling_rate is not None:
        freqs, psd = signal.welch(tc, fs=sampling_rate, nperseg=min(len(tc)//4, 256))
        
        # Find dominant frequency
        dominant_freq_idx = np.argmax(psd[1:]) + 1  # Skip DC component
        dominant_freq = freqs[dominant_freq_idx]
        
        # Power in different bands
        low_freq_power = np.sum(psd[(freqs >= 0.01) & (freqs < 0.1)])  # 0.01-0.1 Hz (drift)
        physio_freq_power = np.sum(psd[(freqs >= 0.1) & (freqs < 2.0)])  # 0.1-2 Hz (physiological)
        high_freq_power = np.sum(psd[freqs >= 2.0])  # >2 Hz (noise/artifacts)
        
        total_power = np.sum(psd[1:])  # Exclude DC
        
        freq_metrics = {
            'dominant_frequency': dominant_freq,
            'low_freq_fraction': low_freq_power / total_power if total_power > 0 else 0,
            'physio_freq_fraction': physio_freq_power / total_power if total_power > 0 else 0,
            'high_freq_fraction': high_freq_power / total_power if total_power > 0 else 0
        }
    
    return {
        'mean': mean_val,
        'std': std_val,
        'snr': snr,
        'range': signal_range,
        'cv': coefficient_of_variation,
        'drift_slope': slope,
        'drift_magnitude': drift_magnitude,
        'drift_r2': r_value**2,
        'max_jump': max_jump,
        'mean_abs_diff': mean_abs_diff,
        'outlier_fraction': outlier_fraction,
        **freq_metrics
    }

def assess_roi_signal_quality(roi_timecourses, roi_labels, sampling_rate=None, 
                             signal_type='calcium'):
    """
    Assess signal quality across all ROIs and flag problematic ones.
    
    Parameters
    ----------
    roi_timecourses : dict or 2D array
        ROI timecourses. If dict: {roi_id: timecourse}, if array: (n_rois, timepoints)
    roi_labels : list
        ROI labels corresponding to timecourses
    sampling_rate : float, optional
        Sampling rate in Hz
    signal_type : str
        Type of signal ('calcium', 'hbo', 'hbr', 'hbt', 'cmro2', etc.)
        
    Returns
    -------
    dict with quality assessment results
    """
    print(f"=== {signal_type.upper()} Signal Quality Assessment ===")
    
    # Convert to consistent format
    if isinstance(roi_timecourses, dict):
        timecourses = [roi_timecourses[roi] for roi in roi_labels]
    else:
        timecourses = [roi_timecourses[i] for i in range(len(roi_labels))]
    
    # Calculate metrics for each ROI
    all_metrics = {}
    for i, (roi_id, tc) in enumerate(zip(roi_labels, timecourses)):
        if len(tc) < 10:  # Skip if too few points
            continue
        all_metrics[roi_id] = calculate_signal_metrics(tc, sampling_rate)
    
    if not all_metrics:
        print("No valid ROIs found!")
        return {}
    
    # Aggregate statistics across ROIs
    metric_names = list(all_metrics[list(all_metrics.keys())[0]].keys())
    aggregated = {}
    
    for metric in metric_names:
        values = [all_metrics[roi][metric] for roi in all_metrics 
                 if not np.isnan(all_metrics[roi][metric]) and not np.isinf(all_metrics[roi][metric])]
        if values:
            aggregated[metric] = {
                'median': np.median(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
    
    # Define quality criteria based on signal type
    quality_criteria = get_quality_criteria(signal_type)
    
    # Flag problematic ROIs
    problematic_rois = {}
    for roi_id, metrics in all_metrics.items():
        issues = []
        
        # Check each criterion
        for criterion, (threshold, direction) in quality_criteria.items():
            if criterion in metrics:
                value = metrics[criterion]
                if direction == 'max' and value > threshold:
                    issues.append(f"{criterion}={value:.3f} > {threshold}")
                elif direction == 'min' and value < threshold:
                    issues.append(f"{criterion}={value:.3f} < {threshold}")
        
        if issues:
            problematic_rois[roi_id] = issues
    
    # Summary statistics
    n_total = len(all_metrics)
    n_problematic = len(problematic_rois)
    quality_fraction = (n_total - n_problematic) / n_total if n_total > 0 else 0
    
    print(f"Total ROIs: {n_total}")
    print(f"Problematic ROIs: {n_problematic} ({100*(1-quality_fraction):.1f}%)")
    print(f"Good quality ROIs: {n_total - n_problematic} ({100*quality_fraction:.1f}%)")
    
    # Show most common issues
    if problematic_rois:
        all_issues = []
        for issues in problematic_rois.values():
            all_issues.extend([issue.split('=')[0] for issue in issues])
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        print("\nMost common issues:")
        for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {issue}: {count} ROIs ({100*count/n_total:.1f}%)")
    
    return {
        'individual_metrics': all_metrics,
        'aggregated_metrics': aggregated,
        'problematic_rois': problematic_rois,
        'quality_summary': {
            'total_rois': n_total,
            'good_rois': n_total - n_problematic,
            'problematic_rois': n_problematic,
            'quality_fraction': quality_fraction
        }
    }

def get_quality_criteria(signal_type):
    """
    Define quality criteria thresholds for different signal types.
    """
    if signal_type.lower() == 'calcium':
        return {
            'snr': (0.5, 'min'),           # Minimum SNR
            'cv': (5.0, 'max'),            # Maximum coefficient of variation
            'outlier_fraction': (0.1, 'max'),  # Max 10% outliers
            'drift_r2': (0.3, 'max'),      # Max R² for linear drift
            'high_freq_fraction': (0.4, 'max'),  # Max high-frequency content
        }
    elif signal_type.lower() in ['hbo', 'hbr', 'hbt']:
        return {
            'snr': (0.3, 'min'),           # Lower SNR acceptable for hemodynamics
            'cv': (10.0, 'max'),           # More variable than calcium
            'outlier_fraction': (0.15, 'max'),  # Slightly more outliers OK
            'drift_r2': (0.4, 'max'),      # Drift less critical
            'high_freq_fraction': (0.5, 'max'),
        }
    elif signal_type.lower() in ['cmro2', 'cbf']:
        return {
            'snr': (0.2, 'min'),           # Even lower SNR for derived metrics
            'cv': (15.0, 'max'),           # Very variable
            'outlier_fraction': (0.2, 'max'),
            'high_freq_fraction': (0.6, 'max'),
        }
    else:
        # Generic criteria
        return {
            'snr': (0.3, 'min'),
            'cv': (8.0, 'max'),
            'outlier_fraction': (0.15, 'max'),
        }
