# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 16:29:58 2025

@author: User
"""

# -----------------------
# Visualization Functions
# -----------------------
def plot_signal_quality_overview(quality_results, signal_type='calcium'):
    """
    Create overview plots of signal quality across all ROIs.
    """
    metrics = quality_results['individual_metrics']
    problematic = quality_results['problematic_rois']
    
    if not metrics:
        print("No metrics to plot!")
        return
    
    # Key metrics to plot
    key_metrics = ['snr', 'cv', 'drift_r2', 'outlier_fraction']
    if 'high_freq_fraction' in list(metrics.values())[0]:
        key_metrics.append('high_freq_fraction')
    
    # Filter to available metrics
    key_metrics = [m for m in key_metrics if m in list(metrics.values())[0]]
    
    n_metrics = len(key_metrics)
    fig, axes = plt.subplots(2, int(np.ceil(n_metrics/2)), figsize=(15, 8))
    if n_metrics <= 2:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    quality_criteria = get_quality_criteria(signal_type)
    
    for i, metric in enumerate(key_metrics):
        values = [metrics[roi][metric] for roi in metrics 
                 if not np.isnan(metrics[roi][metric]) and not np.isinf(metrics[roi][metric])]
        
        if not values:
            continue
            
        # Color code by quality
        colors = ['red' if roi in problematic else 'blue' for roi in metrics]
        
        axes[i].hist(values, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
        
        # Add threshold line if available
        if metric in quality_criteria:
            threshold, direction = quality_criteria[metric]
            axes[i].axvline(threshold, color='red', linestyle='--', 
                           label=f'Threshold: {threshold}')
        
        axes[i].set_xlabel(metric.replace('_', ' ').title())
        axes[i].set_ylabel('Number of ROIs')
        axes[i].set_title(f'{signal_type.upper()} - {metric.replace("_", " ").title()}')
        if metric in quality_criteria:
            axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'{signal_type.upper()} Signal Quality Overview', fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_sample_timecourses(roi_timecourses, roi_labels, quality_results, 
                           n_good=3, n_bad=3, signal_type='calcium', 
                           time_window=None, sampling_rate=None):
    """
    Plot examples of good and bad quality timecourses.
    """
    metrics = quality_results['individual_metrics']
    problematic = quality_results['problematic_rois']
    
    # Convert to consistent format
    if isinstance(roi_timecourses, dict):
        timecourses = {roi: roi_timecourses[roi] for roi in roi_labels if roi in roi_timecourses}
    else:
        timecourses = {roi_labels[i]: roi_timecourses[i] for i in range(len(roi_labels))}
    
    # Identify good and bad ROIs
    good_rois = [roi for roi in timecourses.keys() if roi not in problematic]
    bad_rois = list(problematic.keys())
    
    # Sort by SNR for selection
    if good_rois and 'snr' in metrics.get(good_rois[0], {}):
        good_rois = sorted(good_rois, key=lambda x: metrics[x]['snr'], reverse=True)
    if bad_rois and 'snr' in metrics.get(bad_rois[0], {}):
        bad_rois = sorted(bad_rois, key=lambda x: metrics[x]['snr'])
    
    # Select samples
    good_sample = good_rois[:n_good]
    bad_sample = bad_rois[:n_bad]
    
    fig, axes = plt.subplots(2, max(n_good, n_bad), figsize=(18, 8))
    if max(n_good, n_bad) == 1:
        axes = axes.reshape(-1, 1)
    
    # Time axis
    if time_window is not None:
        time_slice = time_window
    else:
        time_slice = slice(None)
    
    x_axis = np.arange(len(list(timecourses.values())[0]))[time_slice]
    if sampling_rate is not None:
        x_axis = x_axis / sampling_rate
        x_label = 'Time (s)'
    else:
        x_label = 'Frame'
    
    # Plot good examples
    for i, roi in enumerate(good_sample):
        if i >= axes.shape[1]:
            break
        tc = timecourses[roi][time_slice]
        axes[0, i].plot(x_axis, tc, 'b-', alpha=0.8)
        axes[0, i].set_title(f'Good: ROI {roi}\nSNR={metrics[roi]["snr"]:.2f}', color='green')
        axes[0, i].set_ylabel(f'{signal_type.title()} Signal')
        axes[0, i].grid(True, alpha=0.3)
    
    # Plot bad examples  
    for i, roi in enumerate(bad_sample):
        if i >= axes.shape[1]:
            break
        tc = timecourses[roi][time_slice]
        axes[1, i].plot(x_axis, tc, 'r-', alpha=0.8)
        issues = '; '.join(problematic[roi][:2])  # Show first 2 issues
        axes[1, i].set_title(f'Bad: ROI {roi}\n{issues}', color='red')
        axes[1, i].set_ylabel(f'{signal_type.title()} Signal')
        axes[1, i].set_xlabel(x_label)
        axes[1, i].grid(True, alpha=0.3)
    
    # Fill empty subplots
    for i in range(max(len(good_sample), len(bad_sample)), axes.shape[1]):
        axes[0, i].axis('off')
        axes[1, i].axis('off')
    
    plt.suptitle(f'{signal_type.upper()} Signal Quality Examples', fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_spatial_quality_map(quality_results, atlas_labels, brain_mask, 
                            metric='snr', signal_type='calcium'):
    """
    Create spatial map showing signal quality across brain regions.
    """
    metrics = quality_results['individual_metrics']
    
    # Create quality map
    quality_map = np.zeros_like(atlas_labels, dtype=float)
    quality_map[quality_map == 0] = np.nan
    
    for roi_id, roi_metrics in metrics.items():
        if metric in roi_metrics and not np.isnan(roi_metrics[metric]):
            roi_mask = atlas_labels == roi_id
            quality_map[roi_mask] = roi_metrics[metric]
    
    # Apply brain mask
    if brain_mask is not None:
        quality_map[~brain_mask] = np.nan
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Quality map
    im1 = axes[0].imshow(quality_map, cmap='viridis')
    axes[0].set_title(f'{signal_type.upper()} - {metric.replace("_", " ").title()} Map')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], shrink=0.8)
    
    # Atlas outline for reference
    axes[1].imshow(np.mean([quality_map, quality_map, quality_map], axis=0) * 0.3, cmap='gray')
    axes[1].contour(atlas_labels, colors='white', linewidths=0.5, alpha=0.7)
    axes[1].set_title('ROI Boundaries')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    

def plot_rois_on_atlas(atlas, valid_rois, timecourse_dict=None, title="Registered ROIs"):
    plot_map = np.zeros_like(atlas, dtype=float)

    if timecourse_dict is not None:
        for roi in valid_rois:
            if roi in timecourse_dict:
                plot_map[atlas == roi] = timecourse_dict[roi]
        cmap = "viridis"
    else:
        for i, roi in enumerate(valid_rois, start=1):
            plot_map[atlas == roi] = i
        cmap = "tab20"

    plt.figure(figsize=(6,6))
    if np.any(plot_map > 0):
        vmin = np.min(plot_map[plot_map > 0])
        vmax = np.max(plot_map)
        plt.imshow(plot_map, cmap=cmap, interpolation="none", vmin=vmin, vmax=vmax)
        plt.colorbar(label="Mean signal")
    else:
        plt.imshow(plot_map, cmap=cmap, interpolation="none")
        plt.title("No valid ROI signals found")

    plt.title(title)
    plt.axis("off")
    plt.show()


# -----------------------
# Comprehensive QC Function
# -----------------------
def comprehensive_signal_qc(roi_timecourses_dict, roi_labels, atlas_labels=None, 
                           brain_mask=None, sampling_rate=None):
    """
    Run comprehensive quality control on all signal types.
    
    Parameters
    ----------
    roi_timecourses_dict : dict
        Dictionary with signal types as keys, ROI timecourses as values
        e.g., {'calcium': roi_tc, 'hbo': hbo_tc, 'hbr': hbr_tc, ...}
    roi_labels : list
        ROI labels
    atlas_labels : 2D array, optional
        Atlas for spatial quality mapping
    brain_mask : 2D array, optional
        Brain mask
    sampling_rate : float, optional
        Sampling rate in Hz
    
    Returns
    -------
    dict with QC results for each signal type
    """
    print("=== COMPREHENSIVE SIGNAL QUALITY CONTROL ===\n")
    
    all_results = {}
    
    for signal_type, timecourses in roi_timecourses_dict.items():
        print(f"\n{'='*50}")
        print(f"Processing {signal_type.upper()}")
        print('='*50)
        
        # Quality assessment
        quality_results = assess_roi_signal_quality(
            timecourses, roi_labels, sampling_rate, signal_type
        )
        
        if not quality_results:
            continue
            
        all_results[signal_type] = quality_results
        
        # Generate plots
        print(f"\nGenerating quality plots for {signal_type}...")
        
        # Overview plots
        plot_signal_quality_overview(quality_results, signal_type)
        
        # Sample timecourses
        plot_sample_timecourses(
            timecourses, roi_labels, quality_results,
            signal_type=signal_type, sampling_rate=sampling_rate
        )
        
        # Spatial map (if atlas provided)
        if atlas_labels is not None:
            plot_spatial_quality_map(
                quality_results, atlas_labels, brain_mask, 
                metric='snr', signal_type=signal_type
            )
    
    # Cross-signal comparison
    if len(all_results) > 1:
        print(f"\n{'='*50}")
        print("CROSS-SIGNAL QUALITY COMPARISON")
        print('='*50)
        
        comparison_data = []
        for signal_type, results in all_results.items():
            summary = results['quality_summary']
            comparison_data.append([
                signal_type,
                summary['total_rois'],
                summary['good_rois'],
                f"{summary['quality_fraction']*100:.1f}%"
            ])
        
        print(f"{'Signal':<12} {'Total ROIs':<12} {'Good ROIs':<12} {'Quality %':<12}")
        print("-" * 48)
        for row in comparison_data:
            print(f"{row[0]:<12} {row[1]:<12} {row[2]:<12} {row[3]:<12}")
    
    return all_results

# -----------------------
# Integration helper
# -----------------------
def run_full_qc_pipeline(roi_timecourses, hemo_roi_timecourses, 
                        valid_rois, atlas_masked, brain_mask, 
                        sampling_rate=None):
    """
    Helper function to run QC on your extracted signals.
    
    This assumes you have:
    - roi_timecourses: calcium ROI timecourses (dict or array)
    - hemo_roi_timecourses: hemodynamic ROI timecourses (dict with hbo, hbr, etc.)
    - valid_rois: list of valid ROI labels
    - atlas_masked: your registered atlas
    - brain_mask: your brain mask
    """
    
    # Combine all signals
    all_signals = {'calcium': roi_timecourses}
    all_signals.update(hemo_roi_timecourses)
    
    # Run comprehensive QC
    qc_results = comprehensive_signal_qc(
        all_signals, 
        valid_rois, 
        atlas_masked, 
        brain_mask, 
        sampling_rate
    )
    
    return qc_results

# Watch data as movie
class FrameViewer:
    def __init__(self, movie):
        self.movie = movie
        self.num_frames = movie.shape[0]
        self.idx = 0
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(self.movie[self.idx], cmap="gray", vmin=np.min(movie), vmax=np.max(movie))
        self.ax.set_title(f"Frame {self.idx+1}/{self.num_frames}")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        plt.show()

    def on_key(self, event):
        if event.key == "right":
            self.idx = (self.idx + 1) % self.num_frames
        elif event.key == "left":
            self.idx = (self.idx - 1) % self.num_frames
        elif event.key == "q":
            plt.close(self.fig)
            return
        self.update_frame()

    def update_frame(self):
        self.im.set_data(self.movie[self.idx])
        self.ax.set_title(f"Frame {self.idx+1}/{self.num_frames}")
        self.fig.canvas.draw_idle()