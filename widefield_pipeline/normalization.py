# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 16:29:33 2025

@author: User
"""

import numpy as np
from scipy.signal import butter, filtfilt

# -----------------------
# Normalization: dF/F
# -----------------------
# def compute_dff(stack, baseline_frames=None, method='percentile', pct=20):
#     """
#     Compute dF/F per pixel.
#     - stack: (T,H,W) or (nTrials,T,H,W)
#     - baseline_frames: indices list or slice for baseline. If None:
#         - if method=='percentile' use per-pixel percentile (default)
#         - if method=='mean' use mean over time
#     Returns dff (same shape as stack) and F0 used.
#     """
#     single_trial = False
#     if stack.ndim == 3:
#         single_trial = True
#         stack = stack[np.newaxis, ...]
#     nTrials, T, H, W = stack.shape
#     dff = np.empty_like(stack, dtype=np.float32)
#     F0_all = np.empty((nTrials, H, W), dtype=np.float32)

#     for n in range(nTrials):
#         s = stack[n].astype(np.float32)  # (T,H,W)
#         if baseline_frames is not None:
#             F0 = np.mean(s[baseline_frames], axis=0)
#         else:
#             if method == 'percentile':
#                 F0 = np.percentile(s, pct, axis=0)
#             else:
#                 F0 = np.mean(s, axis=0)
#         # avoid zero divide
#         F0[F0 == 0] = np.nanmedian(F0[F0 > 0]) if np.any(F0>0) else 1.0
#         F0_all[n] = F0
#         dff[n] = (s - F0) / F0
#     if single_trial:
#         dff = dff[0]
#         F0_all = F0_all[0]
#     return dff, F0_all

# def compute_dff(stack, baseline_frames=None, method='percentile', pct=20, roll_window=100):
#     """
#     Compute dF/F per pixel.
#     - stack: (T,H,W) or (nTrials,T,H,W)
#     - baseline_frames: indices list or slice for baseline. If None:
#         - if method=='percentile': use per-pixel percentile (pct) over time
#         - if method=='mean':       use mean over time
#         - if method=='roll_mean':  use rolling mean baseline (roll_window frames)
#         - if method=='division':   use per-pixel percentile (pct) over time
#     - pct: percentile for 'percentile' and 'division' methods (default 20)
#     - roll_window: window size (frames) for 'roll_mean' method (default 100)

#     Returns dff (same shape as stack) and F0 used.
#       - For 'roll_mean', F0_all contains the mean F0 across time per pixel.
#     """
#     single_trial = False
#     if stack.ndim == 3:
#         single_trial = True
#         stack = stack[np.newaxis, ...]
#     nTrials, T, H, W = stack.shape
#     dff = np.empty_like(stack, dtype=np.float32)
#     F0_all = np.empty((nTrials, H, W), dtype=np.float32)

#     for n in range(nTrials):
#         s = stack[n].astype(np.float32)  # (T,H,W)

#         if method == 'roll_mean':
#             # Build a rolling mean baseline F0t: shape (T, H, W)
#             # Uses a causal (trailing) window; edges use available frames.
#             s_2d = s.reshape(T, -1)  # (T, H*W)
#             F0t_2d = np.empty_like(s_2d)
#             for t in range(T):
#                 start = max(0, t - roll_window + 1)
#                 F0t_2d[t] = s_2d[start:t + 1].mean(axis=0)
#             F0t = F0t_2d.reshape(T, H, W)

#             # Avoid zero divide per frame
#             for t in range(T):
#                 frame = F0t[t]
#                 zero_mask = frame == 0
#                 if zero_mask.any():
#                     pos_vals = frame[frame > 0]
#                     frame[zero_mask] = np.nanmedian(pos_vals) if pos_vals.size > 0 else 1.0

#             dff[n]   = (s - F0t) / F0t
#             F0_all[n] = F0t.mean(axis=0)  # store mean baseline for inspection

#         else:
#             # Static baseline F0: shape (H, W)
#             if baseline_frames is not None:
#                 F0 = np.mean(s[baseline_frames], axis=0)
#             else:
#                 if method == 'percentile':
#                     F0 = np.percentile(s, pct, axis=0)
#                 elif method == 'mean':
#                     F0 = np.mean(s, axis=0)
#                 elif method == 'division':
#                     F0 = np.percentile(s, pct, axis=0)
#                 else:
#                     raise ValueError(f"Unknown method '{method}'. "
#                                      f"Choose from: 'percentile', 'mean', 'roll_mean', 'division'.")

#             # Avoid zero divide
#             zero_mask = F0 == 0
#             if zero_mask.any():
#                 pos_vals = F0[F0 > 0]
#                 F0[zero_mask] = np.nanmedian(pos_vals) if pos_vals.size > 0 else 1.0

#             F0_all[n] = F0

#             if method in ('percentile', 'mean'):
#                 dff[n] = (s - F0) / F0
#             elif method == 'division':
#                 dff[n] = s / F0

#     if single_trial:
#         dff    = dff[0]
#         F0_all = F0_all[0]

#     return dff, F0_all

def compute_dff(stack, baseline_frames=None, method='mean', roll_window=100):
    """
    Compute dF/F per pixel.

    Parameters
    ----------
    stack : np.ndarray
        Shape (T, H, W) or (nTrials, T, H, W)
    baseline_frames : slice or list, optional
        Frames to use as baseline for 'mean' and 'divide' methods.
        If None, uses all frames.
    method : str
        'mean'      : F0 = mean over baseline_frames. dF/F = (s - F0) / F0
        'divide'    : F0 = mean over baseline_frames. dF/F = s / F0
        'roll_mean' : F0t = causal rolling mean (roll_window frames). dF/F = (s - F0t) / F0t
    roll_window : int
        Window size (frames) for 'roll_mean' method (default 100).

    Returns
    -------
    dff    : same shape as stack
    F0_all : (H, W) or (nTrials, H, W)
             For 'roll_mean', this is the time-averaged rolling baseline.
    """
    single_trial = False
    if stack.ndim == 3:
        single_trial = True
        stack = stack[np.newaxis, ...]
    nTrials, T, H, W = stack.shape

    dff    = np.empty_like(stack, dtype=np.float32)
    F0_all = np.empty((nTrials, H, W), dtype=np.float32)

    for n in range(nTrials):
        s = stack[n].astype(np.float32)  # (T, H, W)

        if method == 'roll_mean':
            s_2d   = s.reshape(T, -1)
            F0t_2d = np.empty_like(s_2d)
            for t in range(T):
                start       = max(0, t - roll_window + 1)
                F0t_2d[t]   = s_2d[start:t + 1].mean(axis=0)
            F0t = F0t_2d.reshape(T, H, W)

            for t in range(T):
                frame     = F0t[t]
                zero_mask = frame == 0
                if zero_mask.any():
                    pos_vals         = frame[frame > 0]
                    frame[zero_mask] = np.nanmedian(pos_vals) if pos_vals.size > 0 else 1.0

            dff[n]    = (s - F0t) / F0t
            F0_all[n] = F0t.mean(axis=0)

        elif method in ('mean', 'divide'):
            baseline = s[baseline_frames] if baseline_frames is not None else s
            F0       = baseline.mean(axis=0)  # (H, W)

            zero_mask = F0 == 0
            if zero_mask.any():
                pos_vals      = F0[F0 > 0]
                F0[zero_mask] = np.nanmedian(pos_vals) if pos_vals.size > 0 else 1.0

            F0_all[n] = F0
            dff[n]    = (s - F0) / F0 if method == 'mean' else s / F0

        else:
            raise ValueError(f"Unknown method '{method}'. Choose from: 'mean', 'divide', 'roll_mean'.")

    if single_trial:
        dff    = dff[0]
        F0_all = F0_all[0]

    return dff, F0_all

# -----------------------
# Detrend
# -----------------------
def detrend_quadratic(data):
    """
    data: (T,) or (T, N)
    """
    T = data.shape[0]
    t = np.linspace(-1, 1, T)

    X = np.vstack([
        np.ones(T),
        t,
        t**2
    ]).T  # (T, 3)

    beta = np.linalg.lstsq(X, data, rcond=None)[0]
    trend = X @ beta

    return data - trend

# ------------------------
# Butterworth filter 
# ------------------------
def butter_filter(data, fs, lowcut=None, highcut=None, order=3):
    """
    data: (T,) or (T, N)
    fs: sampling rate (Hz)
    lowcut: Hz (high-pass)
    highcut: Hz (low-pass)
    
    Recommended
    lowcut = 0.01 (remove drift)
    highcut = 3.0 (high-frequency noise)
    """

    nyq = fs / 2
    btype = None
    Wn = []

    if lowcut is not None:
        Wn.append(lowcut / nyq)
    if highcut is not None:
        Wn.append(highcut / nyq)

    if lowcut and highcut:
        btype = 'bandpass'
    elif lowcut:
        btype = 'highpass'
        Wn = Wn[0]
    elif highcut:
        btype = 'lowpass'
        Wn = Wn[0]
    else:
        return data  # no filtering

    b, a = butter(order, Wn, btype=btype)

    return filtfilt(b, a, data, axis=0)
