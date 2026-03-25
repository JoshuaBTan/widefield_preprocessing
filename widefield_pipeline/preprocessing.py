# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 16:29:25 2025

@author: User
"""

import numpy as np
from skimage.transform import resize
from skimage.registration import phase_cross_correlation
from skimage.morphology import disk
from scipy.ndimage import fourier_shift, median_filter, binary_closing
import warnings

# -----------------------
# Downsampling utilities
# -----------------------
def downsample_stack(stack, scale=None, target_shape=None, interp_order=1):
    """
    Downsample a stack (works for shapes (T,H,W) or (nTrials, T, H, W)).
    - scale: float or (scale_h, scale_w). If float, applied to both dims.
    - target_shape: (H_new, W_new) overrides scale (preferred).
    - interp_order: interpolation order for images (1=linear). Use 0 for labels.
    Returns downsampled stack of same leading dims but new H/W.
    """
    if target_shape is None:
        if scale is None:
            raise ValueError("Provide either scale or target_shape")
        if isinstance(scale, (int,float)):
            scale_h = scale_w = float(scale)
        else:
            scale_h, scale_w = scale
    else:
        target_shape = tuple(target_shape)
        scale_h = scale_w = None

    def _resize_frame(frame, out_shape):
        # preserve_range to avoid scaling pixel values
        return resize(frame, out_shape, order=interp_order, preserve_range=True, anti_aliasing=True)

    if stack.ndim == 3:
        T, H, W = stack.shape
        if target_shape is None:
            H2 = int(round(H * scale_h))
            W2 = int(round(W * scale_w))
        else:
            H2, W2 = target_shape
        out = np.empty((T, H2, W2), dtype=stack.dtype)
        for t in range(T):
            out[t] = _resize_frame(stack[t], (H2, W2))
        return out

    elif stack.ndim == 4:
        N, T, H, W = stack.shape
        if target_shape is None:
            H2 = int(round(H * scale_h))
            W2 = int(round(W * scale_w))
        else:
            H2, W2 = target_shape
        out = np.empty((N, T, H2, W2), dtype=stack.dtype)
        for n in range(N):
            for t in range(T):
                out[n, t] = _resize_frame(stack[n, t], (H2, W2))
        return out
    else:
        raise ValueError("stack must be 3D or 4D (T,H,W) or (nTrials,T,H,W)")
        

def downsample_stack_nanmean(
    stack,
    scale=None,
    target_shape=None,
    interp_order=1
):
    """
    Downsample a stack (T,H,W) or (N,T,H,W) with NaN-aware handling.

    - If scale is integer-like → NaN-safe block averaging
    - If target_shape is provided → NaN-weighted resize
    - interp_order=0 recommended for labels
    """

    if target_shape is None:
        if scale is None:
            raise ValueError("Provide either scale or target_shape")
        if isinstance(scale, (int, float)):
            scale_h = scale_w = float(scale)
        else:
            scale_h, scale_w = map(float, scale)
    else:
        target_shape = tuple(target_shape)

    def _is_int_scale(s):
        return np.isclose(s, round(s))

    def _block_nanmean(frame, fy, fx):
        H, W = frame.shape
        H2 = H // fy
        W2 = W // fx
        frame = frame[:H2 * fy, :W2 * fx]
        frame = frame.reshape(H2, fy, W2, fx)
        return np.nanmean(frame, axis=(1, 3))

    def _resize_nan_weighted(frame, out_shape):
        valid = np.isfinite(frame).astype(float)
        frame_filled = np.nan_to_num(frame, nan=0.0)

        data_resized = resize(
            frame_filled,
            out_shape,
            order=interp_order,
            preserve_range=True,
            anti_aliasing=(interp_order > 0)
        )

        weight_resized = resize(
            valid,
            out_shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False
        )

        with np.errstate(invalid="ignore", divide="ignore"):
            out = data_resized / weight_resized
            out[weight_resized == 0] = np.nan

        return out

    # Determine output spatial size
    if stack.ndim == 3:
        T, H, W = stack.shape
    elif stack.ndim == 4:
        N, T, H, W = stack.shape
    else:
        raise ValueError("stack must be 3D or 4D")

    if target_shape is None:
        H2 = int(round(H * scale_h))
        W2 = int(round(W * scale_w))
    else:
        H2, W2 = target_shape

    # Decide strategy
    use_block = (
        target_shape is None and
        _is_int_scale(1 / scale_h) and
        _is_int_scale(1 / scale_w)
    )

    fy = int(round(1 / scale_h)) if use_block else None
    fx = int(round(1 / scale_w)) if use_block else None

    def _process_frame(frame):
        if interp_order == 0:
            # Labels: nearest neighbor resize only
            return resize(
                frame,
                (H2, W2),
                order=0,
                preserve_range=True,
                anti_aliasing=False
            ).astype(frame.dtype)

        if use_block:
            return _block_nanmean(frame, fy, fx)
        else:
            return _resize_nan_weighted(frame, (H2, W2))

    # Allocate output
    if stack.ndim == 3:
        out = np.empty((T, H2, W2), dtype=np.float32)
        for t in range(T):
            out[t] = _process_frame(stack[t])
        return out

    else:
        out = np.empty((stack.shape[0], T, H2, W2), dtype=np.float32)
        for n in range(stack.shape[0]):
            for t in range(T):
                out[n, t] = _process_frame(stack[n, t])
        return out


        
        
# -----------------------
# Channel separation
# -----------------------
def separate_channels_from_interleaved(stack, frames_per_cycle=3, order=('blue','green','red')):
    """
    Split an interleaved stack into channel stacks.
    - stack: (T,H,W) or (nTrials,T,H,W)
    - frames_per_cycle: e.g. 3 for RGB-like interleaving
    - order: tuple mapping frame index 0.. to channel names (used for return order)
    Returns dict: {'blue': (Tch,H,W) , ... } or for multi-trial {'blue': (nTrials,Tch,H,W)}
    """
    single_trial = False
    if stack.ndim == 3:
        single_trial = True
        stack = stack[np.newaxis, ...]  # shape (1,T,H,W)

    nTrials, T, H, W = stack.shape
    if T % frames_per_cycle != 0:
        warnings.warn("Total frames not divisible by frames_per_cycle; last incomplete cycle will be ignored.")
    n_cycles = T // frames_per_cycle
    T_ch = n_cycles

    ch_dict = {ch: np.empty((nTrials, T_ch, H, W), dtype=stack.dtype) for ch in order}

    for c in range(frames_per_cycle):
        ch_name = order[c]
        # take frames c, c+frames_per_cycle, ...
        idxs = np.arange(c, c + frames_per_cycle * n_cycles, frames_per_cycle)
        ch_dict[ch_name][:, :, :, :] = stack[:, idxs, :, :]

    if single_trial:
        # squeeze trial dimension
        for k in ch_dict:
            ch_dict[k] = ch_dict[k].squeeze(axis=0)  # (T_ch, H, W)
    return ch_dict


# -----------------------
# Motion correction (rigid, FFT)
# -----------------------
def motion_correction_rigid(stack, reference=None, upsample_factor=10, return_shifts=True):
    """
    Rigid motion correction using FFT-based subpixel shifts.
    - stack: (T,H,W) or (nTrials,T,H,W)
    - reference: 2D array to align to (H,W). If None use median image across first trial/time.
    - upsample_factor: passed to phase_cross_correlation for subpixel precision.
    Returns corrected stack (same shape as input) and shifts (list per frame).
    """
    single_trial = False
    if stack.ndim == 3:
        single_trial = True
        stack = stack[np.newaxis, ...]  # (1,T,H,W)

    nTrials, T, H, W = stack.shape
    corrected = np.empty_like(stack, dtype=np.float32)
    all_shifts = []

    # choose reference
    if reference is None:
        reference = np.median(stack[0].astype(np.float32), axis=0)

    for n in range(nTrials):
        shifts = np.zeros((T, 2), dtype=np.float32)
        for t in range(T):
            cur = stack[n, t].astype(np.float32)
            shift, error, phasediff = phase_cross_correlation(reference, cur, upsample_factor=upsample_factor)
            shifts[t] = shift  # shift is (shift_y, shift_x)
            # apply shift in Fourier domain for subpixel
            corrected_frame = np.fft.ifftn(fourier_shift(np.fft.fftn(cur), shift)).real
            corrected[n, t] = corrected_frame
        all_shifts.append(shifts)
    all_shifts = np.array(all_shifts)  # (nTrials, T, 2)

    if single_trial:
        corrected = corrected[0]
        all_shifts = all_shifts[0]

    if return_shifts:
        return corrected, all_shifts
    else:
        return corrected

# -----------------------
# IMAGE PADDING
# -----------------------
def pad_to_size(img, target_h, target_w, pad_value=0):
    """
    Pads img to (target_h, target_w) without changing aspect ratio.
    img: (H,W) or (H,W,C)
    """
    h, w = img.shape[:2]

    if h > target_h or w > target_w:
        raise ValueError("Image larger than target size")

    pad_h = target_h - h
    pad_w = target_w - w

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if img.ndim == 2:
        padding = ((pad_top, pad_bottom), (pad_left, pad_right))
    else:
        padding = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))

    return np.pad(img, padding, mode="constant", constant_values=pad_value)


# -----------------------
# Create vasculature mask
# -----------------------
def create_vasculature_mask(ref_img,
                            median_size=125,
                            z_thresh=2.5,
                            closing_radius=2):
    """
    ref_img: 2D image (e.g. middle frame or mean raw image)
    Returns: boolean mask (True = keep pixel)
    """

    # High-pass spatial filtering
    smooth = median_filter(ref_img, size=median_size)
    hp_img = ref_img - smooth

    # Threshold dark pixels (vasculature)
    mu = hp_img.mean()
    sigma = hp_img.std()

    vessel_mask = hp_img < (mu - z_thresh * sigma)

    # Morphological closing (fills gaps in vessels)
    vessel_mask = binary_closing(vessel_mask, structure=disk(closing_radius))

    # Invert: True = valid neural pixels
    brain_mask = ~vessel_mask

    return brain_mask


def create_vasculature_mask_percentile(
    ref_img,
    median_size=11,
    percentile=3,
    closing_radius=1):
    
    smooth = median_filter(ref_img, size=median_size)
    hp = ref_img - smooth

    thresh = np.percentile(hp, percentile)
    vessel_mask = hp < thresh
    vessel_mask = binary_closing(vessel_mask, structure=disk(closing_radius))
    
    return ~vessel_mask



def apply_spatial_mask(data, mask):
    """
    data: (T, H, W)
    mask: (H, W) boolean
    """
    masked = data.copy()
    masked[:, ~mask] = np.nan
    return masked