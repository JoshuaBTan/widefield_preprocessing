# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 16:29:41 2025

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from skimage.transform import warp, SimilarityTransform, resize
from skimage import measure, exposure
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
from skimage.morphology import remove_small_objects, binary_closing, binary_opening, disk
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import gaussian_filter


# ----------------------
# Load in Allen Brain CCF Atlas (Dorsal Top View)
# ----------------------
def load_allen_atlas(mat_file, label_file):
    """Load Allen dorsal atlas and ROI label mapping."""
    atlas_mat = sio.loadmat(mat_file)
    atlas = atlas_mat["data"]

    labels = {}
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    idx = int(parts[0])
                    name = " ".join(parts[1:])
                    labels[idx] = name
                except ValueError:
                    continue
    return atlas, labels

# -----------------------
# Apply transformation to data stacks
# -----------------------
def apply_transform_to_stack(data_stack, tform, output_shape, order=1):
    """
    Apply transformation to entire image stack (e.g., timeseries data)
    
    Parameters
    ----------
    data_stack : 3D array
        (T, H, W) or (H, W, T) image stack
    tform : skimage transform object
        Transformation to apply
    output_shape : tuple
        (H, W) shape of output images
    order : int
        Interpolation order (0=nearest, 1=bilinear, 3=cubic)
        
    Returns
    -------
    transformed_stack : 3D array
        Transformed stack (same time axis as input, new spatial dims)
    """
    
    # Detect format
    if data_stack.shape[0] > max(data_stack.shape[1], data_stack.shape[2]):
        # (T, H, W) format
        T, H, W = data_stack.shape
        time_first = True
    else:
        # (H, W, T) format
        H, W, T = data_stack.shape
        data_stack = np.transpose(data_stack, (2, 0, 1))
        time_first = False
    
    print(f"Transforming stack of {T} frames to shape {output_shape}...")
    
    transformed_stack = np.empty((T, output_shape[0], output_shape[1]), dtype=data_stack.dtype)
    
    for t in range(T):
        if t % 50 == 0:
            print(f"  Frame {t}/{T}")
        transformed_stack[t] = warp(
            data_stack[t],
            inverse_map=tform.inverse,
            output_shape=output_shape,
            order=order,
            preserve_range=True,
            cval=np.nan
        )
    
    # Return in original format
    if not time_first:
        transformed_stack = np.transpose(transformed_stack, (1, 2, 0))
    
    print("Done!")
    return transformed_stack

def apply_transform_to_mask(mask, tform, output_shape):
    """
    Apply transformation to binary mask
    
    Parameters
    ----------
    mask : 2D bool array
        Binary mask
    tform : skimage transform object
        Transformation
    output_shape : tuple
        Output shape (H, W)
    """
    transformed = warp(
        mask.astype(float),
        inverse_map=tform.inverse,
        output_shape=output_shape,
        order=0,  # Nearest neighbor for masks
        preserve_range=True,
        cval=0
    )
    return transformed > 0.5  # Re-binarize

def crop_atlas_to_fov(atlas, brain_mask_in_atlas_space, min_overlap=10):
    """
    Restrict atlas to only regions visible in your FOV
    
    Parameters
    ----------
    atlas : 2D array
        Atlas labels in atlas space
    brain_mask_in_atlas_space : 2D bool array
        Your brain mask transformed to atlas space
    min_overlap : int
        Minimum number of pixels required to keep an atlas region
        
    Returns
    -------
    atlas_cropped : 2D array
        Atlas restricted to FOV
    valid_regions : list
        List of atlas region IDs that are in FOV
    """
    atlas_cropped = atlas.copy()
    atlas_cropped[~brain_mask_in_atlas_space] = 0
    
    unique_regions = np.unique(atlas_cropped)
    unique_regions = unique_regions[unique_regions > 0]
    
    valid_regions = []
    for region in unique_regions:
        region_mask = atlas_cropped == region
        n_pixels = np.sum(region_mask)
        if n_pixels >= min_overlap:
            valid_regions.append(region)
        else:
            atlas_cropped[region_mask] = 0  # Remove small regions
    
    print(f"Atlas cropped to FOV: {len(valid_regions)}/{len(np.unique(atlas))-1} regions retained")
    
    return atlas_cropped, valid_regions

# -----------------------
# Interactive landmark collector (replaces ginput for n>=10 points)
# -----------------------
def _collect_landmarks(image, n_points, title, cmap="gray"):
    """
    Interactive landmark collection with live numbered markers.
    Left-click to place a landmark; right-click to remove the last one.
    Closes automatically once n_points landmarks have been placed,
    or when the user presses Enter to confirm early (only allowed when
    at least 4 points have been placed, to ensure a well-conditioned fit).

    Parameters
    ----------
    image : 2D numpy array
        Image to display.
    n_points : int
        Target number of landmarks (>=4 recommended for similarity).
    title : str
        Window title / instruction string.
    cmap : str
        Colormap for display.

    Returns
    -------
    pts : (N, 2) numpy array  [x, y] in image pixel coordinates, N >= 4
    """
    MIN_POINTS = 4  # hard floor for a well-determined similarity fit

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.imshow(image, cmap=cmap)
    ax.set_title(f"{title}\nLeft-click: add | Right-click: undo last | Enter: confirm (min {MIN_POINTS})")
    fig.tight_layout()

    pts = []          # list of [x, y]
    scatters = []     # list of PathCollection objects (one dot per point)
    texts = []        # list of Text objects (one label per point)

    def _redraw():
        """Refresh all markers and labels."""
        for s in scatters:
            s.remove()
        for t in texts:
            t.remove()
        scatters.clear()
        texts.clear()
        colors = plt.cm.cool(np.linspace(0, 1, max(len(pts), 1)))
        for i, (x, y) in enumerate(pts):
            sc = ax.scatter(x, y, c=[colors[i]], s=80, zorder=5, edgecolors='white', linewidths=0.8)
            tx = ax.text(x + 6, y - 6, str(i + 1), color='white', fontsize=9,
                         fontweight='bold', zorder=6,
                         bbox=dict(boxstyle='round,pad=0.15', fc=colors[i], alpha=0.7, lw=0))
            scatters.append(sc)
            texts.append(tx)
        remaining = n_points - len(pts)
        if remaining > 0:
            ax.set_xlabel(f"{len(pts)}/{n_points} placed  |  {remaining} remaining", fontsize=10)
        else:
            ax.set_xlabel(f"{len(pts)}/{n_points} placed  |  Press Enter to confirm", fontsize=10)
        fig.canvas.draw_idle()

    def _on_click(event):
        if event.inaxes != ax:
            return
        if event.button == 1:   # left-click → add
            pts.append([event.xdata, event.ydata])
            _redraw()
            if len(pts) >= n_points:
                print(f"  {n_points} landmarks placed. Close the window or press Enter to confirm.")
        elif event.button == 3:  # right-click → undo
            if pts:
                pts.pop()
                _redraw()
                print(f"  Last point removed. {len(pts)}/{n_points} placed.")

    def _on_key(event):
        if event.key == 'enter':
            if len(pts) < MIN_POINTS:
                print(f"  Need at least {MIN_POINTS} points before confirming (have {len(pts)}).")
                return
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', _on_click)
    fig.canvas.mpl_connect('key_press_event', _on_key)

    _redraw()
    plt.show(block=True)

    if len(pts) < MIN_POINTS:
        raise RuntimeError(
            f"Landmark collection cancelled or insufficient points "
            f"(need >= {MIN_POINTS}, got {len(pts)})."
        )

    print(f"  Collected {len(pts)} landmarks.")
    return np.array(pts)


# -----------------------
# Register Landmark Annotations (n >= 4, similarity transform)
# -----------------------
def register_atlas_landmarks(mean_image, atlas, n_points=10, mode="similarity",
                             registration_direction="atlas_to_mouse"):
    """
    Landmark-based atlas registration using interactive clicks.
    Supports n >= 4 landmarks for a well-conditioned similarity or affine fit.
    With n >= 10 landmarks the fit is robustly over-determined.

    The brain mask is NO LONGER drawn manually. After registration the atlas
    footprint (atlas > 0 warped into the target space) serves as the brain mask.

    Parameters
    ----------
    mean_image : 2D numpy array
        Calcium mean image (HxW).
    atlas : 2D numpy array
        Integer atlas labels (HxW) in atlas space.
    n_points : int
        Target number of landmark pairs to click (>= 4; >= 10 recommended).
    mode : str
        "similarity" (translation + rotation + uniform scale) [recommended]
        "affine"     (translation + rotation + scale + shear)
    registration_direction : str
        "atlas_to_mouse": Warp atlas into mouse image space (default).
        "mouse_to_atlas": Warp mouse image into atlas space.

    Returns
    -------
    image_warped : 2D numpy array
        Warped result in the target space.
    tform : skimage transform object
        Estimated transform (use tform.inverse for warp calls).
    brain_mask : 2D bool array
        Atlas footprint in the target space — use this as the brain mask.
    pts_mouse : (N, 2) array   [x, y] landmarks on the mouse image.
    pts_atlas : (N, 2) array   [x, y] landmarks on the atlas image.
    """

    if n_points < 4:
        raise ValueError("Need at least 4 landmarks for a well-conditioned similarity fit.")

    if registration_direction not in ["atlas_to_mouse", "mouse_to_atlas"]:
        raise ValueError("registration_direction must be 'atlas_to_mouse' or 'mouse_to_atlas'")

    # ------------------------------------------------------------------
    # Step 1: Interactive landmark collection
    # ------------------------------------------------------------------
    if registration_direction == "atlas_to_mouse":
        print(f"\n=== ATLAS → MOUSE registration ({n_points} landmarks) ===")
        print("Step 1/2 — Click landmarks on the MOUSE BRAIN image.")
        print("  Suggested landmarks: bregma, lambda, and prominent vessel branch-points.")
        pts_mouse = _collect_landmarks(
            mean_image, n_points,
            title="MOUSE BRAIN — click landmarks (bregma, lambda, vessel branch-points…)"
        )

        print("Step 2/2 — Click the SAME landmarks on the ATLAS (same order).")
        pts_atlas = _collect_landmarks(
            atlas, n_points,
            title="ATLAS — click the SAME landmarks in the same order",
            cmap="tab20"
        )

        src_pts = pts_atlas   # atlas coords  → source
        dst_pts = pts_mouse   # mouse coords  → destination

    else:  # mouse_to_atlas
        print(f"\n=== MOUSE → ATLAS registration ({n_points} landmarks) ===")
        print("Step 1/2 — Click landmarks on the ATLAS image.")
        pts_atlas = _collect_landmarks(
            atlas, n_points,
            title="ATLAS — click landmarks (bregma, lambda, vessel branch-points…)",
            cmap="tab20"
        )

        print("Step 2/2 — Click the SAME landmarks on the MOUSE BRAIN (same order).")
        pts_mouse = _collect_landmarks(
            mean_image, n_points,
            title="MOUSE BRAIN — click the SAME landmarks in the same order"
        )

        src_pts = pts_mouse   # mouse coords  → source
        dst_pts = pts_atlas   # atlas coords  → destination

    # ------------------------------------------------------------------
    # Step 2: Estimate transform (least-squares over all n points)
    # ------------------------------------------------------------------
    if mode == "similarity":
        tform = SimilarityTransform()
        tform.estimate(src=src_pts, dst=dst_pts)
    elif mode == "affine":
        from skimage.transform import AffineTransform
        tform = AffineTransform()
        tform.estimate(src=src_pts, dst=dst_pts)
    else:
        raise ValueError("mode must be 'similarity' or 'affine'")

    # Residuals — useful for QC / catching misclicks
    src_proj = tform(src_pts)   # predicted destination coords
    residuals = np.linalg.norm(src_proj - dst_pts, axis=1)
    print(f"\nLandmark residuals (pixels): mean={residuals.mean():.2f}  "
          f"max={residuals.max():.2f}  std={residuals.std():.2f}")
    if residuals.max() > 20:
        print("  WARNING: large residual detected — check for misclicked landmarks.")

    # ------------------------------------------------------------------
    # Step 3: Warp image & derive atlas-footprint brain mask
    # ------------------------------------------------------------------
    atlas_binary = (atlas > 0).astype(np.float32)   # 1 inside atlas, 0 outside

    if registration_direction == "atlas_to_mouse":
        output_shape = mean_image.shape

        # Warp atlas labels into mouse space (nearest-neighbour preserves integer IDs)
        image_warped = warp(
            atlas,
            inverse_map=tform.inverse,
            output_shape=output_shape,
            order=0,
            preserve_range=True,
            cval=0
        ).astype(np.int32)

        # Warp atlas footprint → brain mask in mouse space
        brain_mask = warp(
            atlas_binary,
            inverse_map=tform.inverse,
            output_shape=output_shape,
            order=0,
            preserve_range=True,
            cval=0
        ) > 0.5

        ref_img   = mean_image
        ovl_img   = image_warped
        ref_pts   = pts_mouse
        result_title = f"Atlas → Mouse  |  {mode}  |  {len(pts_mouse)} landmarks"

    else:  # mouse_to_atlas
        output_shape = atlas.shape

        # Warp mouse intensity into atlas space
        image_warped = warp(
            mean_image,
            inverse_map=tform.inverse,
            output_shape=output_shape,
            order=1,
            preserve_range=True,
            cval=0
        )

        # Brain mask in atlas space = the atlas footprint itself
        brain_mask = atlas_binary > 0.5

        ref_img   = atlas
        ovl_img   = image_warped
        ref_pts   = pts_atlas
        result_title = f"Mouse → Atlas  |  {mode}  |  {len(pts_mouse)} landmarks"

    # ------------------------------------------------------------------
    # Step 4: QC overlay
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: warped result overlaid on reference
    axes[0].imshow(ref_img, cmap="gray")
    axes[0].contour(brain_mask, colors="cyan", linewidths=1.0, alpha=0.8)
    if registration_direction == "atlas_to_mouse":
        axes[0].contour(image_warped, colors="r", linewidths=0.4, alpha=0.6)
    else:
        axes[0].imshow(ovl_img, cmap="gray", alpha=0.5)
    axes[0].plot(ref_pts[:, 0], ref_pts[:, 1], 'yo', markersize=6,
                 label='Landmarks', zorder=5)
    for i, (x, y) in enumerate(ref_pts):
        axes[0].text(x + 4, y - 4, str(i + 1), color='yellow', fontsize=7, zorder=6)
    axes[0].set_title(result_title)
    axes[0].legend(fontsize=8)

    # Panel 2: brain mask
    axes[1].imshow(ref_img, cmap="gray")
    axes[1].imshow(brain_mask, cmap="Reds", alpha=0.35)
    axes[1].set_title("Atlas-derived brain mask")

    # Panel 3: per-point residuals
    colors_r = plt.cm.RdYlGn_r(residuals / max(residuals.max(), 1e-6))
    axes[2].bar(np.arange(1, len(residuals) + 1), residuals, color=colors_r, edgecolor='k', linewidth=0.5)
    axes[2].axhline(5,  color='green',  linestyle='--', linewidth=1, label='5 px')
    axes[2].axhline(10, color='orange', linestyle='--', linewidth=1, label='10 px')
    axes[2].axhline(20, color='red',    linestyle='--', linewidth=1, label='20 px')
    axes[2].set_xlabel("Landmark index")
    axes[2].set_ylabel("Residual (pixels)")
    axes[2].set_title("Per-landmark registration error")
    axes[2].legend(fontsize=8)

    plt.suptitle(result_title, fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return image_warped, tform, brain_mask, pts_mouse, pts_atlas


# -----------------------
# Derive brain mask from registered atlas footprint
# -----------------------
def make_brain_mask_from_atlas(atlas_registered):
    """
    Create a binary brain mask from the registered atlas label image.
    Any pixel with a non-zero atlas label is considered 'inside' the brain.

    This replaces manual brain-mask drawing when the FOV reliably matches
    the dorsal Allen CCF view and the atlas is well-registered.

    Parameters
    ----------
    atlas_registered : 2D int array
        Atlas labels in the target image space (mouse or atlas space).
        Pixels outside the atlas footprint should be 0 / NaN.

    Returns
    -------
    brain_mask : 2D bool array
        True where atlas label > 0.
    """
    return np.nan_to_num(atlas_registered, nan=0) > 0

# ----------------------
# Make Mask of Brain FOV
# ----------------------

def _poly_mask_from_image_fixed(image):
    """
    Interactive polygon drawing on an image with improved event handling.
    Left-click: add vertices
    Right-click/double-click: finish polygon
    Returns: binary mask (H, W) or None if cancelled.
    """
    # Ensure we're using an interactive backend
    plt.ion()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image, cmap="gray")
    ax.set_title("Left-click: add vertices | Right/double-click: close polygon | Press 'q' to cancel")

    coords = []
    line, = ax.plot([], [], 'r-', linewidth=2, alpha=0.7)  # polygon line
    scat = ax.scatter([], [], c='r', s=40, marker='o')     # vertices

    def onselect(verts):
        coords[:] = verts
        if len(verts) > 0:
            xs, ys = zip(*verts)
            # Close the polygon for visualization
            xs_closed = xs + (xs[0],)
            ys_closed = ys + (ys[0],)
            line.set_data(xs_closed, ys_closed)
            scat.set_offsets(np.c_[xs, ys])
            fig.canvas.draw_idle()

    selector = PolygonSelector(
        ax, onselect,
        useblit=False,
        props=dict(color='r', linewidth=2, alpha=0.7)
    )
    
    # Add keyboard event handling
    def on_key(event):
        if event.key == 'q':
            plt.close(fig)
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    print("Instructions:")
    print("1. Left-click to add vertices")
    print("2. Right-click or double-click to finish polygon")
    print("3. Press 'q' to cancel")
    
    plt.show()
    
    # Wait for the figure to be closed
    while plt.fignum_exists(fig.number):
        plt.pause(0.1)

    if not coords:
        print("No polygon created or cancelled")
        return None  # user cancelled

    print(f"Polygon created with {len(coords)} vertices")

    # Convert polygon into binary mask
    ny, nx = image.shape
    y, x = np.mgrid[:ny, :nx]
    points = np.vstack((x.flatten(), y.flatten())).T
    path = Path(coords)
    mask = path.contains_points(points).reshape((ny, nx))

    return mask

def _simple_click_mask(image):
    """
    Alternative simpler approach using ginput for polygon creation.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(image, cmap="gray")
    plt.title("Click points for polygon (press Enter when done)")
    
    print("Click points to create polygon outline")
    print("Press Enter when finished")
    
    # Get points from user clicks
    points = plt.ginput(n=-1, timeout=0)  # n=-1 means unlimited points
    plt.close()
    
    if len(points) < 3:
        print("Need at least 3 points for polygon")
        return None
    
    print(f"Creating mask from {len(points)} points")
    
    # Convert to mask
    ny, nx = image.shape
    y, x = np.mgrid[:ny, :nx]
    grid_points = np.vstack((x.flatten(), y.flatten())).T
    path = Path(points)
    mask = path.contains_points(grid_points).reshape((ny, nx))
    
    # Show result for verification
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original")
    
    plt.subplot(1, 2, 2)
    plt.imshow(image, cmap="gray", alpha=0.7)
    plt.imshow(mask, cmap="Reds", alpha=0.3)
    plt.title("Mask overlay")
    plt.show()
    
    return mask

def make_brain_mask_fixed(image,
                         stack_for_std=None,
                         atlas=None,
                         method="auto",      # "auto" | "manual" | "hybrid" | "simple"
                         auto_params=None,
                         qc=True,
                         interactive_correction=True):
    """
    Robust brain-mask generator with improved interactive functionality.

    Parameters
    ----------
    image : 2D array
        Mean raw reflectance image (orientation should match atlas).
    stack_for_std : 3D array, optional
        Time stack used to compute temporal std (T,H,W) or (H,W,T). If None, std-based steps are skipped.
    atlas : 2D array, optional
        Registered atlas for QC overlay.
    method : str
        "auto" (intensity + std), "manual" (draw polygon), "simple" (simple clicking), or "hybrid" (auto then allow manual correction).
    auto_params : dict, optional
        Tunable parameters. Default values used when None.
    qc : bool
        Show QC overlays (intensity, std, masks).
    interactive_correction : bool
        If True and method in ('auto','hybrid'), allow the user to draw polygons to add/subtract from the auto mask.

    Returns
    -------
    brain_mask : 2D bool array
    """
    
    # Set matplotlib to interactive mode
    plt.ion()

    # default params
    if auto_params is None:
        auto_params = dict(
            tophat_disk=30,
            intensity_percentile=10,
            std_percentile=40,
            min_area=500,
            remove_small=200,
            glue_area_thresh=500,
            glue_intensity_z=5
        )

    # normalize input shapes
    img = np.array(image, dtype=float)
    
    # compute temporal std if stack provided
    std_map = None
    if stack_for_std is not None:
        s = np.array(stack_for_std)
        # accept (T,H,W) or (H,W,T)
        if s.ndim == 3:
            if s.shape[0] > 50:   # likely time is first axis (T,H,W)
                std_map = np.std(s, axis=0)
            elif s.shape[-1] > 50:  # likely time is last axis (H,W,T)
                std_map = np.std(s, axis=2)
            else:
                raise ValueError(f"Cannot determine time axis from shape {s.shape}. Please provide stack as (T,H,W) or (H,W,T).")
        else:
            raise ValueError("stack_for_std must be 3D (T,H,W) or (H,W,T).")

    if method == "manual":
        print("Using PolygonSelector method...")
        mask = _poly_mask_from_image_fixed(img)
        if mask is None:
            print("PolygonSelector failed, trying simple click method...")
            mask = _simple_click_mask(img)
        if mask is None:
            raise RuntimeError("Manual polygon failed or was cancelled.")
        mask = mask.astype(bool)
        
    elif method == "simple":
        print("Using simple click method...")
        mask = _simple_click_mask(img)
        if mask is None:
            raise RuntimeError("Simple polygon failed or was cancelled.")
        mask = mask.astype(bool)
        
    else:
        # AUTO METHOD (same as original)
        # -----------------------
        # 1) Background remove / enhance (top-hat or large gaussian subtract)
        # -----------------------
        try:
            from skimage.morphology import white_tophat
            from skimage.morphology import disk as sk_disk
            selem = sk_disk(auto_params['tophat_disk'])
            img_tophat = white_tophat(img, selem)
        except Exception:
            img_blur = gaussian_filter(img, sigma=auto_params['tophat_disk']/3.0)
            img_tophat = img - img_blur
            # rescale to positive
            img_tophat = img_tophat - np.min(img_tophat)
        # enhance contrast a bit
        img_tophat = exposure.rescale_intensity(img_tophat, out_range=(0,1))

        # -----------------------
        # 2) Intensity mask (conservative)
        # -----------------------
        p = auto_params['intensity_percentile']
        thr_int = np.percentile(img_tophat.flatten(), p)
        mask_int = img_tophat > thr_int

        # -----------------------
        # 3) Temporal-std mask (if available)
        # -----------------------
        if std_map is not None:
            # smooth std_map a bit
            std_s = gaussian_filter(std_map.astype(float), sigma=2)
            pstd = auto_params['std_percentile']
            thr_std = np.percentile(std_s.flatten(), pstd)
            mask_std = std_s > thr_std
            mask_comb = np.logical_and(mask_int, mask_std)
        else:
            mask_comb = mask_int

        # -----------------------
        # 4) Remove very bright static blobs (glue/edge) using intensity & std
        # -----------------------
        mean_val = np.mean(img_tophat)
        std_val = np.std(img_tophat)
        glue_z = auto_params['glue_intensity_z']
        bright_thresh = mean_val + glue_z * std_val
        bright_mask = img_tophat > bright_thresh
        # label bright components and remove those that are small (likely glue)
        lab = measure.label(bright_mask)
        props = measure.regionprops(lab)
        remove_labels = [p.label for p in props if p.area < auto_params['glue_area_thresh']]
        glue_removal_mask = np.isin(lab, remove_labels)
        # also check std in those regions: if std small, likely glue
        if std_map is not None:
            glue_removal_mask = np.logical_and(glue_removal_mask, (std_map < np.percentile(std_map, 30)))
        # subtract glue removal from mask_comb
        mask_comb = np.logical_and(mask_comb, ~glue_removal_mask)

        # -----------------------
        # 5) Morphological cleanup
        # -----------------------
        mask_clean = binary_closing(mask_comb, footprint=disk(5))
        mask_clean = binary_opening(mask_clean, footprint=disk(3))
        mask_clean = remove_small_objects(mask_clean, min_size=auto_params['remove_small'])
        mask_clean = binary_fill_holes(mask_clean)

        # -----------------------
        # 6) Keep largest connected component
        # -----------------------
        labels = measure.label(mask_clean)
        if labels.max() == 0:
            mask = mask_clean.astype(bool)
        else:
            props = measure.regionprops(labels)
            if len(props) == 0:
                mask = mask_clean.astype(bool)
            else:
                # choose largest area component
                largest = max(props, key=lambda x: x.area).label
                mask = labels == largest

    # -----------------------
    # optional interactive correction (add/subtract polygons)
    # -----------------------
    if interactive_correction and (method in ('auto', 'hybrid')):
        print("Automatic mask created. You can optionally CORRECT it.")
        
        # Show current mask
        plt.figure(figsize=(10, 8))
        plt.imshow(img, cmap='gray')
        plt.contour(mask, colors='r', linewidths=2)
        plt.title("Current mask (red outline)")
        plt.show()
        
        response = input("Do you want to add/subtract regions? (y/n): ").lower()
        
        if response == 'y':
            # ADD step
            print("Draw polygon to ADD to mask (or close window to skip)")
            add = _simple_click_mask(img)
            if add is not None:
                mask = np.logical_or(mask, add)
            
            # Show updated mask
            plt.figure(figsize=(10, 8))
            plt.imshow(img, cmap='gray')
            plt.contour(mask, colors='r', linewidths=2)
            plt.title("Updated mask after addition")
            plt.show()
            
            # SUBTRACT step
            response2 = input("Do you want to subtract regions? (y/n): ").lower()
            if response2 == 'y':
                print("Draw polygon to SUBTRACT from mask (or close window to skip)")
                sub = _simple_click_mask(img)
                if sub is not None:
                    mask = np.logical_and(mask, ~sub)

            # final cleanup after manual edits
            mask = binary_closing(mask, footprint=disk(5))
            mask = binary_fill_holes(mask)
            mask = remove_small_objects(mask, min_size=auto_params['remove_small'] // 2)

    brain_mask = mask.astype(bool)

    # -----------------------
    # QC plots
    # -----------------------
    if qc:
        ncol = 3 if (stack_for_std is not None and atlas is not None) else (2 if atlas is not None or stack_for_std is not None else 1)
        fig, axes = plt.subplots(1, ncol, figsize=(4*ncol, 4))
        if ncol == 1:
            axes = [axes]
        
        axidx = 0
        # show intensity and mask
        axes[axidx].imshow(img, cmap='bone')
        axes[axidx].contour(brain_mask, colors='r', linewidths=0.8)
        axes[axidx].set_title('Intensity + mask')
        axidx += 1

        if stack_for_std is not None:
            axes[axidx].imshow(std_map, cmap='magma')
            axes[axidx].set_title('Temporal std map')
            axidx += 1

        if atlas is not None:
            axes[axidx].imshow(img, cmap='bone')
            axes[axidx].contour(brain_mask, colors='r', linewidths=0.8)
            axes[axidx].contour(atlas, colors='b', linewidths=0.6)
            axes[axidx].set_title('Mask vs atlas')
        
        plt.tight_layout()
        plt.show()

    return brain_mask

def resample_timeseries(data, target_shape, is_label=False):
    """
    data: (T, H, W)
    target_shape: (H_target, W_target)
    """
    T = data.shape[0]
    out = np.zeros((T, *target_shape), dtype=data.dtype)

    for t in range(T):
        out[t] = resize(
            data[t],
            target_shape,
            order=0 if is_label else 1,
            preserve_range=True,
            anti_aliasing=not is_label
        )

    return out

def resample_frame(frame, target_shape, is_label=False):
    """
    frame: (H, W)
    target_shape: (H_target, W_target)
    """
    return resize(
        frame,
        target_shape,
        order=0 if is_label else 1,
        preserve_range=True,
        anti_aliasing=not is_label
    )