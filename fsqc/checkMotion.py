"""
This module provides MRIQC-style image quality metrics related to motion/noise.

Implemented measures:
- EFC       : Entropy Focus Criterion
- QI2       : Mortamet's quality index 2
- FBER      : Foreground-Background Energy Ratio
- SNR       : Signal-to-Noise Ratio, averaged over GM/WM/CSF tissue masks
              taken from a FreeSurfer/FastSurfer segmentation
- SNR_HEAD  : Signal-to-Noise Ratio over the whole (Otsu-thresholded) head
              mask, requiring no segmentation
- BG        : Background summary statistics
"""

import os

# -----------------------------------------------------------------------------
# private helper functions and variables
# -----------------------------------------------------------------------------

# FreeSurferColorLUT label IDs used to build tissue masks for SNR estimation.
# WM/GM lists match fsqc.checkSNR.checkSNR for consistency across the codebase.
_WM_LABELS = [2, 41, 7, 46, 251, 252, 253, 254, 255, 77, 78, 79]
_GM_LABELS = [3, 42]
_CSF_LABELS = [24, 4, 43, 5, 44, 14, 15]

#: Fallback chain (relative to a subject's mri/ dir) for checkMotion's
#: reference image, in preference order: the true pre-conform, native
#: volume first, then progressively more processed substitutes.
_REF_IMAGE_CANDIDATES = (
    os.path.join("orig", "001.mgz"),
    "rawavg.mgz",
    "orig.mgz",
)

def _airmask_from_headmask(headmask, rotmask=None):
    """Create an air mask as the complement of the head mask."""
    import numpy as np

    airmask = np.ones_like(headmask, dtype=np.uint8)
    airmask[headmask > 0] = 0
    if rotmask is not None:
        airmask[rotmask > 0] = 0

    return airmask


def _tissue_mask_from_labels(seg_data, labels, nb_erode=0):
    """Build a binary mask for the given segmentation label IDs, optionally eroded."""
    import numpy as np
    from skimage.morphology import binary_erosion

    mask = np.isin(seg_data, labels).astype(np.uint8)
    if nb_erode > 0:
        mask = binary_erosion(mask, np.ones((nb_erode, nb_erode, nb_erode))).astype(np.uint8)

    return mask


def _save_mask_nii(mask, affine, out_path):
    """Save a binary mask array as a NIfTI (.nii/.nii.gz) image."""
    import nibabel as nib

    nib.save(nib.nifti1.Nifti1Image(mask.astype("uint8"), affine), out_path)


def _load_external_mask(mask_file, mask_label):
    """
    Load and validate an externally supplied mask file against ``img``'s shape.

    Returns the binarized mask array on success. Raises ``FileNotFoundError``
    or ``ValueError`` on failure (missing file, load error, shape mismatch);
    the caller is expected to warn and fail closed (return NaNs) on exception,
    since an explicitly supplied mask should never be silently ignored.
    """
    if not os.path.exists(mask_file):
        raise FileNotFoundError("could not find external " + mask_label + " " + mask_file)
    candidate = nib.load(mask_file).get_fdata()
    if candidate.shape[:3] != img.shape[:3]:
        raise ValueError(
            "external " + mask_label + " " + mask_file + " has shape "
            + str(candidate.shape[:3]) + ", expected " + str(img.shape[:3])
        )
    return (candidate > 0).astype(np.uint8)


def _resolve_ref_image(subjects_dir, subject, ref_image=None):
    """
    Resolve the reference anatomical image path for checkMotion.

    If ``ref_image`` is given explicitly, it is used as-is (joined under
    ``mri/``), with no fallback. Otherwise, the preference chain
    ``mri/orig/001.mgz`` -> ``mri/rawavg.mgz`` -> ``mri/orig.mgz`` is tried in
    order, returning the first candidate that exists. A warning is issued
    whenever a fallback (2nd or later) candidate is used, since each
    represents a degraded substitute for the preferred pre-conform image.

    Parameters
    ----------
    subjects_dir : str
        The directory containing subject data.
    subject : str
        The name of the subject.
    ref_image : str or None, optional
        Explicit path (relative to ``mri/``) to use, bypassing the fallback
        chain.

    Returns
    -------
    str or None
        Absolute path to the resolved reference image, or ``None`` if no
        candidate exists.
    """
    import warnings

    if ref_image is not None:
        return os.path.join(subjects_dir, subject, "mri", ref_image)

    for i, candidate in enumerate(_REF_IMAGE_CANDIDATES):
        candidate_path = os.path.join(subjects_dir, subject, "mri", candidate)
        if os.path.exists(candidate_path):
            if i > 0:
                warnings.warn(
                    "WARNING: preferred reference image(s) ("
                    + ", ".join(_REF_IMAGE_CANDIDATES[:i])
                    + ") not found for " + subject + "; falling back to "
                    + candidate + ".",
                    stacklevel=3,
                )
            return candidate_path

    return None


def _conformImageToRAS(in_img):
    """
    Conform an anatomical image the way mriqc's anatomical workflow does.

    Mirrors mriqc's ``mriqc.interfaces.common.ConformImage`` as it is
    actually invoked in mriqc's anatomical workflow
    (``ConformImage(check_dtype=False)``): squeeze a redundant 4th
    dimension, then reorient to RAS canonical orientation
    (``nib.as_closest_canonical``). No dtype casting and no resampling to a
    fixed voxel size/shape is performed (unlike FreeSurfer's own
    ``mri_convert --conform``).

    Parameters
    ----------
    in_img : nibabel.spatialimages.SpatialImage or str or os.PathLike
        Input image, or a path to load one from.

    Returns
    -------
    nibabel.spatialimages.SpatialImage
        Conformed (squeezed + RAS-reoriented) image.
    """
    import nibabel as nib

    if isinstance(in_img, (str, os.PathLike)):
        in_img = nib.load(in_img)

    return nib.as_closest_canonical(nib.squeeze_image(in_img))


def _computeRotmaskMortamet(img, min_size=500):
    """
    Compute a rotation-artifact mask using the method of [Mortamet2009]_.

    Ported from mriqc's ``mriqc.interfaces.anatomical.RotationMask``. Flags
    hard-zero (``<= 0``) voxels left by an obliquely prescribed acquisition
    being reconstructed into a rectangular voxel grid (not resampling
    padding from a conform step). Thin/noisy zero specks are removed via
    binary opening, then only the largest couple of connected components
    (the real cut-corner region(s), which merge with the padded border) are
    kept; the whole mask is discarded if it is too small to be a genuine
    artifact.

    Parameters
    ----------
    img : numpy.ndarray
        3D image array.
    min_size : int, optional
        Minimum number of voxels for the detected mask to be kept; smaller
        masks are treated as noise and zeroed out (default: 500, matching
        mriqc's hardcoded threshold).

    Returns
    -------
    numpy.ndarray
        Binary rotation mask (uint8, 1 = detected hard-zero artifact region).
    """
    import numpy as np
    from scipy import ndimage as nd

    mask = img <= 0

    # Pad one voxel so real cut-corner regions (which touch the volume
    # border) merge into a single component with the padding, separating
    # them from small interior zero specks during the labeling step below.
    mask = np.pad(mask, pad_width=(1,), mode="constant", constant_values=1)

    struct = nd.generate_binary_structure(3, 2)
    mask = nd.binary_opening(mask, structure=struct).astype(np.uint8)

    label_im, nb_labels = nd.label(mask)
    if nb_labels > 2:
        sizes = nd.sum(mask, label_im, list(range(nb_labels + 1)))
        ordered = sorted(zip(sizes, list(range(nb_labels + 1))), reverse=True)
        for _, label in ordered[2:]:
            mask[label_im == label] = 0

    mask = mask[1:-1, 1:-1, 1:-1]

    if mask.sum() < min_size:
        mask = np.zeros_like(mask, dtype=np.uint8)

    return mask.astype(np.uint8)


def _computeHeadmaskOtsu(img, rotmask=None, aseg_data=None):
    """
    Compute a head mask from image intensities using Otsu thresholding.

    Parameters
    ----------
    img : numpy.ndarray
        3D image array.
    rotmask : numpy.ndarray or None, optional
        Optional rotmask where non-zero voxels are excluded from threshold
        estimation and output mask.
    aseg_data : numpy.ndarray or None, optional
        Optional FreeSurfer/FastSurfer ``aseg`` segmentation, same shape as
        ``img``. Any voxel with a non-background label is unioned into the
        head mask, since it's unambiguously part of the head regardless of
        its intensity.

    Returns
    -------
    numpy.ndarray
        Binary head mask (uint8, 1 = foreground/head). Thin channels
        connecting interior cavities to the background are closed first,
        then interior holes (e.g. ventricles/CSF, or other dark-but-
        inside-the-head voxels that fall below the threshold) are filled
        in, so they aren't misclassified as background via the airmask
        complement.
    """
    import numpy as np
    from scipy.ndimage import binary_closing, binary_fill_holes

    finite = np.isfinite(img)
    if rotmask is not None:
        finite = np.logical_and(finite, rotmask == 0)

    data = img[finite]
    data = data[data > 0]
    if data.size == 0:
        return np.zeros_like(img, dtype=np.uint8)

    # Otsu threshold from a fixed histogram for reproducibility.
    hist, bin_edges = np.histogram(data, bins=256)
    hist = hist.astype(np.float64)

    if not np.any(hist):
        return np.zeros_like(img, dtype=np.uint8)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    mu1 = np.cumsum(hist * bin_centers) / np.maximum(weight1, 1.0)
    mu2 = (
        np.cumsum((hist * bin_centers)[::-1]) / np.maximum(weight2[::-1], 1.0)
    )[::-1]

    sigma_b2 = weight1[:-1] * weight2[1:] * (mu1[:-1] - mu2[1:]) ** 2
    if sigma_b2.size == 0:
        threshold = float(np.median(data))
    else:
        threshold = float(bin_centers[np.argmax(sigma_b2)])

    headmask = np.zeros_like(img, dtype=np.uint8)
    headmask[img > threshold] = 1

    if aseg_data is not None:
        # Any voxel already segmented as brain tissue is definitely part of
        # the head, regardless of its (possibly dark) intensity.
        headmask = np.logical_or(headmask, aseg_data > 0).astype(np.uint8)

    # Seal thin channels (e.g. sinuses, gaps in the thresholded shell) that
    # would otherwise connect large interior cavities to the background, so
    # fill_holes below can recognize them as fully enclosed.
    headmask = binary_closing(headmask, iterations=5)

    # Fill interior holes (e.g. ventricles/CSF) so dark-but-inside-the-head
    # voxels aren't misclassified as background via the airmask complement.
    headmask = binary_fill_holes(headmask).astype(np.uint8)

    if rotmask is not None:
        headmask[rotmask > 0] = 0

    return headmask

# -----------------------------------------------------------------------------
# metrics
# -----------------------------------------------------------------------------

def efc(img, rotmask=None):
    """
    Compute MRIQC's entropy focus criterion (EFC).

    Lower values are better.
    """
    import numpy as np

    if rotmask is None:
        rotmask = np.zeros_like(img, dtype=np.uint8)

    valid = rotmask == 0
    n_vox = int(np.sum(valid))
    if n_vox < 2:
        return np.nan

    b_max = np.sqrt((img[valid] ** 2).sum())
    if b_max <= 0:
        return np.nan

    efc_max = n_vox * (1.0 / np.sqrt(n_vox)) * np.log(1.0 / np.sqrt(n_vox))

    return float(
        (1.0 / efc_max)
        * np.sum((img[valid] / b_max) * np.log((img[valid] + 1.0e-16) / b_max))
    )


def fber(img, headmask, rotmask=None):
    """
    Compute MRIQC's foreground-background energy ratio (FBER).

    Higher values are better.
    """
    import numpy as np

    fg = img[headmask > 0]
    if fg.size == 0:
        return np.nan

    fg_mu = np.median(np.abs(fg) ** 2)

    airmask = np.ones_like(headmask, dtype=np.uint8)
    airmask[headmask > 0] = 0

    if rotmask is not None:
        airmask[rotmask > 0] = 0

    bg = img[airmask == 1]
    if bg.size == 0:
        return np.nan

    bg_mu = np.median(np.abs(bg) ** 2)
    if bg_mu < 1.0e-3:
        return -1.0

    return float(fg_mu / bg_mu)


def snr(mu_fg, sigma_fg, n):
    """
    Compute MRIQC's SNR estimate from foreground summary statistics.

    Parameters
    ----------
    mu_fg : float
        Foreground mean or median intensity.
    sigma_fg : float
        Foreground standard deviation.
    n : int
        Number of foreground voxels.
    """
    import numpy as np

    if n <= 1 or sigma_fg <= 0:
        return np.nan

    return float(mu_fg / (sigma_fg * np.sqrt(n / (n - 1))))


def snr_tissue(img, aseg_data, aparc_data, nb_erode_wm=3, nb_erode_csf=1):
    """
    Compute MRIQC-style SNR from FreeSurfer/FastSurfer tissue segmentations.

    Builds gray matter, white matter and CSF (incl. ventricles) masks from
    ``aseg_data``/``aparc_data`` and computes :func:`snr` for each. White
    matter and CSF are eroded to reduce partial-volume contamination; gray
    matter is not eroded (the cortical ribbon is too thin).

    Parameters
    ----------
    img : numpy.ndarray
        3D reference intensity image.
    aseg_data : numpy.ndarray
        FreeSurfer/FastSurfer ``aseg`` segmentation array, same shape as ``img``.
    aparc_data : numpy.ndarray
        FreeSurfer/FastSurfer ``aparc+aseg``-style segmentation array, same
        shape as ``img``.
    nb_erode_wm : int, optional
        Erosion (in voxels) applied to the white matter mask (default: 3).
    nb_erode_csf : int, optional
        Erosion (in voxels) applied to the CSF mask (default: 1).

    Returns
    -------
    dict
        Dictionary with keys ``snr`` (mean over available tissue classes),
        ``snr_gm``, ``snr_wm`` and ``snr_csf``.
    """
    import numpy as np

    gm_mask = _tissue_mask_from_labels(aseg_data, _GM_LABELS, nb_erode=0)
    wm_mask = _tissue_mask_from_labels(aparc_data, _WM_LABELS, nb_erode=nb_erode_wm)
    csf_mask = _tissue_mask_from_labels(aseg_data, _CSF_LABELS, nb_erode=nb_erode_csf)

    result = {}
    for key, mask in (("snr_gm", gm_mask), ("snr_wm", wm_mask), ("snr_csf", csf_mask)):
        fg = img[mask > 0]
        result[key] = (
            snr(float(np.median(fg)), float(np.std(fg)), int(fg.size))
            if fg.size > 0
            else np.nan
        )

    values = [v for v in result.values() if np.isfinite(v)]
    result["snr"] = float(np.mean(values)) if values else np.nan

    return result


def bg(img, airmask):
    """
    Compute background summary statistics akin to MRIQC's summary_bg_* measures.
    """
    import numpy as np
    from scipy.stats import kurtosis

    data = np.nan_to_num(img[airmask > 0], nan=0.0, posinf=0.0, neginf=0.0)

    if data.size == 0:
        return {
            "bg_mean": np.nan,
            "bg_median": np.nan,
            "bg_std": np.nan,
            "bg_mad": np.nan,
            "bg_kurtosis": np.nan,
            "bg_p05": np.nan,
            "bg_p95": np.nan,
            "bg_n": 0,
        }

    med = np.median(data)
    return {
        "bg_mean": float(np.mean(data)),
        "bg_median": float(med),
        "bg_std": float(np.std(data)),
        "bg_mad": float(np.median(np.abs(data - med)) / 0.6745),
        "bg_kurtosis": float(kurtosis(data)),
        "bg_p05": float(np.percentile(data, 5)),
        "bg_p95": float(np.percentile(data, 95)),
        "bg_n": int(data.size),
    }


def qi2(img, airmask, min_voxels=int(1e3), max_voxels=int(3e5), coil_elements=32):
    """
    Compute MRIQC's Mortamet QI2 goodness-of-fit score.

    Lower values are better.
    """
    import numpy as np
    from scipy.stats import chi2
    from sklearn.neighbors import KernelDensity

    np.random.seed(1191935)

    data = np.nan_to_num(img[airmask > 0], nan=0.0, posinf=0.0, neginf=0.0)
    data[data < 0] = 0

    if int((data > 0).sum()) < int(min_voxels):
        return 0.0

    p99 = np.percentile(data, 99)
    if p99 <= 0:
        return np.nan

    data = data * (100.0 / p99)
    modelx = data if len(data) < int(max_voxels) else np.random.choice(data, size=int(max_voxels))

    x_grid = np.linspace(0.0, 110.0, 1000)

    # Estimate empirical PDF and fit a chi-square model on the same support.
    kde_skl = KernelDensity(kernel="gaussian", bandwidth=4.0).fit(modelx[:, np.newaxis])
    kde_pdf = np.exp(kde_skl.score_samples(x_grid[:, np.newaxis]))

    kdethi = int(np.argmax(kde_pdf[::-1] > kde_pdf.max() * 0.5))

    params = chi2.fit(modelx, coil_elements)
    chi_pdf = chi2.pdf(x_grid, *params[:-2], loc=params[-2], scale=params[-1])

    return float(np.abs(kde_pdf[-kdethi:] - chi_pdf[-kdethi:]).mean())

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def checkMotion(
    subjects_dir,
    subject,
    ref_image=None,
    output_dir=None,
    write_masks=True, # False # TODO: revert for production
    rotmask_file=None,
    headmask_file=None,
    airmask_file=None,
    aseg_image="aseg.mgz",
    aparc_image="aparc+aseg.mgz",
    nb_erode_wm=3,
    nb_erode_csf=1,
):
    """
    Compute MRIQC-style EFC, QI2, FBER, SNR and BG measures for a subject.

    Parameters
    ----------
    subjects_dir : str
        The directory containing subject data.
    subject : str
        The name of the subject.
    ref_image : str or None, optional
        Reference MRI volume under ``mri/``. If omitted (default), it is
        resolved via a fallback chain: ``orig/001.mgz`` (the true
        pre-conform, native volume) first, then ``rawavg.mgz``, then
        ``orig.mgz`` (FreeSurfer's own conformed volume), with a warning
        whenever a fallback is used. Pass an explicit path (relative to
        ``mri/``) to bypass the fallback chain. Whichever image is resolved
        is conformed the way mriqc's anatomical workflow does (see
        :func:`_conformImageToRAS`) before any metric is computed.
    output_dir : str or None, optional
        Subject-specific metrics output folder to write debug mask images
        to (e.g. the caller's ``metrics_outdir``). Required for
        ``write_masks`` to have any effect.
    write_masks : bool, optional
        If true and ``output_dir`` is given, save the ``rotmask``,
        ``headmask``, and ``airmask`` actually used (computed or externally
        supplied) as NIfTI images under ``output_dir`` (default: ``False``).
    rotmask_file : str or None, optional
        Full path to an externally supplied rotation mask (NIfTI), in the
        same (conformed) grid as the resolved ``ref_image``. If omitted, the
        rotmask is computed internally via :func:`_computeRotmaskMortamet`
        (mriqc's Mortamet2009 hard-zero detection) directly on the conformed
        reference image; no pre-conform comparison is needed.
    headmask_file : str or None, optional
        Full path to an externally supplied head mask (NIfTI), in the same
        (conformed) grid as the resolved ``ref_image``. If omitted, the
        headmask is computed internally via Otsu thresholding.
    airmask_file : str or None, optional
        Full path to an externally supplied air mask (NIfTI), in the same
        (conformed) grid as the resolved ``ref_image``, used for QI2, FBER,
        and BG. If omitted, the complement of the (computed or supplied)
        ``headmask``/``rotmask`` is used.
    aseg_image : str, optional
        FreeSurfer/FastSurfer ``aseg`` segmentation under ``mri/``, used for
        the gray matter and CSF SNR masks (default: ``aseg.mgz``). Tissue-
        based SNR always loads this together with ``mri/orig.mgz`` (not the
        resolved ``ref_image``), since ``aseg``/``aparc`` segmentations only
        ever exist in FreeSurfer's own conformed grid.
    aparc_image : str, optional
        FreeSurfer/FastSurfer ``aparc+aseg``-style segmentation under ``mri/``,
        used for the white matter SNR mask (default: ``aparc+aseg.mgz``; pass
        e.g. ``aparc.DKTatlas+aseg.deep.mgz`` for FastSurfer output).
    nb_erode_wm : int, optional
        Erosion (in voxels) applied to the white matter SNR mask (default: 3).
    nb_erode_csf : int, optional
        Erosion (in voxels) applied to the CSF SNR mask (default: 1).

    Returns
    -------
    dict
        Dictionary with keys ``efc``, ``qi2``, ``fber``, ``snr`` (mean SNR over
        GM/WM/CSF tissue masks), ``snr_gm``, ``snr_wm``, ``snr_csf``,
        ``snr_head`` (whole-head SNR, requires no segmentation), and BG summary
        statistics: ``bg_mean``, ``bg_median``, ``bg_std``, ``bg_mad``,
        ``bg_kurtosis``, ``bg_p05``, ``bg_p95``, ``bg_n``.
    """
    import logging
    import os
    import warnings

    import nibabel as nib
    import numpy as np

    def _nan_metrics_dict():
        return {
            "efc": np.nan,
            "qi2": np.nan,
            "fber": np.nan,
            "snr": np.nan,
            "snr_gm": np.nan,
            "snr_wm": np.nan,
            "snr_csf": np.nan,
            "snr_head": np.nan,
            "bg_mean": np.nan,
            "bg_median": np.nan,
            "bg_std": np.nan,
            "bg_mad": np.nan,
            "bg_kurtosis": np.nan,
            "bg_p05": np.nan,
            "bg_p95": np.nan,
            "bg_n": 0,
        }

    logging.captureWarnings(True)
    logging.info("Computing MRIQC-style motion/noise metrics ...")

    # make sure the debug-mask output folder exists before anything tries to write to it
    if write_masks and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # locate the reference volume (orig/001.mgz -> rawavg.mgz -> orig.mgz
    # fallback chain, unless an explicit ref_image override was given); bail
    # out with NaNs if none of the candidates exist
    ref_path = _resolve_ref_image(subjects_dir, subject, ref_image)
    if ref_path is None:
        warnings.warn(
            "WARNING: could not open reference image for " + subject
            + " (tried " + ", ".join(_REF_IMAGE_CANDIDATES) + "), returning NaNs.",
            stacklevel=2,
        )
        return _nan_metrics_dict()

    # conform the reference volume the way mriqc's anatomical workflow does
    # (squeeze + RAS-reorient, no resampling) before computing any metric
    ref_img = _conformImageToRAS(ref_path)
    img = ref_img.get_fdata()

    # Resolve the rotmask: either an externally supplied mask, or (the common
    # case) computed via mriqc's Mortamet2009 hard-zero detection directly
    # on the conformed reference image (no pre-conform comparison needed).
    if rotmask_file is not None:
        try:
            rotmask = _load_external_mask(rotmask_file, "rotmask")
            logging.info("Using external rotmask " + rotmask_file)
        except Exception as exc:
            warnings.warn(
                "WARNING: could not use external rotmask " + rotmask_file
                + " (" + str(exc) + "), returning NaNs.",
                stacklevel=2,
            )
            return _nan_metrics_dict()
    else:
        rotmask = _computeRotmaskMortamet(img)

    if write_masks and output_dir is not None:
        _save_mask_nii(rotmask, ref_img.affine, os.path.join(output_dir, "rotmask.nii.gz"))

    # Resolve the headmask: either an externally supplied mask, or (the common
    # case) computed via Otsu thresholding on ref_image intensities, excluding
    # any rotmask (outside-FOV) voxels from the estimate.
    if headmask_file is not None:
        try:
            headmask = _load_external_mask(headmask_file, "headmask")
            logging.info("Using external headmask " + headmask_file)
        except Exception as exc:
            warnings.warn(
                "WARNING: could not use external headmask " + headmask_file
                + " (" + str(exc) + "), returning NaNs.",
                stacklevel=2,
            )
            return _nan_metrics_dict()
    else:
        headmask = _computeHeadmaskOtsu(img, rotmask=rotmask)

    if write_masks and output_dir is not None:
        _save_mask_nii(headmask, ref_img.affine, os.path.join(output_dir, "headmask.nii.gz"))

    # Resolve the airmask used for QI2, FBER, and BG: either an externally
    # supplied mask, or (the common case) the complement of headmask/rotmask.
    if airmask_file is not None:
        try:
            airmask = _load_external_mask(airmask_file, "airmask")
            logging.info("Using external airmask " + airmask_file)
        except Exception as exc:
            warnings.warn(
                "WARNING: could not use external airmask " + airmask_file
                + " (" + str(exc) + "), returning NaNs.",
                stacklevel=2,
            )
            return _nan_metrics_dict()
    else:
        airmask = _airmask_from_headmask(headmask, rotmask=rotmask)

    if write_masks and output_dir is not None:
        _save_mask_nii(airmask, ref_img.affine, os.path.join(output_dir, "airmask.nii.gz"))

    # Whole-head SNR needs no segmentation: it summarizes intensities over headmask directly.
    fg = img[headmask > 0]
    if fg.size == 0:
        snr_head_value = np.nan
    else:
        snr_head_value = snr(float(np.median(fg)), float(np.std(fg)), int(fg.size))

    # Tissue-based SNR (GM/WM/CSF) always uses FreeSurfer's own conformed
    # grid (orig.mgz + aseg.mgz + aparc_image): aseg/aparc segmentations only
    # ever exist in that grid, independent of whichever image was resolved
    # as the main reference above.
    tissue_ref_path = os.path.join(subjects_dir, subject, "mri", "orig.mgz")
    aseg_path = os.path.join(subjects_dir, subject, "mri", aseg_image)
    aparc_path = os.path.join(subjects_dir, subject, "mri", aparc_image)
    missing = [p for p in (tissue_ref_path, aseg_path, aparc_path) if not os.path.exists(p)]

    if not missing:
        tissue_img = nib.load(tissue_ref_path).get_fdata()
        aseg_data = nib.load(aseg_path).get_fdata()
        aparc_data = nib.load(aparc_path).get_fdata()
        tissue_snr = snr_tissue(
            tissue_img,
            aseg_data,
            aparc_data,
            nb_erode_wm=nb_erode_wm,
            nb_erode_csf=nb_erode_csf,
        )
    else:
        warnings.warn(
            "WARNING: could not open " + ", ".join(missing)
            + ", returning NaNs for tissue-based SNR.",
            stacklevel=2,
        )
        tissue_snr = {"snr": np.nan, "snr_gm": np.nan, "snr_wm": np.nan, "snr_csf": np.nan}

    # Assemble the final metrics dict: EFC/QI2/FBER/whole-head SNR first, then
    # tissue-based SNR and background summary stats merged in.
    metrics = {
        "efc": efc(img, rotmask=rotmask),
        "qi2": qi2(img, airmask=airmask),
        "fber": fber(img, headmask=headmask, rotmask=rotmask),
        "snr_head": snr_head_value,
    }
    metrics.update(tissue_snr)
    metrics.update(bg(img, airmask))

    # Summary logging for quick visual sanity-checking in pipeline logs.
    logging.info("EFC: " + f"{metrics['efc']:.4}" if np.isfinite(metrics["efc"]) else "EFC: nan")
    logging.info("QI2: " + f"{metrics['qi2']:.4}" if np.isfinite(metrics["qi2"]) else "QI2: nan")
    logging.info("FBER: " + f"{metrics['fber']:.4}" if np.isfinite(metrics["fber"]) else "FBER: nan")
    logging.info("SNR: " + f"{metrics['snr']:.4}" if np.isfinite(metrics["snr"]) else "SNR: nan")
    logging.info(
        "SNR_HEAD: " + f"{metrics['snr_head']:.4}"
        if np.isfinite(metrics["snr_head"])
        else "SNR_HEAD: nan"
    )

    return metrics
