"""
This module provides MRIQC-style image quality metrics related to motion/noise.

Metrics (EFC, FBER, SNR, background summary stats) are computed via
``mriqc.qc.anatomical``, on a single unified image grid shared with the
subject's FreeSurfer/FastSurfer ``aseg``/``aparc`` segmentation (``orig.mgz``
by default); QI2 is computed by a local, pure-computation port of mriqc's
``art_qi2`` (see :func:`_computeQi2`), to avoid its unconditional SVG report
output. Before EFC/FBER/SNR/background stats are computed, the reference
image is harmonized (rescaled so the white-matter mask's median intensity is
1000), mirroring mriqc's own ``Harmonize`` step; QI2 uses the
merely-conformed (non-harmonized) image, matching mriqc's actual wiring.

Implemented measures:
- EFC              : Entropy Focus Criterion
- QI2              : Mortamet's quality index 2
- FBER             : Foreground-Background Energy Ratio
- SNR_HEAD         : Signal-to-Noise Ratio over the (externally supplied)
                      head mask, requiring no segmentation
- SNR_TISSUE_*     : Signal-to-Noise Ratio in the GM/WM/CSF tissue masks
                      derived from a FreeSurfer/FastSurfer segmentation
                      (``_gm``, ``_wm``, ``_csf``), and their mean
                      (``_total``)
- BG               : Background summary statistics
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


def _load_external_mask(mask_file, mask_label, ref_img_shape, binarize=True):
    """
    Load and validate an externally supplied volume against ``ref_img_shape``.

    Returns the loaded array (binarized, unless ``binarize=False``) on
    success. Raises ``FileNotFoundError`` or ``ValueError`` on failure
    (missing file, load error, shape mismatch); the caller is expected to
    warn and fail closed (return NaNs) on exception, since an explicitly
    supplied mask/segmentation should never be silently ignored.
    """
    import nibabel as nib

    if not os.path.exists(mask_file):
        raise FileNotFoundError("could not find external " + mask_label + " " + mask_file)
    candidate = nib.load(mask_file).get_fdata()
    if candidate.shape[:3] != ref_img_shape:
        raise ValueError(
            "external " + mask_label + " " + mask_file + " has shape "
            + str(candidate.shape[:3]) + ", expected " + str(ref_img_shape)
        )
    return (candidate > 0).astype("uint8") if binarize else candidate


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



def _harmonizeImage(img, wm_mask):
    """
    Rescale ``img`` so that its median intensity within ``wm_mask`` is 1000.

    Mirrors mriqc's ``mriqc.interfaces.anatomical.Harmonize._run_interface``
    (not importable as a plain function -- only available as a nipype
    Interface). ``wm_mask`` is expected to already be eroded (fsqc reuses
    the same eroded white-matter mask built for tissue-based SNR, rather
    than eroding a second time as mriqc's ``Harmonize`` does internally).

    Parameters
    ----------
    img : numpy.ndarray
        3D image array (typically the conformed reference image).
    wm_mask : numpy.ndarray
        Binary white-matter mask, same shape as ``img``.

    Returns
    -------
    numpy.ndarray or None
        The rescaled image, or ``None`` if ``wm_mask`` is empty or its
        median intensity is non-positive/non-finite (harmonization not
        possible).
    """
    import numpy as np

    if not np.any(wm_mask):
        return None

    wm_median = float(np.median(img[wm_mask > 0]))
    if not np.isfinite(wm_median) or wm_median <= 0:
        return None

    return img * (1000.0 / wm_median)


def _computeRotmaskMortamet(img, min_size=500):
    """
    Compute a rotation-artifact mask using the method of [Mortamet2009]_.

    Ported from mriqc's ``mriqc.interfaces.anatomical.RotationMask`` (not
    importable as a plain function -- only available as a nipype
    Interface). Flags hard-zero (``<= 0``) voxels left by an obliquely
    prescribed acquisition being reconstructed into a rectangular voxel grid
    (not resampling padding from a conform step). Thin/noisy zero specks are
    removed via binary opening, then only the largest couple of connected
    components (the real cut-corner region(s), which merge with the padded
    border) are kept; the whole mask is discarded if it is too small to be a
    genuine artifact.

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


def _computeQi2(img, airmask, min_voxels=int(1e3), max_voxels=int(3e5), coil_elements=32):
    """
    Compute Mortamet's QI2: goodness-of-fit of the background noise
    intensity distribution (within ``airmask``) onto a centered chi-square
    distribution.

    Ported from mriqc's ``mriqc.qc.anatomical.art_qi2``, keeping only the
    numeric computation. Unlike mriqc's version, this never writes an
    ``error.svg`` placeholder or fit plot to disk -- those are purely a
    reporting side effect, unconditional even when unwanted, that fsqc has
    no use for.

    Parameters
    ----------
    img : numpy.ndarray
        3D image array.
    airmask : numpy.ndarray
        Binary air/background mask, same shape as ``img``.
    min_voxels : int, optional
        Minimum number of positive-intensity background voxels required to
        attempt the fit; below this, QI2 is defined as 0.0 (default: 1000,
        matching mriqc).
    max_voxels : int, optional
        Background voxels are subsampled to at most this many for the KDE /
        chi-square fit (default: 300000, matching mriqc).
    coil_elements : int, optional
        Number of coil elements, used as the initial degrees-of-freedom
        guess for the chi-square fit (default: 32, matching mriqc).

    Returns
    -------
    float
        QI2: mean absolute difference between the KDE-estimated background
        intensity distribution and the fitted chi-square distribution, over
        the upper tail (above the KDE's half-maximum cutoff).
    """
    import numpy as np
    from scipy.stats import chi2
    from sklearn.neighbors import KernelDensity

    # S. Ogawa was born
    np.random.seed(1191935)

    data = np.nan_to_num(img[airmask > 0], posinf=0.0)
    data[data < 0] = 0

    if (data > 0).sum() < min_voxels:
        return 0.0

    data *= 100 / np.percentile(data, 99)
    modelx = data if len(data) < max_voxels else np.random.choice(data, size=max_voxels)

    x_grid = np.linspace(0.0, 110, 1000)

    # Estimate data pdf with KDE on a random subsample
    kde_skl = KernelDensity(kernel="gaussian", bandwidth=4.0).fit(modelx[:, np.newaxis])
    kde = np.exp(kde_skl.score_samples(x_grid[:, np.newaxis]))

    # Find cutoff
    kdethi = np.argmax(kde[::-1] > kde.max() * 0.5)

    # Fit X^2
    param = chi2.fit(modelx, coil_elements)
    chi_pdf = chi2.pdf(x_grid, *param[:-2], loc=param[-2], scale=param[-1])

    # Compute goodness-of-fit (gof)
    return float(np.abs(kde[-kdethi:] - chi_pdf[-kdethi:]).mean())


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def checkMotion(
    subjects_dir,
    subject,
    ref_image="orig.mgz",
    output_dir=None,
    write_masks=False,
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

    All measures are computed on a single reference image grid, shared with
    the subject's FreeSurfer/FastSurfer ``aseg``/``aparc`` segmentation
    (``ref_image`` defaults to ``orig.mgz``, i.e. FreeSurfer's own conformed
    volume, precisely so it lines up with ``aseg``/``aparc``). The resolved
    image is conformed the way mriqc's anatomical workflow does (see
    :func:`_conformImageToRAS`), then harmonized (see
    :func:`_harmonizeImage`) by rescaling it so the eroded white-matter
    mask's median intensity is 1000, mirroring mriqc's own ``Harmonize``
    step. EFC, FBER, SNR and background stats are computed on the harmonized
    image; QI2 is computed on the conformed-but-unharmonized image, matching
    mriqc's actual wiring (its QI2 node is fed ``in_ras``, not
    ``in_noinu``), via a local port of mriqc's QI2 computation (see
    :func:`_computeQi2`).

    Parameters
    ----------
    subjects_dir : str
        The directory containing subject data.
    subject : str
        The name of the subject.
    ref_image : str, optional
        Reference MRI volume under ``mri/`` (default: ``orig.mgz``). Must be
        on the same grid as ``aseg_image``/``aparc_image`` -- overriding it
        with an image on a different grid will fail closed (NaN metrics)
        once the tissue-segmentation shape check fails.
    output_dir : str or None, optional
        Subject-specific metrics output folder to write debug mask images
        to (e.g. the caller's ``metrics_outdir``). Required for
        ``write_masks`` to have any effect.
    write_masks : bool, optional
        If true and ``output_dir`` is given, save the ``rotmask``,
        ``headmask``, ``airmask`` (whichever computed or externally
        supplied) and the harmonized image as NIfTI images under
        ``output_dir`` (default: ``False``).
    rotmask_file : str or None, optional
        Full path to an externally supplied rotation mask (NIfTI), in the
        same (conformed) grid as the resolved ``ref_image``. If omitted, the
        rotmask is computed internally via :func:`_computeRotmaskMortamet`
        (mriqc's Mortamet2009 hard-zero detection) directly on the conformed
        reference image.
    headmask_file : str, optional
        Full path to an externally supplied head mask (NIfTI), in the same
        (conformed) grid as the resolved ``ref_image``. Required: no
        internal computation is currently performed. If omitted or
        unusable, the motion metrics are returned as NaN.
    airmask_file : str, optional
        Full path to an externally supplied air mask (NIfTI), in the same
        (conformed) grid as the resolved ``ref_image``, used for QI2, FBER,
        and the background statistics. Required: no internal computation is
        currently performed. If omitted or unusable, the motion metrics are
        returned as NaN.
    aseg_image : str, optional
        FreeSurfer/FastSurfer ``aseg`` segmentation under ``mri/``, used for
        the gray matter and CSF SNR masks (default: ``aseg.mgz``). Required
        to be present and on the same grid as ``ref_image``.
    aparc_image : str, optional
        FreeSurfer/FastSurfer ``aparc+aseg``-style segmentation under
        ``mri/``, used for the white matter SNR mask and for harmonization
        (default: ``aparc+aseg.mgz``; pass e.g.
        ``aparc.DKTatlas+aseg.deep.mgz`` for FastSurfer output). Required to
        be present and on the same grid as ``ref_image``.
    nb_erode_wm : int, optional
        Erosion (in voxels) applied to the white matter mask used both for
        SNR and for harmonization (default: 3).
    nb_erode_csf : int, optional
        Erosion (in voxels) applied to the CSF SNR mask (default: 1).

    Returns
    -------
    dict
        Dictionary with keys ``efc``, ``qi2``, ``fber``, ``snr_head``
        (SNR over the externally supplied head mask), ``snr_tissue_gm``,
        ``snr_tissue_wm``, ``snr_tissue_csf``, ``snr_tissue_total`` (mean
        over GM/WM/CSF), and BG summary statistics: ``bg_mean``,
        ``bg_median``, ``bg_std``, ``bg_mad``, ``bg_kurtosis``, ``bg_p05``,
        ``bg_p95``, ``bg_n``.
    """
    import logging
    import warnings

    import numpy as np
    from mriqc.qc.anatomical import efc, fber, snr, summary_stats

    def _nan_metrics_dict():
        return {
            "efc": np.nan,
            "qi2": np.nan,
            "fber": np.nan,
            "snr_head": np.nan,
            "snr_tissue_gm": np.nan,
            "snr_tissue_wm": np.nan,
            "snr_tissue_csf": np.nan,
            "snr_tissue_total": np.nan,
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

    if write_masks and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # resolve + conform the reference volume (single grid, shared with
    # aseg/aparc below); bail out with NaNs if it doesn't exist
    ref_path = os.path.join(subjects_dir, subject, "mri", ref_image)
    if not os.path.exists(ref_path):
        warnings.warn(
            "WARNING: could not find reference image " + ref_path
            + " for " + subject + ", returning NaNs.",
            stacklevel=2,
        )
        return _nan_metrics_dict()

    ref_img = _conformImageToRAS(ref_path)
    img_ras = ref_img.get_fdata()

    # headmask/airmask are currently external-file-only (no internal
    # computation); both are required.
    masks = {}
    for mask_name, mask_file in (("headmask", headmask_file), ("airmask", airmask_file)):
        if mask_file is None:
            warnings.warn(
                "WARNING: no " + mask_name + "_file given for " + subject
                + " (currently required, no internal computation), returning NaNs.",
                stacklevel=2,
            )
            return _nan_metrics_dict()
        try:
            masks[mask_name] = _load_external_mask(mask_file, mask_name, img_ras.shape[:3])
            logging.info("Using external " + mask_name + " " + mask_file)
        except Exception as exc:
            warnings.warn(
                "WARNING: could not use external " + mask_name + " " + mask_file
                + " (" + str(exc) + "), returning NaNs.",
                stacklevel=2,
            )
            return _nan_metrics_dict()
    headmask = masks["headmask"]
    airmask = masks["airmask"]

    # rotmask: externally supplied, or computed internally (the one mask
    # still auto-computable)
    if rotmask_file is not None:
        try:
            rotmask = _load_external_mask(rotmask_file, "rotmask", img_ras.shape[:3])
            logging.info("Using external rotmask " + rotmask_file)
        except Exception as exc:
            warnings.warn(
                "WARNING: could not use external rotmask " + rotmask_file
                + " (" + str(exc) + "), returning NaNs.",
                stacklevel=2,
            )
            return _nan_metrics_dict()
    else:
        rotmask = _computeRotmaskMortamet(img_ras)

    if write_masks and output_dir is not None:
        _save_mask_nii(rotmask, ref_img.affine, os.path.join(output_dir, "rotmask.nii.gz"))
        _save_mask_nii(headmask, ref_img.affine, os.path.join(output_dir, "headmask.nii.gz"))
        _save_mask_nii(airmask, ref_img.affine, os.path.join(output_dir, "airmask.nii.gz"))

    # aseg/aparc segmentation, on the same grid as ref_image, required for
    # tissue masks and harmonization
    aseg_path = os.path.join(subjects_dir, subject, "mri", aseg_image)
    aparc_path = os.path.join(subjects_dir, subject, "mri", aparc_image)
    seg = {}
    for seg_name, seg_path in (("aseg", aseg_path), ("aparc", aparc_path)):
        try:
            seg[seg_name] = _load_external_mask(seg_path, seg_name, img_ras.shape[:3], binarize=False)
        except Exception as exc:
            warnings.warn(
                "WARNING: could not use " + seg_name + " " + seg_path
                + " (" + str(exc) + "), returning NaNs.",
                stacklevel=2,
            )
            return _nan_metrics_dict()
    aseg_data = seg["aseg"]
    aparc_data = seg["aparc"]

    gm_mask = _tissue_mask_from_labels(aseg_data, _GM_LABELS, nb_erode=0)
    wm_mask = _tissue_mask_from_labels(aparc_data, _WM_LABELS, nb_erode=nb_erode_wm)
    csf_mask = _tissue_mask_from_labels(aseg_data, _CSF_LABELS, nb_erode=nb_erode_csf)

    # harmonize: rescale so the white-matter mask's median intensity is
    # 1000, mirroring mriqc's in_noinu
    img_harmonized = _harmonizeImage(img_ras, wm_mask)
    if img_harmonized is None:
        warnings.warn(
            "WARNING: could not harmonize reference image for " + subject
            + " (empty or degenerate white-matter mask), returning NaNs.",
            stacklevel=2,
        )
        return _nan_metrics_dict()

    if write_masks and output_dir is not None:
        _save_mask_nii(
            img_harmonized, ref_img.affine, os.path.join(output_dir, "harmonized.nii.gz")
        )

    # one summary_stats() call, on the harmonized image, feeds tissue SNR,
    # head SNR, and background stats
    stats = summary_stats(
        img_harmonized,
        {"gm": gm_mask, "wm": wm_mask, "csf": csf_mask, "head": headmask, "bg": airmask},
    )

    tissue_snr = {}
    for label in ("gm", "wm", "csf"):
        tissue_snr["snr_tissue_" + label] = snr(
            stats[label]["median"], stats[label]["stdv"], stats[label]["n"]
        )
    finite_tissue_snr = [v for v in tissue_snr.values() if np.isfinite(v)]
    tissue_snr["snr_tissue_total"] = (
        float(np.mean(finite_tissue_snr)) if finite_tissue_snr else np.nan
    )

    snr_head_value = snr(stats["head"]["median"], stats["head"]["stdv"], stats["head"]["n"])

    bg_stats = {
        "bg_mean": stats["bg"]["mean"],
        "bg_median": stats["bg"]["median"],
        "bg_std": stats["bg"]["stdv"],
        "bg_mad": stats["bg"]["mad"],
        "bg_kurtosis": stats["bg"]["k"],
        "bg_p05": stats["bg"]["p05"],
        "bg_p95": stats["bg"]["p95"],
        "bg_n": int(stats["bg"]["n"]),
    }

    metrics = {
        "efc": efc(img_harmonized, framemask=rotmask),
        "qi2": _computeQi2(img_ras, airmask=airmask),
        "fber": fber(img_harmonized, headmask=headmask, rotmask=rotmask),
        "snr_head": snr_head_value,
    }
    metrics.update(tissue_snr)
    metrics.update(bg_stats)

    # Summary logging for quick visual sanity-checking in pipeline logs.
    logging.info("EFC: " + f"{metrics['efc']:.4}" if np.isfinite(metrics["efc"]) else "EFC: nan")
    logging.info("QI2: " + f"{metrics['qi2']:.4}" if np.isfinite(metrics["qi2"]) else "QI2: nan")
    logging.info("FBER: " + f"{metrics['fber']:.4}" if np.isfinite(metrics["fber"]) else "FBER: nan")
    logging.info(
        "SNR_HEAD: " + f"{metrics['snr_head']:.4}"
        if np.isfinite(metrics["snr_head"])
        else "SNR_HEAD: nan"
    )
    logging.info(
        "SNR_TISSUE_TOTAL: " + f"{metrics['snr_tissue_total']:.4}"
        if np.isfinite(metrics["snr_tissue_total"])
        else "SNR_TISSUE_TOTAL: nan"
    )

    return metrics