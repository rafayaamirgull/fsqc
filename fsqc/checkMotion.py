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


# -----------------------------------------------------------------------------

# FreeSurferColorLUT label IDs used to build tissue masks for SNR estimation.
# WM/GM lists match fsqc.checkSNR.checkSNR for consistency across the codebase.
_WM_LABELS = [2, 41, 7, 46, 251, 252, 253, 254, 255, 77, 78, 79]
_GM_LABELS = [3, 42]
_CSF_LABELS = [24, 4, 43, 5, 44, 14, 15]


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


def computeConformRotmask(source_img, target_img, threshold=0.5, out_file=None):
    """
    Compute a FreeSurfer-like conform rotmask from source and target images.

    This helper mimics the key conform behavior relevant to rotmask creation:
    resample an all-ones source volume into the target grid and mark voxels that
    fall outside the source field-of-view after resampling.

    Parameters
    ----------
    source_img : nibabel.spatialimages.SpatialImage or str
        Pre-conform (source) image, or a path to load one from.
    target_img : nibabel.spatialimages.SpatialImage or str
        Conformed/resampled (target) image, or a path to load one from.
    threshold : float, optional
        Threshold for in-bounds interpolation values. Values below this
        threshold are treated as outside-FOV.
    out_file : str or None, optional
        If provided, save the resulting rotmask as a NIfTI image to this path.

    Returns
    -------
    numpy.ndarray
        Binary rotmask in target space (1 = resampling padding / outside-FOV).
    """
    import os

    import nibabel as nib
    import numpy as np
    from scipy.ndimage import affine_transform

    if isinstance(source_img, (str, os.PathLike)):
        source_img = nib.load(source_img)
    if isinstance(target_img, (str, os.PathLike)):
        target_img = nib.load(target_img)

    src_shape = source_img.shape[:3]
    tgt_shape = target_img.shape[:3]

    if len(src_shape) != 3 or len(tgt_shape) != 3:
        raise ValueError("source_img and target_img must be 3D volumes")

    # Map target voxel indices to source voxel indices: src = inv(A_src) * A_tgt * tgt
    src_to_world = source_img.affine
    tgt_to_world = target_img.affine
    tgt_to_src = np.linalg.inv(src_to_world) @ tgt_to_world

    ones = np.ones(src_shape, dtype=np.float32)
    sampled = affine_transform(
        ones,
        matrix=tgt_to_src[:3, :3],
        offset=tgt_to_src[:3, 3],
        output_shape=tgt_shape,
        order=0,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )

    rotmask = (sampled < float(threshold)).astype(np.uint8)

    if out_file is not None:
        _save_mask_nii(rotmask, target_img.affine, out_file)

    return rotmask


def computeHeadmaskOtsu(img, rotmask=None, aseg_data=None):
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


def efc(img, framemask=None):
    """
    Compute MRIQC's entropy focus criterion (EFC).

    Lower values are better.
    """
    import numpy as np

    if framemask is None:
        framemask = np.zeros_like(img, dtype=np.uint8)

    valid = framemask == 0
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

    airmask = _airmask_from_headmask(headmask, rotmask=rotmask)
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


def bg(img, headmask, rotmask=None):
    """
    Compute background summary statistics akin to MRIQC's summary_bg_* measures.
    """
    import numpy as np
    from scipy.stats import kurtosis

    airmask = _airmask_from_headmask(headmask, rotmask=rotmask)
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


def checkMotion(
    subjects_dir,
    subject,
    ref_image="orig.mgz",
    output_dir=None,
    write_masks=True, # False # TODO: revert for production
    qi2_airmask_image=None,
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
    ref_image : str, optional
        Reference MRI volume under ``mri/`` (default: ``orig.mgz``).
    output_dir : str or None, optional
        Subject-specific metrics output folder to write debug mask images
        to (e.g. the caller's ``metrics_outdir``). Required for
        ``write_masks`` to have any effect.
    write_masks : bool, optional
        If true and ``output_dir`` is given, save the computed ``rotmask``,
        ``headmask``, and ``airmask`` as NIfTI images under ``output_dir``
        (default: ``False``).
    qi2_airmask_image : str or None, optional
        Optional dedicated air mask under ``mri/`` for QI2. If omitted,
        the complement of the computed ``headmask`` (minus computed ``rotmask``)
        is used.
    aseg_image : str, optional
        FreeSurfer/FastSurfer ``aseg`` segmentation under ``mri/``, used for the
        gray matter and CSF SNR masks (default: ``aseg.mgz``).
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

    logging.captureWarnings(True)
    logging.info("Computing MRIQC-style motion/noise metrics ...")

    # make sure the debug-mask output folder exists before anything tries to write to it
    if write_masks and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # locate and load the reference volume; bail out with NaNs if it's missing
    ref_path = os.path.join(subjects_dir, subject, "mri", ref_image)
    if not os.path.exists(ref_path):
        warnings.warn(
            "WARNING: could not open " + ref_path + ", returning NaNs.",
            stacklevel=2,
        )
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

    ref_img = nib.load(ref_path)
    img = ref_img.get_fdata()

    # Try to build a rotmask by mapping a pre-conform source to ref_image: the
    # rotmask flags voxels in ref_image's grid that fall outside the original
    # (pre-conform) field-of-view, which would otherwise bias the other metrics.
    rotmask = None
    preconform_candidates = [
        os.path.join(subjects_dir, subject, "mri", "rawavg.mgz"),
        os.path.join(subjects_dir, subject, "mri", "orig", "001.mgz"),
    ]
    preconform_source = next((p for p in preconform_candidates if os.path.exists(p)), None)

    rotmask_out_path = None
    if write_masks and output_dir is not None:
        rotmask_out_path = os.path.join(output_dir, "rotmask.nii.gz")

    if preconform_source is not None:
        try:
            rotmask = computeConformRotmask(
                source_img=preconform_source,
                target_img=ref_path,
                out_file=rotmask_out_path,
            )
            logging.info("Computed rotmask from " + preconform_source)
        except Exception as exc:
            warnings.warn(
                "WARNING: could not compute rotmask from "
                + preconform_source
                + " ("
                + str(exc)
                + "), proceeding without rotmask.",
                stacklevel=2,
            )
    else:
        warnings.warn(
            "WARNING: could not find pre-conform source (expected rawavg.mgz "
            "or orig/001.mgz), proceeding without rotmask.",
            stacklevel=2,
        )

    # Load the aseg segmentation early (if available): it's used both to
    # reinforce the headmask below (any already-segmented tissue voxel is
    # unambiguously part of the head) and later for the GM/CSF SNR masks.
    aseg_path = os.path.join(subjects_dir, subject, "mri", aseg_image)
    aseg_data = None
    if os.path.exists(aseg_path):
        aseg_data = nib.load(aseg_path).get_fdata()
    else:
        warnings.warn(
            "WARNING: could not open " + aseg_path
            + ", proceeding without aseg-based headmask reinforcement.",
            stacklevel=2,
        )

    # Compute the head foreground mask via Otsu thresholding on ref_image
    # intensities, excluding any rotmask (outside-FOV) voxels from the estimate.
    headmask = computeHeadmaskOtsu(img, rotmask=rotmask, aseg_data=aseg_data)

    if write_masks and output_dir is not None:
        _save_mask_nii(headmask, ref_img.affine, os.path.join(output_dir, "headmask.nii.gz"))

    # Resolve the airmask used for QI2: either an externally supplied mask, or
    # (the common case) the complement of headmask/rotmask computed above.
    if qi2_airmask_image is not None:
        airmask_path = os.path.join(subjects_dir, subject, "mri", qi2_airmask_image)
        if os.path.exists(airmask_path):
            airmask = nib.load(airmask_path).get_fdata()
        else:
            warnings.warn(
                "WARNING: could not open " + airmask_path + ", using complement of headmask.",
                stacklevel=2,
            )
            airmask = _airmask_from_headmask(headmask, rotmask=rotmask)
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

    # Tissue-based SNR (GM/WM/CSF) additionally requires aseg/aparc segmentations.
    aparc_path = os.path.join(subjects_dir, subject, "mri", aparc_image)
    if aseg_data is not None and os.path.exists(aparc_path):
        aparc_data = nib.load(aparc_path).get_fdata()
        tissue_snr = snr_tissue(
            img,
            aseg_data,
            aparc_data,
            nb_erode_wm=nb_erode_wm,
            nb_erode_csf=nb_erode_csf,
        )
    else:
        warnings.warn(
            "WARNING: could not open " + aseg_path + " and/or " + aparc_path
            + ", returning NaNs for tissue-based SNR.",
            stacklevel=2,
        )
        tissue_snr = {"snr": np.nan, "snr_gm": np.nan, "snr_wm": np.nan, "snr_csf": np.nan}

    # Assemble the final metrics dict: EFC/QI2/FBER/whole-head SNR first, then
    # tissue-based SNR and background summary stats merged in.
    metrics = {
        "efc": efc(img, framemask=rotmask),
        "qi2": qi2(img, airmask),
        "fber": fber(img, headmask, rotmask=rotmask),
        "snr_head": snr_head_value,
    }
    metrics.update(tissue_snr)
    metrics.update(bg(img, headmask, rotmask=rotmask))

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
