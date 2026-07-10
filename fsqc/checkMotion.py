"""
This module provides MRIQC-style image quality metrics related to motion/noise.

Implemented measures:
- EFC  : Entropy Focus Criterion
- QI2  : Mortamet's quality index 2
- FBER : Foreground-Background Energy Ratio
- SNR  : Signal-to-Noise Ratio
- BG   : Background summary statistics
"""


# -----------------------------------------------------------------------------


def _airmask_from_headmask(headmask, rotmask=None):
    """Create an air mask as the complement of the head mask."""
    import numpy as np

    airmask = np.ones_like(headmask, dtype=np.uint8)
    airmask[headmask > 0] = 0
    if rotmask is not None:
        airmask[rotmask > 0] = 0

    return airmask


def computeConformRotmask(source_img, target_img, threshold=0.5):
    """
    Compute a FreeSurfer-like conform rotmask from source and target images.

    This helper mimics the key conform behavior relevant to rotmask creation:
    resample an all-ones source volume into the target grid and mark voxels that
    fall outside the source field-of-view after resampling.

    Parameters
    ----------
    source_img : nibabel.spatialimages.SpatialImage
        Pre-conform (source) image.
    target_img : nibabel.spatialimages.SpatialImage
        Conformed/resampled (target) image.
    threshold : float, optional
        Threshold for in-bounds interpolation values. Values below this
        threshold are treated as outside-FOV.

    Returns
    -------
    numpy.ndarray
        Binary rotmask in target space (1 = resampling padding / outside-FOV).
    """
    import numpy as np
    from scipy.ndimage import affine_transform

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

    return (sampled < float(threshold)).astype(np.uint8)


def computeConformRotmaskFromFiles(source_file, target_file, out_file=None, threshold=0.5):
    """
    Compute a conform rotmask from image files, optionally saving to disk.

    Parameters
    ----------
    source_file : str
        Path to the pre-conform source image.
    target_file : str
        Path to the conformed/resampled target image.
    out_file : str or None, optional
        If provided, save rotmask image to this path.
    threshold : float, optional
        Threshold used by :func:`computeConformRotmask`.

    Returns
    -------
    numpy.ndarray
        Binary rotmask in target space (1 = resampling padding / outside-FOV).
    """
    import os

    import nibabel as nib
    import numpy as np

    src_img = nib.load(source_file)
    tgt_img = nib.load(target_file)

    rotmask = computeConformRotmask(src_img=src_img, target_img=tgt_img, threshold=threshold)

    if out_file is not None:
        _, ext = os.path.splitext(out_file)
        if ext.lower() in [".mgz", ".mgh"]:
            rotmask_img = nib.MGHImage(rotmask.astype(np.float32), tgt_img.affine)
        else:
            rotmask_img = nib.nifti1.Nifti1Image(rotmask.astype("uint8"), tgt_img.affine)
        nib.save(rotmask_img, out_file)

    return rotmask


def computeHeadmaskOtsu(img, rotmask=None):
    """
    Compute a head mask from image intensities using Otsu thresholding.

    Parameters
    ----------
    img : numpy.ndarray
        3D image array.
    rotmask : numpy.ndarray or None, optional
        Optional rotmask where non-zero voxels are excluded from threshold
        estimation and output mask.

    Returns
    -------
    numpy.ndarray
        Binary head mask (uint8, 1 = foreground/head).
    """
    import numpy as np

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


def bg(img, headmask, rotmask=None):
    """
    Compute background summary statistics akin to MRIQC's summary_bg_* measures.
    """
    import numpy as np

    airmask = _airmask_from_headmask(headmask, rotmask=rotmask)
    data = np.nan_to_num(img[airmask > 0], nan=0.0, posinf=0.0, neginf=0.0)
    data = data[data >= 0]

    if data.size == 0:
        return {
            "bg_mean": np.nan,
            "bg_median": np.nan,
            "bg_std": np.nan,
            "bg_mad": np.nan,
            "bg_p05": np.nan,
            "bg_p95": np.nan,
            "bg_n": 0,
        }

    med = np.median(data)
    return {
        "bg_mean": float(np.mean(data)),
        "bg_median": float(med),
        "bg_std": float(np.std(data)),
        "bg_mad": float(np.median(np.abs(data - med))),
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
    from scipy.stats import chi2, gaussian_kde

    np.random.seed(1191935)

    data = np.nan_to_num(img[airmask > 0], nan=0.0, posinf=0.0, neginf=0.0)
    data[data < 0] = 0

    if int((data > 0).sum()) < int(min_voxels):
        return 0.0

    p99 = np.percentile(data, 99)
    if p99 <= 0:
        return np.nan

    data = data * (100.0 / p99)
    modelx = data if len(data) < int(max_voxels) else np.random.choice(data, size=int(max_voxels), replace=False)

    x_grid = np.linspace(0.0, 110.0, 1000)

    # Estimate empirical PDF and fit a chi-square model on the same support.
    kde = gaussian_kde(modelx)
    kde_pdf = kde(x_grid)

    kdethi = int(np.argmax(kde_pdf[::-1] > kde_pdf.max() * 0.5))
    kdethi = max(kdethi, 1)

    params = chi2.fit(modelx, coil_elements)
    chi_pdf = chi2.pdf(x_grid, *params[:-2], loc=params[-2], scale=params[-1])

    return float(np.abs(kde_pdf[-kdethi:] - chi_pdf[-kdethi:]).mean())


def checkMotion(
    subjects_dir,
    subject,
    ref_image="norm.mgz",
    headmask_out_image=None,
    qi2_airmask_image=None,
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
        Reference MRI volume under ``mri/`` (default: ``norm.mgz``).
    headmask_out_image : str or None, optional
        If provided, save the computed Otsu headmask under ``mri/``. Absolute
        paths are also accepted.
    qi2_airmask_image : str or None, optional
        Optional dedicated air mask under ``mri/`` for QI2. If omitted,
        the complement of the computed ``headmask`` (minus computed ``rotmask``)
        is used.

    Returns
    -------
    dict
        Dictionary with keys ``efc``, ``qi2``, ``fber``, ``snr`` and BG summary
        statistics: ``bg_mean``, ``bg_median``, ``bg_std``, ``bg_mad``,
        ``bg_p05``, ``bg_p95``, ``bg_n``.
    """
    import logging
    import os
    import warnings

    import nibabel as nib
    import numpy as np

    logging.captureWarnings(True)
    logging.info("Computing MRIQC-style motion/noise metrics ...")

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
            "bg_mean": np.nan,
            "bg_median": np.nan,
            "bg_std": np.nan,
            "bg_mad": np.nan,
            "bg_p05": np.nan,
            "bg_p95": np.nan,
            "bg_n": 0,
        }

    ref_img = nib.load(ref_path)
    img = ref_img.get_fdata()

    # Try to build a rotmask by mapping a pre-conform source to ref_image.
    rotmask = None
    preconform_candidates = [
        os.path.join(subjects_dir, subject, "mri", "rawavg.mgz"),
        os.path.join(subjects_dir, subject, "mri", "orig", "001.mgz"),
    ]
    preconform_source = next((p for p in preconform_candidates if os.path.exists(p)), None)

    if preconform_source is not None:
        try:
            rotmask = computeConformRotmaskFromFiles(
                source_file=preconform_source,
                target_file=ref_path,
                out_file=None,
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

    headmask = computeHeadmaskOtsu(img, rotmask=rotmask)

    if headmask_out_image is not None:
        if os.path.isabs(headmask_out_image):
            headmask_out_path = headmask_out_image
        else:
            headmask_out_path = os.path.join(subjects_dir, subject, "mri", headmask_out_image)

        _, ext = os.path.splitext(headmask_out_path)
        if ext.lower() in [".mgz", ".mgh"]:
            headmask_img = nib.MGHImage(headmask.astype(np.float32), ref_img.affine)
        else:
            headmask_img = nib.nifti1.Nifti1Image(headmask.astype("uint8"), ref_img.affine)
        nib.save(headmask_img, headmask_out_path)

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

    fg = img[headmask > 0]
    if fg.size == 0:
        snr_value = np.nan
    else:
        snr_value = snr(float(np.median(fg)), float(np.std(fg)), int(fg.size))

    metrics = {
        "efc": efc(img, framemask=rotmask),
        "qi2": qi2(img, airmask),
        "fber": fber(img, headmask, rotmask=rotmask),
        "snr": snr_value,
    }
    metrics.update(bg(img, headmask, rotmask=rotmask))

    logging.info("EFC: " + f"{metrics['efc']:.4}" if np.isfinite(metrics["efc"]) else "EFC: nan")
    logging.info("QI2: " + f"{metrics['qi2']:.4}" if np.isfinite(metrics["qi2"]) else "QI2: nan")
    logging.info("FBER: " + f"{metrics['fber']:.4}" if np.isfinite(metrics["fber"]) else "FBER: nan")
    logging.info("SNR: " + f"{metrics['snr']:.4}" if np.isfinite(metrics["snr"]) else "SNR: nan")

    return metrics
