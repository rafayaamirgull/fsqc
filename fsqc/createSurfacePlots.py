"""
This module provides a function to create surface plots
"""
# -----------------------------------------------------------------------------


def createSurfacePlots(SUBJECT, SUBJECTS_DIR, SURFACES_OUTDIR, VIEWS, FASTSURFER):
    """
    Create surface plots.

    Parameters
    ----------
    SUBJECT : str
        The subject.
    SUBJECTS_DIR : str
        The subjects directory.
    SURFACES_OUTDIR : str
        The output directory for surface plots.
    VIEWS : list
        List of views for which surface plots should be created.
    FASTSURFER : bool
        Flag indicating whether FastSurfer processing was used.

    Returns
    -------
    None
        The function returns nothing.
    """
    # imports

    import os

    from whippersnappy import core
    from whippersnappy.types import ViewType

    # -----------------------------------------------------------------------------
    # import surfaces and overlays

    triaPialL = os.path.join(SUBJECTS_DIR, SUBJECT, "surf", "lh.pial")
    triaPialR = os.path.join(SUBJECTS_DIR, SUBJECT, "surf", "rh.pial")
    triaInflL = os.path.join(SUBJECTS_DIR, SUBJECT, "surf", "lh.inflated")
    triaInflR = os.path.join(SUBJECTS_DIR, SUBJECT, "surf", "rh.inflated")

    if FASTSURFER is True:
        annotL = os.path.join(SUBJECTS_DIR, SUBJECT, "label", "lh.aparc.DKTatlas.annot")
        annotR = os.path.join(SUBJECTS_DIR, SUBJECT, "label", "rh.aparc.DKTatlas.annot")
    else:
        annotL = os.path.join(SUBJECTS_DIR, SUBJECT, "label", "lh.aparc.annot")
        annotR = os.path.join(SUBJECTS_DIR, SUBJECT, "label", "rh.aparc.annot")

    # -----------------------------------------------------------------------------
    # plots

    _views_available = [
        'left',
        'right',
        'posterior',
        'anterior',
        'inferior',
        'superior',
    ]

    for view in _views_available:

        fpath_lp = os.path.join(SURFACES_OUTDIR, f"lh.pial.{view}.png")
        fpath_rp = os.path.join(SURFACES_OUTDIR, f"rh.pial.{view}.png")
        fpath_li = os.path.join(SURFACES_OUTDIR, f"lh.inflated.{view}.png")
        fpath_ri = os.path.join(SURFACES_OUTDIR, f"rh.inflated.{view}.png")

        if view in VIEWS:

            if view == "superior":
                wview = ViewType.TOP
            elif view == "inferior":
                wview = ViewType.BOTTOM
            elif view == "anterior":
                wview = ViewType.FRONT
            elif view == "posterior":
                wview = ViewType.BACK
            elif view == "left":
                wview = ViewType.LEFT
            elif view == "right":
                wview = ViewType.RIGHT

            core.snap1(
                meshpath=triaPialL,
                annotpath=annotL,
                outpath=fpath_lp,
                view=wview,
                specular=False
            )
            core.snap1(
                meshpath=triaPialR,
                annotpath=annotR,
                outpath=fpath_rp,
                view=wview,
                specular=False
            )
            core.snap1(
                meshpath=triaInflL,
                annotpath=annotL,
                outpath=fpath_li,
                view=wview,
                specular=False
            )
            core.snap1(
                meshpath=triaInflR,
                annotpath=annotR,
                outpath=fpath_ri,
                view=wview,
                specular=False
            )

        else:
            # remove images potentially created in earlier run but not updated now
            for fpath in [fpath_lp, fpath_rp, fpath_li, fpath_ri]:
                if os.path.isfile(fpath):
                    os.remove(fpath)

    return


