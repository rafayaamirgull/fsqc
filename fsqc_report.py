#!/usr/bin/env python3
"""Generate cautious, expert-style reports from DeepMI FSQC output.

The program is a research quality-control (QC) aid for FreeSurfer/FastSurfer
reconstructions.  It is deliberately not a diagnostic system: quantitative QC
metrics can prioritize visual review, but they cannot establish pathology or
replace inspection of the source T1 image, segmentation, and surfaces.

Examples
--------
python3 fsqc_report.py output/fsqc-results.csv
python3 fsqc_report.py output/fsqc-results.csv -R
python3 fsqc_report.py output/fsqc-results.csv --subject sub-001
python3 fsqc_report.py output/fsqc-results.csv --save-report fsqc-report.txt
python3 fsqc_report.py output/fsqc-results.csv --profile descriptive

The default ``cautious`` profile applies transparent screening heuristics.  The
cutoffs are not universal biological reference intervals and should be tuned or
validated for the scanner, sequence, software version, and study population.
Use ``--profile descriptive`` to disable these fixed screening cutoffs while
retaining native FSQC outlier flags and descriptive interpretation.

This script uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import os
import shlex
import statistics
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from urllib.parse import quote


REPORT_WIDTH = 96
MIN_COHORT_SIZE = 10


class Severity(IntEnum):
    """Internal priority used to sort findings and determine QC disposition."""

    INFO = 0
    NOTE = 1
    REVIEW = 2
    HIGH = 3


SEVERITY_LABEL = {
    Severity.INFO: "INFO",
    Severity.NOTE: "NOTE",
    Severity.REVIEW: "REVIEW",
    Severity.HIGH: "HIGH",
}


@dataclass(frozen=True)
class Finding:
    """One reportable QC observation."""

    severity: Severity
    domain: str
    title: str
    evidence: str
    interpretation: str
    action: str


@dataclass(frozen=True)
class ScreeningProfile:
    """Transparent study-screening heuristics, not clinical reference ranges."""

    name: str
    enabled: bool
    wm_snr_review: float = 15.0
    wm_snr_high: float = 10.0
    gm_snr_review: float = 10.0
    gm_snr_high: float = 7.0
    contrast_note: float = 4.0
    contrast_review: float = 3.0
    contrast_high: float = 2.5
    cc_lower: float = 0.0025
    cc_upper: float = 0.0080
    holes_note: int = 10
    holes_review: int = 20
    holes_high: int = 40
    defects_note: int = 30
    defects_review: int = 60
    rotation_note_deg: float = 20.0
    rotation_review_deg: float = 30.0
    volume_ai_note_pct: float = 30.0
    volume_ai_review_pct: float = 50.0


PROFILES = {
    "cautious": ScreeningProfile(name="cautious", enabled=True),
    "descriptive": ScreeningProfile(name="descriptive", enabled=False),
}


CORE_METRICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("wm_snr_orig", ("wm_snr_orig",)),
    ("gm_snr_orig", ("gm_snr_orig",)),
    ("wm_snr_norm", ("wm_snr_norm",)),
    ("gm_snr_norm", ("gm_snr_norm",)),
    ("cc_size", ("cc_size",)),
    ("holes_lh", ("holes_lh", "lh_holes")),
    ("holes_rh", ("holes_rh", "rh_holes")),
    ("defects_lh", ("defects_lh", "lh_defects")),
    ("defects_rh", ("defects_rh", "rh_defects")),
    ("topo_lh", ("topo_lh",)),
    ("topo_rh", ("topo_rh",)),
    ("con_snr_lh", ("con_snr_lh", "con_lh_snr")),
    ("con_snr_rh", ("con_snr_rh", "con_rh_snr")),
    ("rot_tal_x", ("rot_tal_x",)),
    ("rot_tal_y", ("rot_tal_y",)),
    ("rot_tal_z", ("rot_tal_z",)),
    ("n_outlier_norms", ("n_outlier_norms",)),
    (
        "n_outlier_sample_nonpar",
        ("n_outlier_sample_nonpar", "n_outliers_sample_nonpar"),
    ),
    (
        "n_outlier_sample_param",
        ("n_outlier_sample_param", "n_outliers_sample_param"),
    ),
)


MRIQC_STYLE_METRICS: tuple[tuple[str, str, str, str], ...] = (
    ("efc", "Entropy Focus Criterion (EFC)", "lower", "harmonized bias-corrected image"),
    ("qi2", "Mortamet quality index 2 (QI2)", "lower", "conformed orig.mgz / air mask"),
    ("fber", "Foreground-background energy ratio (FBER)", "higher", "harmonized image / head mask"),
    ("snr_tissue_total", "Mean tissue SNR (GM/WM/CSF)", "higher", "harmonized tissue masks"),
    ("snr_head", "Head-mask SNR", "higher", "harmonized head mask"),
)


BACKGROUND_METRICS: tuple[tuple[str, str], ...] = (
    ("bg_mean", "Mean"),
    ("bg_median", "Median"),
    ("bg_std", "Standard deviation"),
    ("bg_mad", "Median absolute deviation"),
    ("bg_kurtosis", "Kurtosis"),
    ("bg_p05", "5th percentile"),
    ("bg_p95", "95th percentile"),
    ("bg_n", "Voxel count"),
)


VOLUME_PAIR_ORDER = (
    "Lateral-Ventricle",
    "Inf-Lat-Vent",
    "Hippocampus",
    "Amygdala",
    "Thalamus-Proper",
    "Caudate",
    "Putamen",
    "Pallidum",
    "Accumbens-area",
    "VentralDC",
    "Cerebellum-Cortex",
    "Cerebellum-White-Matter",
)


def safe_float(value: object) -> float | None:
    """Return a finite float, otherwise ``None``."""

    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "na", "n/a", "none", "null"}:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def safe_bool(value: object) -> bool | None:
    """Parse a boolean field used by FSQC detailed outlier tables."""

    if value is None:
        return None
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def get_value(row: Mapping[str, object], *keys: str) -> float | None:
    """Read the first available finite numeric value from aliases."""

    for key in keys:
        if key in row:
            value = safe_float(row[key])
            if value is not None:
                return value
    return None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a comma-separated FSQC table with useful validation errors."""

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError("the file has no CSV header")
            rows = [dict(row) for row in reader]
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    except csv.Error as exc:
        raise ValueError(f"invalid CSV in {path}: {exc}") from exc

    if not rows:
        raise ValueError(f"{path} contains a header but no subject rows")
    return rows


def subject_id(row: Mapping[str, object]) -> str:
    """Get an FSQC subject identifier without converting it to a number."""

    value = row.get("subject") or row.get("Subject") or row.get("subject_id")
    text = str(value).strip() if value is not None else ""
    return text or "Unknown subject"


def row_for_subject(
    rows: Sequence[Mapping[str, str]], wanted_subject: str
) -> Mapping[str, str] | None:
    """Return a matching subject row from a companion table."""

    for row in rows:
        if subject_id(row) == wanted_subject:
            return row
    return None


def shape_columns(row: Mapping[str, object]) -> list[str]:
    """Identify BrainPrint lateral shape-distance columns in an FSQC row."""

    columns: list[str] = []
    for name in row:
        is_subcortical = name.startswith("Left-") and "_Right-" in name
        is_surface = name.startswith("lh-") and "_rh-" in name
        if is_subcortical or is_surface:
            columns.append(name)
    return columns


def friendly_shape_name(column: str) -> str:
    """Convert a BrainPrint column name into a non-duplicated label."""

    surface_names = {
        "lh-white-2d_rh-white-2d": "Cerebral white surface",
        "lh-pial-2d_rh-pial-2d": "Pial surface",
    }
    if column in surface_names:
        return surface_names[column]

    if column.startswith("Left-") and "_Right-" in column:
        left_name, right_name = column.split("_Right-", maxsplit=1)
        left_name = left_name.removeprefix("Left-")
        if left_name == right_name:
            return left_name.replace("-", " ")
        return f"{left_name.replace('-', ' ')} / {right_name.replace('-', ' ')}"
    return column


def friendly_region_name(column: str) -> str:
    """Format a detailed FSQC outlier column for terminal output."""

    prefixes = {"aseg.": "aseg: ", "aparc.lh.": "left cortex: ", "aparc.rh.": "right cortex: "}
    for prefix, label in prefixes.items():
        if column.startswith(prefix):
            name = column.removeprefix(prefix).replace("-", " ")
            return label + name
    return column.replace("-", " ")


def robust_z(value: float, values: Sequence[float]) -> float | None:
    """Compute a median/MAD robust z-score when cohort context is adequate."""

    finite = [item for item in values if math.isfinite(item)]
    if len(finite) < MIN_COHORT_SIZE:
        return None
    median = statistics.median(finite)
    mad = statistics.median(abs(item - median) for item in finite)
    if mad <= 0:
        return None
    return 0.67448975 * (value - median) / mad


def cohort_values(
    rows: Sequence[Mapping[str, object]], keys: Sequence[str]
) -> list[float]:
    """Collect a metric across main-table subjects."""

    values: list[float] = []
    for row in rows:
        value = get_value(row, *keys)
        if value is not None:
            values.append(value)
    return values


def cohort_context(
    value: float | None,
    rows: Sequence[Mapping[str, object]],
    keys: Sequence[str],
) -> str:
    """Return compact robust cohort context for a metric."""

    if value is None:
        return "not available"
    values = cohort_values(rows, keys)
    score = robust_z(value, values)
    if score is None:
        return f"no robust cohort reference (n={len(values)}; need >= {MIN_COHORT_SIZE})"
    return f"robust cohort z={score:+.2f} (n={len(values)})"


def discover_file(base: Path, candidates: Iterable[Path]) -> Path | None:
    """Return the first existing regular file among candidates."""

    for candidate in candidates:
        path = candidate if candidate.is_absolute() else base / candidate
        if path.is_file():
            return path
    return None


def parse_log_metadata(base: Path) -> dict[str, str]:
    """Extract reproducibility metadata from an adjacent FSQC logfile."""

    path = base / "logfile.txt"
    metadata: dict[str, str] = {}
    if not path.is_file():
        return metadata
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                for key in ("Version", "Date", "Command"):
                    prefix = key + ":"
                    if line.startswith(prefix):
                        metadata[key.lower()] = line.removeprefix(prefix).strip()
    except OSError:
        return {}

    command = metadata.get("command", "")
    if "--fastsurfer" in command:
        metadata["pipeline"] = "FastSurfer"
    elif command:
        metadata["pipeline"] = "FreeSurfer"
    if command:
        try:
            arguments = shlex.split(command)
        except ValueError:
            arguments = []
        for flag, key in (
            ("--subjects_dir", "subjects_dir"),
            ("--output_dir", "output_dir"),
        ):
            if flag in arguments:
                index = arguments.index(flag) + 1
                if index < len(arguments):
                    metadata[key] = arguments[index]
    return metadata


FSQC_STATUS_LABELS = {
    0: "OK",
    1: "FAILED",
    2: "NOT REQUESTED",
    3: "SKIPPED / EXISTING OUTPUT REUSED",
}


def _read_text_if_small(path: Path, limit: int = 8_000_000) -> str:
    """Read a log only when it is reasonably sized for a QC status scan."""

    try:
        if not path.is_file() or path.stat().st_size > limit:
            return ""
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _read_fsqc_status(base: Path, sid: str) -> dict[str, int]:
    """Read FSQC's subject-level module status table."""

    path = base / "status" / sid / "status.txt"
    statuses: dict[str, int] = {}
    if not path.is_file():
        return statuses
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" not in line:
                continue
            module, raw_status = line.split(":", maxsplit=1)
            if module == "subject":
                continue
            try:
                statuses[module.strip()] = int(raw_status.strip())
            except ValueError:
                continue
    except OSError:
        return {}
    return statuses


def inspect_processing_integrity(
    base: Path, sid: str, metadata: Mapping[str, str]
) -> dict[str, object]:
    """Collect QATools-style completion and required-output evidence.

    The legacy QATools workflow checked processing status, output existence, aseg
    outliers, SNR/WM measures, and snapshots.  This function implements the
    defensible status/file-presence subset for modern FSQC output.  Presence is
    necessary but not sufficient for anatomical accuracy.
    """

    statuses = _read_fsqc_status(base, sid)
    result: dict[str, object] = {
        "available": False,
        "subject_dir": None,
        "fsqc_statuses": statuses,
        "completion_markers": [],
        "completion_logs": [],
        "error_markers": [],
        "required_outputs": [],
        "missing_outputs": [],
        "status": "SOURCE SUBJECT DIRECTORY NOT AVAILABLE",
    }
    subjects_dir_text = metadata.get("subjects_dir")
    if not subjects_dir_text:
        return result

    subjects_dir = Path(subjects_dir_text).expanduser()
    if not subjects_dir.is_absolute():
        subjects_dir = (Path.cwd() / subjects_dir).resolve()
    subject_dir = subjects_dir / sid
    result["subject_dir"] = subject_dir
    if not subject_dir.is_dir():
        result["status"] = "SOURCE SUBJECT DIRECTORY NOT FOUND"
        return result
    result["available"] = True

    scripts = subject_dir / "scripts"
    marker_candidates = (
        scripts / "recon-all.done",
        scripts / "recon-surf.done",
    )
    completion_markers = [path for path in marker_candidates if path.exists()]
    error_markers = [path for path in scripts.glob("*.error") if path.is_file()]

    completion_logs: list[Path] = []
    for path in (scripts / "recon-all-status.log", scripts / "recon-surf.log"):
        content = _read_text_if_small(path).lower()
        if "finished without error" in content:
            completion_logs.append(path)

    required = [
        ("Conformed T1", Path("mri/orig.mgz")),
        ("Normalized T1", Path("mri/norm.mgz")),
        ("Brain mask", Path("mri/brainmask.mgz")),
        ("Subcortical segmentation", Path("mri/aseg.mgz")),
        ("Subcortical statistics", Path("stats/aseg.stats")),
        ("Left white surface", Path("surf/lh.white")),
        ("Right white surface", Path("surf/rh.white")),
        ("Left pial surface", Path("surf/lh.pial")),
        ("Right pial surface", Path("surf/rh.pial")),
        ("Left cortical thickness", Path("surf/lh.thickness")),
        ("Right cortical thickness", Path("surf/rh.thickness")),
        ("Left spherical registration", Path("surf/lh.sphere.reg")),
        ("Right spherical registration", Path("surf/rh.sphere.reg")),
    ]
    if metadata.get("pipeline") == "FastSurfer":
        required.extend(
            (
                ("FastSurfer parcellation", Path("mri/aparc.DKTatlas+aseg.mapped.mgz")),
                ("Left cortical statistics", Path("stats/lh.aparc.DKTatlas.mapped.stats")),
                ("Right cortical statistics", Path("stats/rh.aparc.DKTatlas.mapped.stats")),
            )
        )
    else:
        required.extend(
            (
                ("Cortical parcellation", Path("mri/aparc+aseg.mgz")),
                ("Left cortical statistics", Path("stats/lh.aparc.stats")),
                ("Right cortical statistics", Path("stats/rh.aparc.stats")),
            )
        )

    output_rows = [
        (label, relative, (subject_dir / relative).exists()) for label, relative in required
    ]
    missing = [(label, relative) for label, relative, exists in output_rows if not exists]
    result.update(
        {
            "completion_markers": completion_markers,
            "completion_logs": completion_logs,
            "error_markers": error_markers,
            "required_outputs": output_rows,
            "missing_outputs": missing,
        }
    )
    if error_markers or missing:
        result["status"] = "INCOMPLETE / ERROR EVIDENCE - INVESTIGATE"
    elif completion_markers or completion_logs:
        result["status"] = "COMPLETION EVIDENCE AND REQUIRED OUTPUTS PRESENT"
    else:
        result["status"] = "OUTPUTS PRESENT; COMPLETION MARKER NOT CONFIRMED"
    return result


def discover_visuals(base: Path, sid: str) -> list[tuple[str, Path]]:
    """Find companion images that can support the recommended visual review."""

    candidates = (
        ("segmentation/surface screenshot", base / "screenshots" / sid / f"{sid}.png"),
        ("skull-strip screenshot", base / "skullstrip" / sid / f"{sid}.png"),
        ("corpus-callosum/fornix screenshot", base / "fornix" / sid / "cc.png"),
    )
    found = [(label, path) for label, path in candidates if path.is_file()]
    surface_dir = base / "surfaces" / sid
    if surface_dir.is_dir() and any(surface_dir.glob("*.png")):
        found.append(("surface renderings", surface_dir))
    return found


def load_optional_table(path: Path | None) -> list[dict[str, str]]:
    """Read an optional table; absent files produce an empty list."""

    if path is None:
        return []
    try:
        return read_csv_rows(path)
    except ValueError:
        return []


def detailed_outliers(
    rows: Sequence[Mapping[str, str]], sid: str
) -> list[str]:
    """Return region names whose detailed outlier flag is true."""

    row = row_for_subject(rows, sid)
    if row is None:
        return []
    return [
        name
        for name, value in row.items()
        if name != "subject" and safe_bool(value) is True
    ]


def volume_pairs(region_row: Mapping[str, object] | None) -> list[dict[str, object]]:
    """Calculate real bilateral volume asymmetry from ``all.regions.stats``.

    AI is the signed asymmetry index ``200 * (L - R) / (L + R)``.  This is
    distinct from the BrainPrint shape distance in the main FSQC result table.
    """

    if region_row is None:
        return []
    results: list[dict[str, object]] = []
    for structure in VOLUME_PAIR_ORDER:
        left = get_value(region_row, f"aseg.Left-{structure}")
        right = get_value(region_row, f"aseg.Right-{structure}")
        if left is None or right is None or left < 0 or right < 0:
            continue
        total = left + right
        ai = 200.0 * (left - right) / total if total > 0 else None
        ratio = max(left, right) / min(left, right) if min(left, right) > 0 else None
        if ai is None:
            direction = "undefined"
        elif abs(ai) < 0.05:
            direction = "approximately equal"
        elif ai > 0:
            direction = "left larger"
        else:
            direction = "right larger"
        results.append(
            {
                "structure": structure.replace("-", " "),
                "left": left,
                "right": right,
                "ai": ai,
                "ratio": ratio,
                "direction": direction,
            }
        )
    return results


class FSQCAnalyzer:
    """Interpret one subject using FSQC metrics and explicit QC boundaries."""

    def __init__(
        self,
        row: Mapping[str, object],
        all_rows: Sequence[Mapping[str, object]],
        profile: ScreeningProfile,
        region_row: Mapping[str, object] | None = None,
        normative_detail: Sequence[str] = (),
        nonparam_detail: Sequence[str] = (),
        param_detail: Sequence[str] = (),
        processing_integrity: Mapping[str, object] | None = None,
    ) -> None:
        self.row = row
        self.all_rows = all_rows
        self.profile = profile
        self.region_row = region_row
        self.sid = subject_id(row)
        self.normative_detail = list(normative_detail)
        self.nonparam_detail = list(nonparam_detail)
        self.param_detail = list(param_detail)
        self.processing_integrity = dict(processing_integrity or {})
        self.findings: list[Finding] = []

    def value(self, canonical: str) -> float | None:
        """Read a canonical core metric using its accepted aliases."""

        for name, aliases in CORE_METRICS:
            if name == canonical:
                return get_value(self.row, *aliases)
        return get_value(self.row, canonical)

    def add(
        self,
        severity: Severity,
        domain: str,
        title: str,
        evidence: str,
        interpretation: str,
        action: str,
    ) -> None:
        """Append a structured finding."""

        self.findings.append(
            Finding(severity, domain, title, evidence, interpretation, action)
        )

    def analyze(self) -> dict[str, object]:
        """Run all analysis domains and return a report model."""

        self._analyze_processing_integrity()
        self._analyze_signal()
        self._analyze_mriqc_style_metrics()
        self._analyze_topology()
        self._analyze_alignment()
        self._analyze_volume_asymmetry()
        self._analyze_shape()
        self._analyze_outliers()
        self._analyze_cohort_extremes()

        available = 0
        missing: list[str] = []
        for canonical, aliases in CORE_METRICS:
            if get_value(self.row, *aliases) is None:
                missing.append(canonical)
            else:
                available += 1

        return {
            "subject": self.sid,
            "findings": sorted(
                self.findings,
                key=lambda item: (-int(item.severity), item.domain, item.title),
            ),
            "disposition": self._disposition(available),
            "available_core": available,
            "total_core": len(CORE_METRICS),
            "missing_core": missing,
            "volume_pairs": volume_pairs(self.region_row),
            "shape_metrics": self._shape_metrics(),
            "processing_integrity": self.processing_integrity,
        }

    def _analyze_processing_integrity(self) -> None:
        """Flag missing outputs or explicit processing/module failures."""

        integrity = self.processing_integrity
        if not integrity:
            return
        failed_modules = [
            module
            for module, status in dict(integrity.get("fsqc_statuses", {})).items()
            if status == 1
        ]
        error_markers = list(integrity.get("error_markers", []))
        missing_outputs = list(integrity.get("missing_outputs", []))
        if failed_modules or error_markers:
            evidence_parts: list[str] = []
            if failed_modules:
                evidence_parts.append("failed FSQC modules: " + ", ".join(failed_modules))
            if error_markers:
                evidence_parts.append(
                    "error markers: " + ", ".join(Path(path).name for path in error_markers)
                )
            self.add(
                Severity.HIGH,
                "Processing integrity",
                "Explicit processing failure evidence",
                "; ".join(evidence_parts),
                "A failed QC module or reconstruction error marker makes quantitative output "
                "completeness uncertain even when some downstream files exist.",
                "Review processing logs, correct the failure, rerun the affected stages, and "
                "regenerate QC before accepting measurements.",
            )
        if missing_outputs:
            names = ", ".join(str(label) for label, _ in missing_outputs)
            self.add(
                Severity.HIGH,
                "Processing integrity",
                "Required reconstruction outputs are missing",
                names,
                "One or more products required for the reported morphometry could not be found. "
                "This is a completeness problem, separate from image quality.",
                "Confirm the intended pipeline completed and recreate missing outputs before "
                "downstream analysis.",
            )
        elif integrity.get("available") and not (
            integrity.get("completion_markers") or integrity.get("completion_logs")
        ):
            self.add(
                Severity.REVIEW,
                "Processing integrity",
                "Completion marker not confirmed",
                "required outputs were found, but no recognized completion marker/log phrase was found",
                "File presence supports completeness but does not establish that the reconstruction "
                "terminated normally.",
                "Inspect recon-all/recon-surf status logs and document the processing state.",
            )

    def _analyze_signal(self) -> None:
        wm_norm = self.value("wm_snr_norm")
        gm_norm = self.value("gm_snr_norm")
        con_lh = self.value("con_snr_lh")
        con_rh = self.value("con_snr_rh")

        if self.profile.enabled and wm_norm is not None:
            if wm_norm < self.profile.wm_snr_high:
                level = Severity.HIGH
            elif wm_norm < self.profile.wm_snr_review:
                level = Severity.REVIEW
            else:
                level = Severity.INFO
            if level > Severity.INFO:
                self.add(
                    level,
                    "Signal and contrast",
                    "Low normalized white-matter SNR screening value",
                    f"wm_snr_norm={wm_norm:.3f}; review bound <{self.profile.wm_snr_review:g}",
                    "Lower within-WM intensity homogeneity can accompany acquisition artifact, "
                    "intensity inhomogeneity, or segmentation contamination; the metric is not a "
                    "disease marker.",
                    "Inspect the source T1, norm.mgz, and WM labels before using WM-derived measures.",
                )

        if self.profile.enabled and gm_norm is not None:
            if gm_norm < self.profile.gm_snr_high:
                level = Severity.HIGH
            elif gm_norm < self.profile.gm_snr_review:
                level = Severity.REVIEW
            else:
                level = Severity.INFO
            if level > Severity.INFO:
                self.add(
                    level,
                    "Signal and contrast",
                    "Low normalized gray-matter SNR screening value",
                    f"gm_snr_norm={gm_norm:.3f}; review bound <{self.profile.gm_snr_review:g}",
                    "This suggests reduced homogeneity within the FSQC cortical GM mask. It can "
                    "reflect image quality, partial volume, or segmentation behavior and has no "
                    "specific clinical interpretation by itself.",
                    "Review cortical GM labeling and the source image for motion, bias field, and "
                    "gray/white boundary definition.",
                )

        contrasts = [item for item in (con_lh, con_rh) if item is not None]
        if self.profile.enabled and contrasts:
            minimum = min(contrasts)
            if minimum < self.profile.contrast_high:
                level = Severity.HIGH
            elif minimum < self.profile.contrast_review:
                level = Severity.REVIEW
            elif minimum < self.profile.contrast_note:
                level = Severity.NOTE
            else:
                level = Severity.INFO
            if level > Severity.INFO:
                values = f"LH={con_lh:.3f}" if con_lh is not None else "LH=missing"
                values += f", RH={con_rh:.3f}" if con_rh is not None else ", RH=missing"
                self.add(
                    level,
                    "Signal and contrast",
                    "Reduced gray/white contrast consistency",
                    values,
                    "The surface-sampled contrast distribution is less separated or more "
                    "variable under the selected screening heuristic. This does not identify the "
                    "cause and cannot imply a particular disorder.",
                    "Inspect white and pial boundary placement, especially temporal poles, insula, "
                    "orbitofrontal cortex, and areas affected by intensity non-uniformity.",
                )

    def _analyze_mriqc_style_metrics(self) -> None:
        """Flag adverse cohort extremes among FSQC's MRIQC-style IQMs.

        MRIQC describes these as no-reference metrics. We therefore avoid
        universal absolute cutoffs and only flag robust extremes when at least
        ten compatible subjects are present in the input table.
        """

        for key, label, preferred_direction, _ in MRIQC_STYLE_METRICS:
            value = get_value(self.row, key)
            if value is None:
                continue
            values = cohort_values(self.all_rows, (key,))
            score = robust_z(value, values)
            if score is None:
                continue
            adverse = score >= 3.5 if preferred_direction == "lower" else score <= -3.5
            if not adverse:
                continue
            direction_text = (
                "lower values are preferred"
                if preferred_direction == "lower"
                else "higher values are preferred"
            )
            self.add(
                Severity.REVIEW,
                "MRIQC-style image quality",
                f"Cohort-extreme {label}",
                f"{key}={value:.4g}; robust cohort z={score:+.2f}; n={len(values)}",
                f"This FSQC-adapted no-reference IQM is adverse relative to the supplied cohort; "
                f"{direction_text}. It prioritizes artifact review but does not identify a cause "
                "or define clinical normality.",
                "Compare masks and images across a scanner/protocol/software-matched cohort, then "
                "inspect the source T1 for the artifact pattern suggested by this metric.",
            )

        for key in ("bg_std", "bg_mad", "bg_p95"):
            value = get_value(self.row, key)
            if value is None:
                continue
            values = cohort_values(self.all_rows, (key,))
            score = robust_z(value, values)
            if score is not None and score >= 3.5:
                self.add(
                    Severity.REVIEW,
                    "MRIQC-style image quality",
                    f"Cohort-extreme background statistic: {key}",
                    f"{key}={value:.4g}; robust cohort z={score:+.2f}; n={len(values)}",
                    "An unusually broad or high-tailed background distribution may reflect noise, "
                    "ghosting, leakage, or mask behavior. This descriptive statistic is not specific.",
                    "Inspect the air/head/rotation masks and background around the head, then compare "
                    "with matched acquisitions.",
                )

    def _analyze_topology(self) -> None:
        holes_lh = self.value("holes_lh")
        holes_rh = self.value("holes_rh")
        defects_lh = self.value("defects_lh")
        defects_rh = self.value("defects_rh")
        cc_size = self.value("cc_size")

        holes = [item for item in (holes_lh, holes_rh) if item is not None]
        if self.profile.enabled and holes:
            total = sum(holes)
            if total >= self.profile.holes_high:
                level = Severity.HIGH
            elif total >= self.profile.holes_review:
                level = Severity.REVIEW
            elif total >= self.profile.holes_note:
                level = Severity.NOTE
            else:
                level = Severity.INFO
            if level > Severity.INFO:
                self.add(
                    level,
                    "Surface reconstruction",
                    "Elevated initial surface hole burden",
                    f"total={total:.0f}; LH={_fmt(holes_lh, 0)}, RH={_fmt(holes_rh, 0)}",
                    "FSQC counts holes on orig.nofix, before FreeSurfer's topology correction. "
                    "A high count indicates a larger correction burden; it does not mean the final "
                    "surface still contains that many holes or that spherical mapping failed.",
                    "Inspect final white/pial surfaces and the medial wall; examine orig.nofix/defect "
                    "locations if boundary errors are visible.",
                )

        defects = [item for item in (defects_lh, defects_rh) if item is not None]
        if self.profile.enabled and defects:
            total = sum(defects)
            if total >= self.profile.defects_review:
                level = Severity.REVIEW
            elif total >= self.profile.defects_note:
                level = Severity.NOTE
            else:
                level = Severity.INFO
            if level > Severity.INFO:
                self.add(
                    level,
                    "Surface reconstruction",
                    "Elevated automatically corrected topology defects",
                    f"total={total:.0f}; LH={_fmt(defects_lh, 0)}, RH={_fmt(defects_rh, 0)}",
                    "The recon-all log recorded an increased number of defects presented to topology "
                    "correction. Counts measure processing burden, not the anatomical extent or the "
                    "accuracy of the final surfaces.",
                    "Review inflated/pial renderings and volumetric overlays for residual geometric "
                    "errors before cortical morphometry.",
                )

        if (
            self.profile.enabled
            and cc_size is not None
            and not self.profile.cc_lower <= cc_size <= self.profile.cc_upper
        ):
            self.add(
                Severity.REVIEW,
                "Segmentation",
                "Corpus-callosum fraction outside screening bounds",
                f"cc_size={cc_size:.6f}; heuristic interval "
                f"{self.profile.cc_lower:.4f}-{self.profile.cc_upper:.4f} of eTIV",
                "cc_size is the summed FreeSurfer corpus-callosum label volume divided by eTIV. "
                "An unusual value can reflect segmentation, anatomy, or eTIV estimation; it cannot "
                "distinguish these explanations.",
                "Inspect the sagittal corpus-callosum labels and the fornix-focused companion image.",
            )

    def _analyze_alignment(self) -> None:
        rotations = {
            "x": self.value("rot_tal_x"),
            "y": self.value("rot_tal_y"),
            "z": self.value("rot_tal_z"),
        }
        present = [abs(value) for value in rotations.values() if value is not None]
        if not self.profile.enabled or not present:
            return
        maximum_deg = math.degrees(max(present))
        if maximum_deg >= self.profile.rotation_review_deg:
            level = Severity.REVIEW
        elif maximum_deg >= self.profile.rotation_note_deg:
            level = Severity.NOTE
        else:
            level = Severity.INFO
        if level > Severity.INFO:
            evidence = ", ".join(
                f"{axis}={math.degrees(value):+.1f} deg"
                for axis, value in rotations.items()
                if value is not None
            )
            self.add(
                level,
                "Spatial normalization",
                "Large Talairach rotation component",
                evidence,
                "These are Euler components decomposed from talairach.lta. They describe the "
                "rotation used for atlas alignment; they are not a direct head-motion measure and "
                "do not prove registration failure.",
                "Check talairach.xfm/lta alignment and source-image orientation. Investigate header "
                "or acquisition positioning if the overlay is implausible.",
            )

    def _analyze_volume_asymmetry(self) -> None:
        pairs = volume_pairs(self.region_row)
        if not self.profile.enabled or not pairs:
            return
        for pair in pairs:
            ai = pair["ai"]
            if not isinstance(ai, float):
                continue
            magnitude = abs(ai)
            if magnitude >= self.profile.volume_ai_review_pct:
                level = Severity.REVIEW
            elif magnitude >= self.profile.volume_ai_note_pct:
                level = Severity.NOTE
            else:
                continue
            self.add(
                level,
                "Bilateral volumes",
                f"Marked raw volume asymmetry: {pair['structure']}",
                f"L={pair['left']:.1f} mm^3, R={pair['right']:.1f} mm^3, "
                f"AI={ai:+.1f}%, larger/smaller={pair['ratio']:.2f}",
                "The signed AI is descriptive and region-specific normal ranges are not supplied. "
                "Asymmetry may be anatomical, segmentation-related, or influenced by pathology; "
                "the metric alone cannot decide among these possibilities.",
                "Inspect both labels on the T1 image and compare against a matched, quality-controlled "
                "reference cohort before biological interpretation.",
            )

    def _shape_metrics(self) -> list[dict[str, object]]:
        metrics: list[dict[str, object]] = []
        for column in shape_columns(self.row):
            value = get_value(self.row, column)
            if value is None:
                continue
            values = cohort_values(self.all_rows, (column,))
            metrics.append(
                {
                    "column": column,
                    "structure": friendly_shape_name(column),
                    "value": value,
                    "robust_z": robust_z(value, values),
                    "cohort_n": len(values),
                }
            )
        return metrics

    def _analyze_shape(self) -> None:
        for metric in self._shape_metrics():
            score = metric["robust_z"]
            if isinstance(score, float) and score >= 3.5:
                self.add(
                    Severity.REVIEW,
                    "BrainPrint shape",
                    f"Cohort-extreme lateral shape distance: {metric['structure']}",
                    f"distance={metric['value']:.3f}, robust cohort z={score:+.2f}, "
                    f"n={metric['cohort_n']}",
                    "BrainPrint measures a distance between left and right spectral shape "
                    "descriptors. Larger distance means less shape similarity; it is directionless "
                    "and is not a volume ratio.",
                    "Verify bilateral segmentations and interpret only against a compatible cohort "
                    "processed with the same software and settings.",
                )

    def _analyze_outliers(self) -> None:
        configurations = (
            (
                "n_outlier_norms",
                self.normative_detail,
                "FSQC normative-reference outlier",
                "FSQC's configured normative bounds",
            ),
            (
                "n_outlier_sample_nonpar",
                self.nonparam_detail,
                "Non-parametric sample outlier",
                "the 1.5-IQR sample rule",
            ),
            (
                "n_outlier_sample_param",
                self.param_detail,
                "Parametric sample outlier",
                "the +/-2-SD sample rule",
            ),
        )
        for metric, details, title, method in configurations:
            count = self.value(metric)
            if count is None or count <= 0:
                continue
            detail_text = ", ".join(friendly_region_name(name) for name in details)
            evidence = f"count={count:.0f}"
            if detail_text:
                evidence += f"; flagged: {detail_text}"
            self.add(
                Severity.REVIEW,
                "Statistical outliers",
                title,
                evidence,
                f"One or more segmentation-derived measurements fell outside {method}. "
                "Outlier status is a review trigger, not evidence of disease and not proof of a "
                "segmentation error.",
                "Inspect each flagged label and confirm acquisition/software/reference compatibility "
                "before retaining or excluding the measurement.",
            )

    def _analyze_cohort_extremes(self) -> None:
        """Add cohort flags for core metrics not already covered by native outliers."""

        directions = {
            "wm_snr_orig": "two-sided",
            "gm_snr_orig": "two-sided",
            "wm_snr_norm": "two-sided",
            "gm_snr_norm": "two-sided",
            "cc_size": "two-sided",
            "holes_lh": "high",
            "holes_rh": "high",
            "defects_lh": "high",
            "defects_rh": "high",
            "topo_lh": "high",
            "topo_rh": "high",
            "con_snr_lh": "two-sided",
            "con_snr_rh": "two-sided",
        }
        existing_evidence = " ".join(item.evidence for item in self.findings)
        for canonical, aliases in CORE_METRICS:
            if canonical not in directions:
                continue
            value = get_value(self.row, *aliases)
            if value is None:
                continue
            score = robust_z(value, cohort_values(self.all_rows, aliases))
            if score is None:
                continue
            extreme = abs(score) >= 3.5 if directions[canonical] == "two-sided" else score >= 3.5
            if not extreme or f"{canonical}=" in existing_evidence:
                continue
            self.add(
                Severity.REVIEW,
                "Cohort comparison",
                f"Cohort-extreme core metric: {canonical}",
                f"{canonical}={value:.4g}; robust cohort z={score:+.2f}; "
                f"n={len(cohort_values(self.all_rows, aliases))}",
                "The value is unusual within this input table under a median/MAD rule. A batch, "
                "site, age, or biological effect can also produce cohort differences.",
                "Check the scan and processing output, then model relevant cohort/site covariates "
                "rather than treating this flag as an automatic exclusion.",
            )

    def _disposition(self, available: int) -> str:
        high = sum(item.severity == Severity.HIGH for item in self.findings)
        review = sum(item.severity == Severity.REVIEW for item in self.findings)
        if available < len(CORE_METRICS) / 2:
            return "INCOMPLETE METRICS - QC DETERMINATION LIMITED"
        if high or review >= 2:
            return "HIGH-PRIORITY MANUAL QC - HOLD AUTOMATED MORPHOMETRY"
        if review == 1:
            return "TARGETED MANUAL QC REQUIRED"
        if any(item.severity == Severity.NOTE for item in self.findings):
            return "MANUAL QC RECOMMENDED - QUANTITATIVE NOTES PRESENT"
        return "NO QUANTITATIVE RED FLAGS - ROUTINE VISUAL QC STILL REQUIRED"


def _fmt(value: float | None, decimals: int = 3) -> str:
    """Format optional numbers consistently."""

    return "not available" if value is None else f"{value:.{decimals}f}"


def _percent_change(original: float | None, normalized: float | None) -> str:
    """Describe normalization-associated metric change without calling it quality gain."""

    if original is None or normalized is None or original == 0:
        return "not available"
    change = 100.0 * (normalized - original) / abs(original)
    return f"{change:+.1f}%"


def _metric_line(label: str, value: str, context: str = "") -> str:
    """Render one aligned metric row."""

    line = f"  {label:<35} {value:>16}"
    if context:
        line += f"   {context}"
    return line


def _wrapped_bullet(text: str, indent: int = 2, marker: str = "-") -> list[str]:
    """Wrap a bullet while preserving a clean terminal layout."""

    prefix = " " * indent + marker + " "
    return textwrap.wrap(
        text,
        width=REPORT_WIDTH,
        initial_indent=prefix,
        subsequent_indent=" " * len(prefix),
        break_long_words=False,
        break_on_hyphens=False,
    ) or [prefix.rstrip()]


def _section(lines: list[str], number: int, title: str) -> None:
    """Append a numbered report section header."""

    lines.extend(("", f"{number}. {title}", "-" * REPORT_WIDTH))


def review_actions(findings: Sequence[Finding]) -> list[str]:
    """Build a de-duplicated, finding-aware human review checklist."""

    actions: list[str] = []
    for finding in findings:
        if finding.severity >= Severity.NOTE and finding.action not in actions:
            actions.append(finding.action)
    actions.extend(
        action
        for action in (
            "Inspect the native/conformed T1 for motion, ringing, bias field, signal dropout, "
            "clipping, and incomplete coverage.",
            "Inspect brainmask/skull strip, aseg labels, and white/pial surfaces in all three "
            "planes; do not rely on a single screenshot.",
            "Record the final human QC decision and rationale. Reprocess correctable reconstruction "
            "errors; consider reacquisition only when the source acquisition is inadequate and a "
            "new scan is feasible.",
        )
        if action not in actions
    )
    return actions


def format_report(
    model: Mapping[str, object],
    analyzer: FSQCAnalyzer,
    source_path: Path,
    regions_path: Path | None,
    metadata: Mapping[str, str],
    visuals: Sequence[tuple[str, Path]],
) -> str:
    """Render a detailed, self-contained plain-text QC report."""

    row = analyzer.row
    findings = list(model["findings"])
    high_count = sum(item.severity == Severity.HIGH for item in findings)
    review_count = sum(item.severity == Severity.REVIEW for item in findings)
    note_count = sum(item.severity == Severity.NOTE for item in findings)
    generated = datetime.now().astimezone().isoformat(timespec="seconds")

    lines = [
        "=" * REPORT_WIDTH,
        "AUTOMATED FREESURFER / FASTSURFER QUALITY-CONTROL REPORT",
        "=" * REPORT_WIDTH,
        f"Subject ID        : {model['subject']}",
        f"Generated         : {generated}",
        f"Input             : {source_path}",
        f"Processing stream : {metadata.get('pipeline', 'not identified')}",
        f"FSQC version      : {metadata.get('version', 'not available')}",
        f"Screening profile : {analyzer.profile.name}",
        "",
        f"QC DISPOSITION    : {model['disposition']}",
        f"Priority findings : HIGH={high_count}, REVIEW={review_count}, NOTE={note_count}",
        f"Metric coverage   : {model['available_core']}/{model['total_core']} core fields available",
        "",
        "SCOPE: Research reconstruction QC only. This report does not diagnose disease, determine",
        "clinical normality, or replace review by a trained image analyst/radiologist. A quantitative",
        "flag is a reason to inspect the images, not an automatic reason to exclude or rescan.",
    ]

    _section(lines, 1, "EXECUTIVE INTERPRETATION")
    if findings:
        priority = [item for item in findings if item.severity >= Severity.NOTE]
        for item in priority:
            summary = (
                f"[{SEVERITY_LABEL[item.severity]}] {item.title}. {item.evidence}. "
                f"{item.interpretation}"
            )
            lines.extend(_wrapped_bullet(summary))
    else:
        lines.extend(
            _wrapped_bullet(
                "No quantitative flags were generated. This is not equivalent to a visual QC pass; "
                "localized surface, skull-strip, or segmentation errors may not be captured by the "
                "summary metrics."
            )
        )

    _section(lines, 2, "PROCESSING INTEGRITY AND COMPLETENESS")
    integrity = dict(model.get("processing_integrity", {}))
    if integrity.get("available"):
        required_outputs = list(integrity.get("required_outputs", []))
        present_count = sum(bool(item[2]) for item in required_outputs)
        lines.append(_metric_line("Integrity status", str(integrity.get("status", "not available"))))
        lines.append(_metric_line("Required products present", f"{present_count}/{len(required_outputs)}"))
        lines.append(
            _metric_line(
                "Completion markers/logs",
                str(
                    len(integrity.get("completion_markers", []))
                    + len(integrity.get("completion_logs", []))
                ),
            )
        )
        lines.append(_metric_line("Error marker files", str(len(integrity.get("error_markers", [])))))
        statuses = dict(integrity.get("fsqc_statuses", {}))
        if statuses:
            lines.append("  FSQC module states:")
            for module, status in statuses.items():
                lines.append(
                    _metric_line(module.replace("_", " ").title(), FSQC_STATUS_LABELS.get(status, f"UNKNOWN ({status})"))
                )
        missing_outputs = list(integrity.get("missing_outputs", []))
        if missing_outputs:
            lines.extend(
                _wrapped_bullet(
                    "Missing required products: "
                    + ", ".join(f"{label} ({relative})" for label, relative in missing_outputs)
                )
            )
    else:
        lines.extend(
            _wrapped_bullet(
                str(integrity.get("status", "Source subject directory was not available for inspection."))
            )
        )
    lines.extend(
        _wrapped_bullet(
            "QATools-style completion checks establish whether expected processing evidence exists; "
            "they do not establish anatomical correctness. File creation-order checks are not used "
            "because modern parallel pipelines can make timestamp ordering misleading."
        )
    )

    _section(lines, 3, "SIGNAL, INTENSITY NORMALIZATION, AND TISSUE CONTRAST")
    wm_orig = analyzer.value("wm_snr_orig")
    gm_orig = analyzer.value("gm_snr_orig")
    wm_norm = analyzer.value("wm_snr_norm")
    gm_norm = analyzer.value("gm_snr_norm")
    con_lh = analyzer.value("con_snr_lh")
    con_rh = analyzer.value("con_snr_rh")
    lines.extend(
        (
            _metric_line(
                "White-matter SNR (orig.mgz)",
                _fmt(wm_orig),
                cohort_context(wm_orig, analyzer.all_rows, ("wm_snr_orig",)),
            ),
            _metric_line(
                "Gray-matter SNR (orig.mgz)",
                _fmt(gm_orig),
                cohort_context(gm_orig, analyzer.all_rows, ("gm_snr_orig",)),
            ),
            _metric_line(
                "White-matter SNR (norm.mgz)",
                _fmt(wm_norm),
                cohort_context(wm_norm, analyzer.all_rows, ("wm_snr_norm",)),
            ),
            _metric_line(
                "Gray-matter SNR (norm.mgz)",
                _fmt(gm_norm),
                cohort_context(gm_norm, analyzer.all_rows, ("gm_snr_norm",)),
            ),
            _metric_line("WM SNR change after normalization", _percent_change(wm_orig, wm_norm)),
            _metric_line("GM SNR change after normalization", _percent_change(gm_orig, gm_norm)),
            _metric_line("WM/GM contrast SNR - left", _fmt(con_lh)),
            _metric_line("WM/GM contrast SNR - right", _fmt(con_rh)),
        )
    )
    lines.extend(
        _wrapped_bullet(
            "SNR here is mean divided by standard deviation within FSQC tissue masks. A rise after "
            "norm.mgz intensity normalization describes a processing-associated change; it does not "
            "recover lost acquisition information or prove that image quality improved."
        )
    )

    _section(lines, 4, "MRIQC-STYLE NO-REFERENCE IMAGE QUALITY METRICS")
    available_iqms = [key for key, _, _, _ in MRIQC_STYLE_METRICS if get_value(row, key) is not None]
    if available_iqms:
        for key, label, preferred, source in MRIQC_STYLE_METRICS:
            value = get_value(row, key)
            direction = "lower preferred" if preferred == "lower" else "higher preferred"
            context = cohort_context(value, analyzer.all_rows, (key,))
            lines.append(
                _metric_line(label, _fmt(value, 4), f"{direction}; {context}; {source}")
            )
        background_available = [
            (key, label, get_value(row, key))
            for key, label in BACKGROUND_METRICS
            if get_value(row, key) is not None
        ]
        if background_available:
            lines.append("  Harmonized-image background distribution:")
            for key, label, value in background_available:
                decimals = 0 if key == "bg_n" else 4
                lines.append(_metric_line(label, _fmt(value, decimals)))
    else:
        lines.extend(
            _wrapped_bullet(
                "No FSQC MRIQC-style fields were found (efc, qi2, fber, snr_tissue_total, "
                "snr_head, bg_*). If this metrics file predates those fields, recompute the FSQC "
                "core metrics without reusing the older metrics.csv."
            )
        )
    lines.extend(
        _wrapped_bullet(
            "Directionality follows MRIQC: lower is preferred for EFC and QI2; higher is preferred "
            "for FBER and SNR. These are no-reference IQMs, so the report uses compatible-cohort "
            "context rather than universal pass/fail cutoffs."
        )
    )
    lines.extend(
        _wrapped_bullet(
            "QI1 is not exported by FSQC and is not inferred. FSQC reuses FreeSurfer/FastSurfer "
            "bias correction and its own masks instead of MRIQC's full preprocessing workflow, so "
            "FSQC values should not be numerically compared with MRIQC-native reference values."
        )
    )

    _section(lines, 5, "SURFACE TOPOLOGY AND SEGMENTATION BURDEN")
    holes_lh = analyzer.value("holes_lh")
    holes_rh = analyzer.value("holes_rh")
    defects_lh = analyzer.value("defects_lh")
    defects_rh = analyzer.value("defects_rh")
    topo_lh = analyzer.value("topo_lh")
    topo_rh = analyzer.value("topo_rh")
    cc_size = analyzer.value("cc_size")
    lines.extend(
        (
            _metric_line("Initial holes - left", _fmt(holes_lh, 0)),
            _metric_line("Initial holes - right", _fmt(holes_rh, 0)),
            _metric_line(
                "Initial holes - total",
                _fmt(holes_lh + holes_rh, 0)
                if holes_lh is not None and holes_rh is not None
                else "not available",
            ),
            _metric_line("Topology defects - left", _fmt(defects_lh, 0)),
            _metric_line("Topology defects - right", _fmt(defects_rh, 0)),
            _metric_line("Topology-fixing time - left", _fmt(topo_lh) + " min" if topo_lh is not None else "not available"),
            _metric_line("Topology-fixing time - right", _fmt(topo_rh) + " min" if topo_rh is not None else "not available"),
            _metric_line("Corpus callosum / eTIV", _fmt(cc_size, 6)),
        )
    )
    lines.extend(
        _wrapped_bullet(
            "holes_lh/rh are derived from orig.nofix and describe pre-correction topology. "
            "defects_lh/rh are counts reported during automated correction. Neither count directly "
            "states whether final white and pial surfaces are geometrically accurate."
        )
    )

    _section(lines, 6, "TALAIRACH ORIENTATION / SPATIAL NORMALIZATION")
    rotations = [analyzer.value(f"rot_tal_{axis}") for axis in "xyz"]
    for axis, value in zip("xyz", rotations, strict=True):
        text = "not available" if value is None else f"{value:+.4f} rad ({math.degrees(value):+.1f} deg)"
        lines.append(_metric_line(f"Talairach rotation {axis}", text))
    if any(value is not None for value in rotations):
        max_rotation = max(abs(value) for value in rotations if value is not None)
        lines.append(_metric_line("Maximum absolute component", f"{math.degrees(max_rotation):.1f} deg"))
    lines.extend(
        _wrapped_bullet(
            "These values are rotations in the atlas transform, not an estimate of within-scan "
            "motion. Large rotations warrant an overlay/orientation check but cannot, alone, label "
            "registration as failed."
        )
    )

    _section(lines, 7, "TRUE HEMISPHERIC SUBCORTICAL VOLUME ASYMMETRY")
    volume_data = list(model["volume_pairs"])
    if volume_data:
        lines.append("  Structure                         Left mm^3   Right mm^3       AI       Ratio   Direction")
        for pair in volume_data:
            ai_text = "n/a" if pair["ai"] is None else f"{pair['ai']:+.1f}%"
            ratio_text = "n/a" if pair["ratio"] is None else f"{pair['ratio']:.2f}"
            lines.append(
                f"  {pair['structure']:<33} {pair['left']:>10.1f} {pair['right']:>12.1f} "
                f"{ai_text:>8} {ratio_text:>10}   {pair['direction']}"
            )
        lines.extend(
            _wrapped_bullet(
                "AI = 200 x (L - R) / (L + R); positive values are left-larger and negative values "
                "are right-larger. Ratio is larger/smaller and therefore has no direction. These are "
                "raw FreeSurfer/FastSurfer label volumes, not age/sex/eTIV-adjusted reference scores."
            )
        )
        lines.extend(
            _wrapped_bullet(
                f"Volume source: {regions_path}. Generic AI screening is intentionally conservative; "
                "structure-specific, matched reference distributions are preferred."
            )
        )
    else:
        lines.extend(
            _wrapped_bullet(
                "True left/right volume ratios are not present in fsqc-results.csv. No compatible "
                "outliers/all.regions.stats companion row was found, so volume asymmetry was not "
                "invented from the BrainPrint columns. Supply --regions-file to enable this section."
            )
        )

    _section(lines, 8, "BRAINPRINT LATERAL SHAPE ASYMMETRY")
    shape_metrics = list(model["shape_metrics"])
    if shape_metrics:
        lines.append("  Structure                                  Distance      Cohort context")
        for metric in shape_metrics:
            if metric["robust_z"] is None:
                context = f"no robust reference (n={metric['cohort_n']})"
            else:
                context = f"robust z={metric['robust_z']:+.2f} (n={metric['cohort_n']})"
            lines.append(f"  {metric['structure']:<42} {metric['value']:>9.3f}      {context}")
        lines.extend(
            _wrapped_bullet(
                "Important: these values are ShapeDNA/BrainPrint distances between left and right "
                "shape descriptors. They are not L/R volume ratios, have no left-versus-right "
                "direction, and have no universal cutoff. Larger means less shape similarity under "
                "the configured BrainPrint calculation."
            )
        )
    else:
        lines.extend(_wrapped_bullet("No BrainPrint lateral shape-distance fields were available."))

    _section(lines, 9, "STATISTICAL OUTLIER FLAGS")
    norm_count = analyzer.value("n_outlier_norms")
    nonpar_count = analyzer.value("n_outlier_sample_nonpar")
    param_count = analyzer.value("n_outlier_sample_param")
    lines.extend(
        (
            _metric_line("Normative-reference flags", _fmt(norm_count, 0)),
            _metric_line("Sample flags - 1.5 IQR", _fmt(nonpar_count, 0)),
            _metric_line("Sample flags - +/-2 SD", _fmt(param_count, 0)),
        )
    )
    detail_groups = (
        ("Normative", analyzer.normative_detail),
        ("Sample IQR", analyzer.nonparam_detail),
        ("Sample SD", analyzer.param_detail),
    )
    for label, details in detail_groups:
        if details:
            names = ", ".join(friendly_region_name(name) for name in details)
            lines.extend(_wrapped_bullet(f"{label} detail: {names}"))
    if nonpar_count is None or param_count is None:
        lines.extend(
            _wrapped_bullet(
                "Missing sample-based flags are expected when fewer than 10 subjects were supplied "
                "to FSQC; the toolbox returns NaN rather than estimating an unstable sample rule."
            )
        )
    lines.extend(
        _wrapped_bullet(
            "Normative/sample flags apply to segmentation-derived volumes or cortical thicknesses. "
            "They require label-level visual confirmation and do not constitute a radiological finding."
        )
    )

    _section(lines, 10, "RECOMMENDED REVIEW WORKFLOW")
    actions = review_actions(findings)
    for index, action in enumerate(actions, start=1):
        lines.extend(_wrapped_bullet(f"{index}. {action}", marker=""))

    if visuals:
        lines.append("")
        lines.append("  Discovered companion review material:")
        for label, path in visuals:
            lines.extend(_wrapped_bullet(f"{label}: {path}", indent=4))
    else:
        lines.extend(
            _wrapped_bullet(
                "No companion screenshots were discovered beside the input CSV. Generate FSQC "
                "screenshots/surfaces/skullstrip outputs or inspect interactively."
            )
        )

    _section(lines, 11, "METHOD AND INTERPRETATION NOTES")
    if analyzer.profile.enabled:
        lines.extend(
            _wrapped_bullet(
                "The 'cautious' profile uses transparent engineering screening bounds embedded in "
                "this script. They are not universal FSQC-validated clinical reference intervals. "
                "Use --profile descriptive to suppress fixed-cutoff flags, or validate/tune the "
                "ScreeningProfile for the study."
            )
        )
    else:
        lines.extend(
            _wrapped_bullet(
                "The descriptive profile suppresses fixed-cutoff screening. Native FSQC outlier "
                "flags and robust cohort comparisons remain reportable."
            )
        )
    if len(analyzer.all_rows) < MIN_COHORT_SIZE:
        lines.extend(
            _wrapped_bullet(
                f"Only {len(analyzer.all_rows)} subject row(s) were present. Robust cohort z-scores "
                f"require at least {MIN_COHORT_SIZE}; cross-structure BrainPrint distances were not "
                "treated as interchangeable reference values."
            )
        )
    missing = list(model["missing_core"])
    if missing:
        lines.extend(_wrapped_bullet("Missing core fields: " + ", ".join(missing)))
    lines.extend(
        (
            "",
            "  Technical references:",
            "  - DeepMI FSQC: https://deep-mi.org/fsqc/",
            "  - Legacy FreeSurfer QATools workflow: https://surfer.nmr.mgh.harvard.edu/fswiki/QATools",
            "  - MRIQC structural IQMs: https://mriqc.readthedocs.io/en/stable/iqms/t1w.html",
            "  - MRIQC measure taxonomy: https://mriqc.readthedocs.io/en/stable/measures.html",
            "  - BrainPrint asymmetry API: https://deep-mi.org/BrainPrint/dev/api/generated/brainprint.asymmetry.html",
            "  - FreeSurfer reconstruction QC: https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/MorphAndRecon",
            "",
            "FINAL HUMAN QC DECISION:  [ ] ACCEPT   [ ] ACCEPT WITH CAVEAT   [ ] REPROCESS",
            "                          [ ] EXCLUDE  [ ] REACQUIRE             Reviewer: __________",
            "=" * REPORT_WIDTH,
        )
    )
    return "\n".join(lines)


MD_SEVERITY = {
    Severity.INFO: "ℹ️ INFO",
    Severity.NOTE: "🟡 NOTE",
    Severity.REVIEW: "🟠 REVIEW",
    Severity.HIGH: "🔴 HIGH",
}


def _md_escape(value: object) -> str:
    """Escape arbitrary text for a Markdown table cell."""

    text = " ".join(str(value).splitlines()).strip()
    return html.escape(text, quote=False).replace("|", "\\|")


def _md_path(path: Path, document_dir: Path) -> str:
    """Create a portable, URL-safe path relative to the generated README."""

    try:
        relative = Path(os.path.relpath(path.resolve(), document_dir.resolve()))
    except (OSError, ValueError):
        relative = path
    return quote(relative.as_posix(), safe="/._-")


def _disposition_icon(disposition: object) -> str:
    """Return a visual marker for the automated QC disposition."""

    text = str(disposition)
    if text.startswith("HIGH-PRIORITY"):
        return "🔴"
    if text.startswith("TARGETED"):
        return "🟠"
    if text.startswith("MANUAL"):
        return "🟡"
    if text.startswith("NO QUANTITATIVE"):
        return "🟢"
    return "⚪"


def _asymmetry_bar(ai: object) -> str:
    """Render a compact, direction-neutral magnitude bar for a volume AI."""

    if not isinstance(ai, float):
        return "—"
    filled = min(10, max(0, round(abs(ai) / 5.0)))
    return "█" * filled + "░" * (10 - filled)


def _surface_label(path: Path) -> str:
    """Turn an FSQC rendering filename into a readable caption."""

    parts = path.stem.split(".")
    if len(parts) >= 3:
        hemi = parts[0].upper()
        surface = parts[1].capitalize()
        view = " ".join(parts[2:]).capitalize()
        return f"{hemi} {surface} — {view}"
    return path.stem.replace(".", " ").replace("_", " ").title()


def _markdown_image_grid(
    items: Sequence[tuple[str, Path]], document_dir: Path, width: int = 430
) -> list[str]:
    """Render local images two per row using README-compatible HTML."""

    if not items:
        return []
    lines = ["<table>"]
    for index in range(0, len(items), 2):
        pair = list(items[index : index + 2])
        lines.append("  <tr>")
        for label, _ in pair:
            lines.append(f"    <th>{html.escape(label)}</th>")
        if len(pair) == 1:
            lines.append("    <th></th>")
        lines.append("  </tr>")
        lines.append("  <tr>")
        for label, path in pair:
            source = _md_path(path, document_dir)
            lines.append(
                f'    <td><a href="{source}"><img src="{source}" '
                f'alt="{html.escape(label, quote=True)}" width="{width}"></a></td>'
            )
        if len(pair) == 1:
            lines.append("    <td></td>")
        lines.append("  </tr>")
    lines.append("</table>")
    return lines


def _markdown_visuals(
    visuals: Sequence[tuple[str, Path]], document_dir: Path
) -> list[str]:
    """Build primary-image and surface-rendering galleries."""

    primary: list[tuple[str, Path]] = []
    surfaces: list[tuple[str, Path]] = []
    for label, path in visuals:
        if path.is_file():
            primary.append((label.capitalize(), path))
        elif path.is_dir():
            surfaces.extend((_surface_label(image), image) for image in sorted(path.glob("*.png")))

    lines: list[str] = []
    if primary:
        lines.extend(_markdown_image_grid(primary, document_dir))
    if surfaces:
        lines.extend(
            (
                "",
                "<details>",
                f"<summary><strong>Surface rendering gallery ({len(surfaces)} images)</strong></summary>",
                "",
            )
        )
        lines.extend(_markdown_image_grid(surfaces, document_dir, width=400))
        lines.extend(("", "</details>"))
    if not lines:
        lines.append(
            "> No companion images were discovered. Generate FSQC screenshots, surface renderings, "
            "and skull-strip outputs, or inspect the reconstruction interactively."
        )
    return lines


def format_markdown_subject(
    model: Mapping[str, object],
    analyzer: FSQCAnalyzer,
    regions_path: Path | None,
    visuals: Sequence[tuple[str, Path]],
    document_dir: Path,
) -> str:
    """Render one subject as a detailed, pictorial Markdown section."""

    findings = list(model["findings"])
    high_count = sum(item.severity == Severity.HIGH for item in findings)
    review_count = sum(item.severity == Severity.REVIEW for item in findings)
    note_count = sum(item.severity == Severity.NOTE for item in findings)
    icon = _disposition_icon(model["disposition"])
    lines = [
        f"## Subject `{_md_escape(model['subject'])}`",
        "",
        "| QC disposition | Priority findings | Metric coverage | Screening profile |",
        "|---|---:|---:|---|",
        f"| {icon} **{_md_escape(model['disposition'])}** | 🔴 {high_count} · 🟠 {review_count} · "
        f"🟡 {note_count} | {model['available_core']}/{model['total_core']} | "
        f"`{_md_escape(analyzer.profile.name)}` |",
        "",
    ]

    callout = "WARNING" if high_count or review_count >= 2 else "CAUTION"
    lines.extend(
        (
            f"> [!{callout}]",
            "> **Automated triage only.** Hold or review measurements as indicated below, but make the "
            "final accept/reprocess/exclude decision from the source T1, labels, and surface overlays.",
            "",
            "### Executive assessment",
            "",
        )
    )
    priority = [item for item in findings if item.severity >= Severity.NOTE]
    if priority:
        lines.extend(
            (
                "| Priority | Domain | Finding | Evidence |",
                "|---|---|---|---|",
            )
        )
        for finding in priority:
            lines.append(
                f"| {MD_SEVERITY[finding.severity]} | {_md_escape(finding.domain)} | "
                f"**{_md_escape(finding.title)}** | {_md_escape(finding.evidence)} |"
            )
        lines.extend(("", "<details>", "<summary><strong>Expert interpretation and actions</strong></summary>", ""))
        for finding in priority:
            lines.extend(
                (
                    f"#### {MD_SEVERITY[finding.severity]} — {_md_escape(finding.title)}",
                    "",
                    f"- **Evidence:** {_md_escape(finding.evidence)}",
                    f"- **Interpretation:** {_md_escape(finding.interpretation)}",
                    f"- **Review action:** {_md_escape(finding.action)}",
                    "",
                )
            )
        lines.extend(("</details>", ""))
    else:
        lines.extend(
            (
                "No quantitative flags were generated. Localized errors can still be missed, so this "
                "does not constitute a visual QC pass.",
                "",
            )
        )

    lines.extend(("### Processing integrity and completeness", ""))
    integrity = dict(model.get("processing_integrity", {}))
    if integrity.get("available"):
        required_outputs = list(integrity.get("required_outputs", []))
        present_count = sum(bool(item[2]) for item in required_outputs)
        markers = list(integrity.get("completion_markers", []))
        completion_logs = list(integrity.get("completion_logs", []))
        error_markers = list(integrity.get("error_markers", []))
        status_icon = "✅" if not error_markers and present_count == len(required_outputs) else "⚠️"
        lines.extend(
            (
                "| Check | Result | Interpretation |",
                "|---|---:|---|",
                f"| Overall integrity | {status_icon} {_md_escape(integrity.get('status', 'not available'))} | Processing evidence only |",
                f"| Required morphometry products | {present_count}/{len(required_outputs)} present | File-existence check |",
                f"| Completion evidence | {len(markers) + len(completion_logs)} item(s) | Recognized marker or `finished without error` log phrase |",
                f"| Error marker files | {len(error_markers)} | Explicit `*.error` files |",
                "",
            )
        )
        statuses = dict(integrity.get("fsqc_statuses", {}))
        if statuses:
            lines.extend(
                (
                    "#### FSQC module execution state",
                    "",
                    "| Module | Code | Meaning |",
                    "|---|---:|---|",
                )
            )
            for module, status in statuses.items():
                label = FSQC_STATUS_LABELS.get(status, f"UNKNOWN ({status})")
                module_icon = "❌" if status == 1 else "✅" if status == 0 else "↩️" if status == 3 else "➖"
                lines.append(
                    f"| {_md_escape(module.replace('_', ' ').title())} | {status} | {module_icon} {_md_escape(label)} |"
                )
            lines.extend(
                (
                    "",
                    "> Code `3` means the current FSQC run reused an existing result; it is not a "
                    "new computation. Code `2` means the optional module was not requested.",
                    "",
                )
            )
        lines.extend(
            (
                "<details>",
                f"<summary><strong>Expected reconstruction products ({present_count}/{len(required_outputs)} present)</strong></summary>",
                "",
                "| Product | Relative path | Present |",
                "|---|---|:---:|",
            )
        )
        for label, relative, exists in required_outputs:
            lines.append(
                f"| {_md_escape(label)} | `{_md_escape(relative)}` | {'✅' if exists else '❌'} |"
            )
        lines.extend(("", "</details>", ""))
    else:
        lines.extend(
            (
                f"> ⚪ {_md_escape(integrity.get('status', 'Source subject directory was unavailable.'))}",
                "",
            )
        )
    lines.extend(
        (
            "> This section adapts the status/file-presence principle from the legacy "
            "[FreeSurfer QATools workflow](https://surfer.nmr.mgh.harvard.edu/fswiki/QATools). "
            "The linked page targets FreeSurfer 5.3 and is deprecated; its workflow informs this "
            "check, but it does not provide modern universal thresholds. Presence and successful "
            "termination do not guarantee anatomical accuracy.",
            "",
            "### Pictorial QC evidence",
            "",
        )
    )
    lines.extend(_markdown_visuals(visuals, document_dir))

    wm_orig = analyzer.value("wm_snr_orig")
    gm_orig = analyzer.value("gm_snr_orig")
    wm_norm = analyzer.value("wm_snr_norm")
    gm_norm = analyzer.value("gm_snr_norm")
    con_lh = analyzer.value("con_snr_lh")
    con_rh = analyzer.value("con_snr_rh")
    lines.extend(
        (
            "",
            "### Signal and tissue contrast",
            "",
            "| Metric | Value | Cohort context |",
            "|---|---:|---|",
            f"| WM SNR — `orig.mgz` | {_fmt(wm_orig)} | {_md_escape(cohort_context(wm_orig, analyzer.all_rows, ('wm_snr_orig',)))} |",
            f"| GM SNR — `orig.mgz` | {_fmt(gm_orig)} | {_md_escape(cohort_context(gm_orig, analyzer.all_rows, ('gm_snr_orig',)))} |",
            f"| WM SNR — `norm.mgz` | {_fmt(wm_norm)} | {_md_escape(cohort_context(wm_norm, analyzer.all_rows, ('wm_snr_norm',)))} |",
            f"| GM SNR — `norm.mgz` | {_fmt(gm_norm)} | {_md_escape(cohort_context(gm_norm, analyzer.all_rows, ('gm_snr_norm',)))} |",
            f"| WM SNR change after normalization | {_percent_change(wm_orig, wm_norm)} | Descriptive processing-associated change |",
            f"| GM SNR change after normalization | {_percent_change(gm_orig, gm_norm)} | Descriptive processing-associated change |",
            f"| WM/GM contrast SNR — left | {_fmt(con_lh)} | Surface sampled |",
            f"| WM/GM contrast SNR — right | {_fmt(con_rh)} | Surface sampled |",
            "",
            "> SNR is mean/SD within FSQC tissue masks. A post-normalization rise does not recover "
            "lost acquisition information or prove improved source-image quality.",
        )
    )

    iqm_explanations = {
        "efc": "Entropy sensitive to ghosting/blurring; non-specific",
        "qi2": "Goodness-of-fit of background noise to the modeled distribution",
        "fber": "Energy inside the head relative to background",
        "snr_tissue_total": "Mean mask-based SNR across GM, WM, and CSF",
        "snr_head": "Mask-based SNR over the full head region",
    }
    lines.extend(
        (
            "",
            "### MRIQC-style no-reference image quality metrics",
            "",
            "| FSQC field | Measure | Value | Preferred direction | Cohort context |",
            "|---|---|---:|:---:|---|",
        )
    )
    available_iqm_count = 0
    for key, label, preferred, _ in MRIQC_STYLE_METRICS:
        value = get_value(analyzer.row, key)
        if value is not None:
            available_iqm_count += 1
        direction = "↓ lower" if preferred == "lower" else "↑ higher"
        lines.append(
            f"| `{key}` | {_md_escape(label)} | {_fmt(value, 4)} | {direction} | "
            f"{_md_escape(cohort_context(value, analyzer.all_rows, (key,)))} |"
        )
    lines.extend(
        (
            "",
            "| Measure | What it helps screen | What it cannot establish |",
            "|---|---|---|",
        )
    )
    for key, label, _, _ in MRIQC_STYLE_METRICS:
        lines.append(
            f"| {_md_escape(label)} | {_md_escape(iqm_explanations[key])} | Artifact cause, "
            "clinical normality, or a universal pass/fail decision |"
        )
    background_available = [
        (key, label, get_value(analyzer.row, key))
        for key, label in BACKGROUND_METRICS
        if get_value(analyzer.row, key) is not None
    ]
    if background_available:
        lines.extend(
            (
                "",
                "<details>",
                "<summary><strong>Harmonized-image background distribution</strong></summary>",
                "",
                "| Statistic | FSQC field | Value |",
                "|---|---|---:|",
            )
        )
        for key, label, value in background_available:
            decimals = 0 if key == "bg_n" else 4
            lines.append(f"| {_md_escape(label)} | `{key}` | {_fmt(value, decimals)} |")
        lines.extend(("", "</details>"))
    if available_iqm_count == 0:
        lines.extend(
            (
                "",
                "> [!NOTE]",
                "> These fields are absent from this CSV. The existing `metrics.csv` may predate "
                "their addition or may have been reused with `--skip-existing`. Recompute FSQC core "
                "metrics to populate them; QI1 will remain unavailable because FSQC does not export it.",
            )
        )
    lines.extend(
        (
            "",
            "> [!IMPORTANT]",
            "> MRIQC describes these as **no-reference IQMs**. Directionality is useful—EFC/QI2 "
            "lower, FBER/SNR higher—but values should be compared within a scanner-, protocol-, "
            "population-, and software-matched cohort rather than against universal thresholds.",
            "",
            "> FSQC computes adapted versions using FreeSurfer/FastSurfer bias correction, "
            "harmonization, segmentations, and masks. They are therefore not numerically equivalent "
            "to a native MRIQC run. QI1 is not exported and is never inferred by this report.",
        )
    )

    holes_lh = analyzer.value("holes_lh")
    holes_rh = analyzer.value("holes_rh")
    defects_lh = analyzer.value("defects_lh")
    defects_rh = analyzer.value("defects_rh")
    holes_total = holes_lh + holes_rh if holes_lh is not None and holes_rh is not None else None
    defects_total = defects_lh + defects_rh if defects_lh is not None and defects_rh is not None else None
    lines.extend(
        (
            "",
            "### Surface topology and segmentation burden",
            "",
            "| Metric | Left | Right | Total / value |",
            "|---|---:|---:|---:|",
            f"| Initial holes (`orig.nofix`) | {_fmt(holes_lh, 0)} | {_fmt(holes_rh, 0)} | {_fmt(holes_total, 0)} |",
            f"| Topology defects | {_fmt(defects_lh, 0)} | {_fmt(defects_rh, 0)} | {_fmt(defects_total, 0)} |",
            f"| Topology-fixing time (min) | {_fmt(analyzer.value('topo_lh'))} | {_fmt(analyzer.value('topo_rh'))} | — |",
            f"| Corpus callosum / eTIV | — | — | {_fmt(analyzer.value('cc_size'), 6)} |",
            "",
            "> Hole and defect counts describe pre-/during-correction burden. They do not directly "
            "measure the geometric accuracy of final white and pial surfaces.",
        )
    )

    lines.extend(
        (
            "",
            "### Talairach orientation",
            "",
            "| Axis | Radians | Degrees |",
            "|:---:|---:|---:|",
        )
    )
    for axis in "xyz":
        value = analyzer.value(f"rot_tal_{axis}")
        radians = "not available" if value is None else f"{value:+.4f}"
        degrees = "not available" if value is None else f"{math.degrees(value):+.1f}°"
        lines.append(f"| {axis.upper()} | {radians} | {degrees} |")
    lines.extend(
        (
            "",
            "> These are atlas-transform rotations, not within-scan motion estimates. Large values "
            "trigger an orientation/overlay check but do not prove registration failure.",
            "",
            "### Hemispheric subcortical volume asymmetry",
            "",
        )
    )
    volume_data = list(model["volume_pairs"])
    if volume_data:
        lines.extend(
            (
                "| Structure | Left (mm³) | Right (mm³) | AI | Ratio | Magnitude | Direction |",
                "|---|---:|---:|---:|---:|:---:|---|",
            )
        )
        for pair in volume_data:
            ai = pair["ai"]
            ratio = pair["ratio"]
            ai_text = "n/a" if ai is None else f"{ai:+.1f}%"
            ratio_text = "n/a" if ratio is None else f"{ratio:.2f}"
            lines.append(
                f"| {_md_escape(pair['structure'])} | {pair['left']:.1f} | {pair['right']:.1f} | "
                f"{ai_text} | {ratio_text} | `{_asymmetry_bar(ai)}` | {_md_escape(pair['direction'])} |"
            )
        source = _md_path(regions_path, document_dir) if regions_path else "not available"
        lines.extend(
            (
                "",
                "AI = `200 × (L − R) / (L + R)`. Positive is left-larger; negative is right-larger. "
                "Ratio is larger/smaller and directionless.",
                "",
                f"Volume source: [`all.regions.stats`]({source})",
            )
        )
    else:
        lines.append(
            "True bilateral volumes were unavailable. BrainPrint distances were not relabeled as "
            "volume ratios; pass `--regions-file` to populate this section."
        )

    lines.extend(("", "### BrainPrint lateral shape asymmetry", ""))
    shape_metrics = list(model["shape_metrics"])
    if shape_metrics:
        lines.extend(("| Structure | Shape distance | Cohort context |", "|---|---:|---|"))
        for metric in shape_metrics:
            if metric["robust_z"] is None:
                context = f"No robust reference (n={metric['cohort_n']})"
            else:
                context = f"Robust z={metric['robust_z']:+.2f} (n={metric['cohort_n']})"
            lines.append(
                f"| {_md_escape(metric['structure'])} | {metric['value']:.3f} | {_md_escape(context)} |"
            )
        lines.extend(
            (
                "",
                "> BrainPrint values are directionless distances between left/right spectral shape "
                "descriptors—not L/R volume ratios. They have no universal absolute cutoff.",
            )
        )
    else:
        lines.append("No BrainPrint lateral shape-distance fields were available.")

    norm_count = analyzer.value("n_outlier_norms")
    nonpar_count = analyzer.value("n_outlier_sample_nonpar")
    param_count = analyzer.value("n_outlier_sample_param")
    lines.extend(
        (
            "",
            "### Statistical outlier flags",
            "",
            "| Reference method | Count | Flagged structures (when available) |",
            "|---|---:|---|",
            f"| FSQC normative reference | {_fmt(norm_count, 0)} | {_md_escape(', '.join(friendly_region_name(x) for x in analyzer.normative_detail) or '—')} |",
            f"| Sample non-parametric (1.5 IQR) | {_fmt(nonpar_count, 0)} | {_md_escape(', '.join(friendly_region_name(x) for x in analyzer.nonparam_detail) or '—')} |",
            f"| Sample parametric (±2 SD) | {_fmt(param_count, 0)} | {_md_escape(', '.join(friendly_region_name(x) for x in analyzer.param_detail) or '—')} |",
            "",
            "> Outlier status is a label-level review trigger, not evidence of disease and not proof "
            "of segmentation error. Sample flags are normally unavailable when FSQC receives fewer "
            "than 10 subjects.",
        )
    )
    image_count = sum(
        1 if path.is_file() else len(list(path.glob("*.png"))) if path.is_dir() else 0
        for _, path in visuals
    )
    outlier_summary = (
        "none reported"
        if not norm_count
        else f"{_fmt(norm_count, 0)} normative flag(s)"
    )
    lines.extend(
        (
            "",
            "### QATools-informed evidence synthesis",
            "",
            "| QA channel | Current evidence | Useful inference | Required confirmation |",
            "|---|---|---|---|",
            f"| Processing status and files | {_md_escape(integrity.get('status', 'not available'))} | Determines whether expected products and completion evidence exist | Inspect logs when evidence is missing or contradictory |",
            f"| aseg / morphometry outliers | {_md_escape(outlier_summary)} | Prioritizes specific labels for review | Confirm label boundaries and reference compatibility |",
            f"| Tissue SNR and WM/GM contrast | WM norm {_fmt(wm_norm)}; GM norm {_fmt(gm_norm)}; contrast L/R {_fmt(con_lh)}/{_fmt(con_rh)} | Identifies unusual tissue-mask intensity homogeneity or boundary contrast | Inspect native T1, normalization, tissue masks, and surfaces |",
            f"| MRIQC-style no-reference IQMs | {available_iqm_count}/{len(MRIQC_STYLE_METRICS)} principal fields available | Adds complementary entropy, background-noise fit, energy-ratio, and mask-SNR evidence | Compare only with compatible FSQC-derived cohorts and inspect the implicated masks/artifacts |",
            f"| Snapshot evidence | {image_count} image(s) embedded | Makes global or conspicuous errors easier to triage | Inspect all slices/planes interactively; snapshots can miss focal errors |",
            "",
            "> These channels are complementary. A complete run can still be anatomically wrong, "
            "a statistical outlier can be valid anatomy, and apparently good summary metrics can "
            "coexist with a focal surface error.",
            "",
            "### Human review checklist",
            "",
        )
    )
    for action in review_actions(findings):
        lines.append(f"- [ ] {_md_escape(action)}")

    lines.extend(
        (
            "",
            "### Final human QC decision",
            "",
            "- [ ] Accept",
            "- [ ] Accept with caveat",
            "- [ ] Reprocess and reassess",
            "- [ ] Exclude",
            "- [ ] Reacquire if clinically/research-feasible",
            "",
            "**Reviewer:** ____________________  **Date:** ____________________",
            "",
            "**Rationale / notes:**",
            "",
            "> _Add the image-based decision and affected structures here._",
        )
    )
    return "\n".join(lines)


def format_markdown_document(
    report_items: Sequence[
        tuple[Mapping[str, object], FSQCAnalyzer, Sequence[tuple[str, Path]], Path | None]
    ],
    source_path: Path,
    metadata: Mapping[str, str],
    readme_path: Path,
) -> str:
    """Render a complete README containing one or more subject reports."""

    generated = datetime.now().astimezone().isoformat(timespec="seconds")
    source_link = _md_path(source_path, readme_path.parent)
    fsqc_html = source_path.parent / "fsqc-results.html"
    lines = [
        "<!-- Generated by fsqc_report.py - edit the generator, not repeated metric text. -->",
        "# Neuroimaging Quality-Control Report",
        "",
        "> [!IMPORTANT]",
        "> This is an automated **research reconstruction-QC** report for FreeSurfer/FastSurfer "
        "outputs. It is not a medical diagnosis, a radiology report, or a substitute for visual review.",
        "",
        "## Report overview",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Generated | `{_md_escape(generated)}` |",
        f"| Input metrics | [`{_md_escape(source_path.name)}`]({source_link}) |",
        f"| Processing stream | {_md_escape(metadata.get('pipeline', 'not identified'))} |",
        f"| FSQC version | {_md_escape(metadata.get('version', 'not available'))} |",
        f"| Subjects reported | {len(report_items)} |",
        "",
        "| Subject | Automated disposition | Findings | Coverage |",
        "|---|---|---:|---:|",
    ]
    for model, _, _, _ in report_items:
        findings = list(model["findings"])
        priority_count = sum(item.severity >= Severity.NOTE for item in findings)
        icon = _disposition_icon(model["disposition"])
        lines.append(
            f"| `{_md_escape(model['subject'])}` | {icon} {_md_escape(model['disposition'])} | "
            f"{priority_count} | {model['available_core']}/{model['total_core']} |"
        )
    if fsqc_html.is_file():
        lines.extend(("", f"Original interactive FSQC output: [`fsqc-results.html`]({_md_path(fsqc_html, readme_path.parent)})"))

    for model, analyzer, visuals, regions_path in report_items:
        lines.extend(
            (
                "",
                "---",
                "",
                format_markdown_subject(
                    model=model,
                    analyzer=analyzer,
                    regions_path=regions_path,
                    visuals=visuals,
                    document_dir=readme_path.parent,
                ),
            )
        )

    lines.extend(
        (
            "",
            "---",
            "",
            "## Interpretation framework",
            "",
            "- Fixed cutoffs in the `cautious` profile are transparent engineering screening "
            "heuristics, not universal clinical reference intervals.",
            "- Prefer matched, quality-controlled cohort distributions generated with the same "
            "scanner protocol and software version.",
            "- Quantitative flags prioritize review; the final decision must be image-based and documented.",
            "- The linked QATools page documents a deprecated FreeSurfer 5.3 workflow. This report "
            "adopts its multi-channel QA principle—not its software assumptions or universal cutoffs.",
            "- MRIQC-style metrics are no-reference IQMs. Their preferred directions support cohort "
            "triage, but they do not supply ground-truth image quality or clinical thresholds.",
            "- FSQC's adapted IQMs and MRIQC-native IQMs use different preprocessing and should not "
            "be mixed in the same reference distribution.",
            "",
            "## Technical references",
            "",
            "- [DeepMI FSQC documentation](https://deep-mi.org/fsqc/)",
            "- [Legacy FreeSurfer QATools workflow (deprecated)](https://surfer.nmr.mgh.harvard.edu/fswiki/QATools)",
            "- [MRIQC structural-image IQM definitions](https://mriqc.readthedocs.io/en/stable/iqms/t1w.html)",
            "- [MRIQC image-quality measure taxonomy](https://mriqc.readthedocs.io/en/stable/measures.html)",
            "- [BrainPrint asymmetry API](https://deep-mi.org/BrainPrint/dev/api/generated/brainprint.asymmetry.html)",
            "- [FreeSurfer reconstruction QC](https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/MorphAndRecon)",
            "",
            "---",
            "",
            "<sub>Generated by <code>fsqc_report.py -R</code>. Research use only.</sub>",
        )
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate a detailed research QC report from DeepMI FSQC CSV output. "
            "The report is not a clinical diagnosis."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv_file", type=Path, help="Path to fsqc-results.csv")
    parser.add_argument(
        "--subject",
        action="append",
        default=[],
        help="Subject ID to report; repeat for multiple IDs. Default: all rows.",
    )
    parser.add_argument(
        "--save-report",
        type=Path,
        help="Optional output path for the combined plain-text report.",
    )
    parser.add_argument(
        "-R",
        "--readme",
        action="store_true",
        help="Generate a pictorial README.md beside the input CSV.",
    )
    parser.add_argument(
        "--regions-file",
        type=Path,
        help="Optional all.regions.stats used for true L/R volume asymmetry.",
    )
    parser.add_argument(
        "--outlier-dir",
        type=Path,
        help="Optional directory containing FSQC detailed outlier *.stats tables.",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILES),
        default="cautious",
        help="Fixed-cutoff screening profile.",
    )
    parser.add_argument(
        "--no-discovery",
        action="store_true",
        help="Do not auto-discover companion tables, logfile, or images beside the CSV.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args(argv)
    source_path = args.csv_file.expanduser()
    if not source_path.is_absolute():
        source_path = (Path.cwd() / source_path).resolve()
    try:
        rows = read_csv_rows(source_path)
    except ValueError as exc:
        parser.error(str(exc))

    available_subjects = [subject_id(row) for row in rows]
    if args.subject:
        wanted = set(args.subject)
        missing_subjects = sorted(wanted.difference(available_subjects))
        if missing_subjects:
            parser.error(
                "subject(s) not found: "
                + ", ".join(missing_subjects)
                + "; available: "
                + ", ".join(available_subjects)
            )
        selected_rows = [row for row in rows if subject_id(row) in wanted]
    else:
        selected_rows = rows

    base = source_path.parent
    regions_path: Path | None = None
    outlier_dir: Path | None = None
    metadata: dict[str, str] = {}

    if args.regions_file:
        regions_path = args.regions_file.expanduser().resolve()
        if not regions_path.is_file():
            parser.error(f"regions file not found: {regions_path}")
    elif not args.no_discovery:
        regions_path = discover_file(
            base,
            (
                Path("outliers/all.regions.stats"),
                Path("all.regions.stats"),
            ),
        )

    if args.outlier_dir:
        outlier_dir = args.outlier_dir.expanduser().resolve()
        if not outlier_dir.is_dir():
            parser.error(f"outlier directory not found: {outlier_dir}")
    elif not args.no_discovery and (base / "outliers").is_dir():
        outlier_dir = base / "outliers"

    if not args.no_discovery:
        metadata = parse_log_metadata(base)

    region_rows = load_optional_table(regions_path)
    norm_rows: list[dict[str, str]] = []
    nonpar_rows: list[dict[str, str]] = []
    param_rows: list[dict[str, str]] = []
    if outlier_dir is not None:
        norm_rows = load_optional_table(
            discover_file(outlier_dir, (Path("all.outliers.norms.stats"),))
        )
        nonpar_rows = load_optional_table(
            discover_file(outlier_dir, (Path("all.outliers.sample.nonpar.stats"),))
        )
        param_rows = load_optional_table(
            discover_file(outlier_dir, (Path("all.outliers.sample.param.stats"),))
        )

    reports: list[str] = []
    report_items: list[
        tuple[Mapping[str, object], FSQCAnalyzer, Sequence[tuple[str, Path]], Path | None]
    ] = []
    for row in selected_rows:
        sid = subject_id(row)
        region_row = row_for_subject(region_rows, sid)
        processing_integrity = inspect_processing_integrity(base, sid, metadata)
        analyzer = FSQCAnalyzer(
            row=row,
            all_rows=rows,
            profile=PROFILES[args.profile],
            region_row=region_row,
            normative_detail=detailed_outliers(norm_rows, sid),
            nonparam_detail=detailed_outliers(nonpar_rows, sid),
            param_detail=detailed_outliers(param_rows, sid),
            processing_integrity=processing_integrity,
        )
        model = analyzer.analyze()
        visuals = [] if args.no_discovery else discover_visuals(base, sid)
        report_items.append((model, analyzer, visuals, regions_path))
        reports.append(
            format_report(
                model=model,
                analyzer=analyzer,
                source_path=source_path,
                regions_path=regions_path,
                metadata=metadata,
                visuals=visuals,
            )
        )

    output = "\n\n".join(reports)
    try:
        print(output)
    except BrokenPipeError:
        # A closed downstream pipe should not prevent --save-report from completing.
        pass

    if args.save_report:
        save_path = args.save_report.expanduser()
        if not save_path.is_absolute():
            save_path = (Path.cwd() / save_path).resolve()
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(output + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"error: cannot save report to {save_path}: {exc}", file=sys.stderr)
            return 1
        print(f"\nSaved report: {save_path}")

    if args.readme:
        readme_path = base / "README.md"
        markdown = format_markdown_document(
            report_items=report_items,
            source_path=source_path,
            metadata=metadata,
            readme_path=readme_path,
        )
        try:
            readme_path.write_text(markdown, encoding="utf-8")
        except OSError as exc:
            print(f"error: cannot save README to {readme_path}: {exc}", file=sys.stderr)
            return 1
        print(f"\nGenerated pictorial README: {readme_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
