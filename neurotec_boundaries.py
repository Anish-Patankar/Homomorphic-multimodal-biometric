"""
neurotec_boundaries.py
======================
Loads the pupil / limbus boundary points exported by ExportBoundaries.java
(Neurotechnology MegaMatcher / VeriEye iris segmentation) so that
iris_encoder.image_to_iris_code_from_boundaries() can use them in place of our
own circle fitting.

File format (JSON Lines, one object per image; produced by ExportBoundaries):

    {"path": "012left/03.bmp", "status": "OK", "objects": 1,
     "inner": [[235,76],[234,67],...], "outer": [[289,76],[287,57],...]}
    {"path": "012left/04.bmp", "status": "TARGET_NOT_FOUND"}

`path` is relative to the dataset root, with forward slashes, so it can be
matched on any OS with boundary_key().

Point convention (see iris_encoder.resample_boundary): N points per contour,
point 0 at the right-hand side, running counter-clockwise on screen, equally
spaced in angle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np


def boundary_key(image_path: Path, dataset_root: Path) -> str:
    """Key used in the JSONL: image path relative to the dataset root,
    forward slashes (identical on Windows and Linux)."""
    return Path(image_path).relative_to(Path(dataset_root)).as_posix()


def _contour_convention(pts: np.ndarray):
    """(start_angle_deg, mean_step_deg) of a closed contour about its own
    centroid, in image coordinates (y down). The convention assumed by
    iris_encoder.resample_boundary is: point 0 at ~0 deg (right-hand side) and
    each next point 360/N degrees further round COUNTER-clockwise on screen,
    i.e. mean step = -360/N in this y-down angle."""
    c = pts.mean(axis=0)
    ang = np.degrees(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
    step = np.diff(np.r_[ang, ang[0]])
    step = (step + 180.0) % 360.0 - 180.0
    return float(ang[0]), float(step.mean())


def _check_convention(records: Dict[str, dict], verbose: bool = True) -> None:
    """Verify, for every loaded record, that both contours follow the point
    ordering assumed by the encoder; warn loudly if some do not."""
    bad = {"inner": [], "outer": []}
    starts = {"inner": [], "outer": []}
    for path, e in records.items():
        for name in ("inner", "outer"):
            pts = e[name]
            n = len(pts)
            start, step = _contour_convention(pts)
            starts[name].append(start)
            ok = abs(start) <= 15.0 and abs(step + 360.0 / n) <= 2.0
            if not ok:
                bad[name].append(path)
    if not verbose or not records:
        return
    n_rec = len(records)
    for name in ("inner", "outer"):
        frac = 1 - len(bad[name]) / n_rec
        med = float(np.median(starts[name]))
        print(f"  contour ordering check ({name}): {frac:.1%} of {n_rec} follow the assumed convention "
              f"(median start angle {med:+.1f} deg)")
    if bad["inner"] or bad["outer"]:
        worst = sorted(set(bad["inner"]) | set(bad["outer"]))[:5]
        print("  WARNING: some contours do NOT follow the assumed ordering (point 0 at the right, "
              "counter-clockwise, evenly spaced). Those images are unrolled incorrectly and will "
              f"look like bad matches. Examples: {worst}")


def load_boundaries(jsonl_path: Path, verbose: bool = True) -> Dict[str, dict]:
    """Return {relative_path: {"inner": (N,2) float array, "outer": (N,2)
    float array, "objects": int}} for every image the SDK segmented with
    status OK. Images with any other status are not included (so the caller
    treats them as 'no segmentation available', exactly like a failed
    segmentation of our own). A truncated final line (interrupted export) is
    ignored."""
    out: Dict[str, dict] = {}
    n_lines = n_bad = 0
    status_counts: Dict[str, int] = {}
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            status = str(rec.get("status", "?"))
            status_counts[status] = status_counts.get(status, 0) + 1
            if status != "OK":
                continue
            inner = np.asarray(rec["inner"], dtype=np.float64)
            outer = np.asarray(rec["outer"], dtype=np.float64)
            if inner.ndim != 2 or outer.ndim != 2 or len(inner) < 8 or len(outer) < 8:
                n_bad += 1
                continue
            out[rec["path"]] = {"inner": inner, "outer": outer, "objects": int(rec.get("objects", 1))}
    if verbose:
        print(f"Loaded boundaries for {len(out)} images from {jsonl_path} "
              f"({n_lines} records; statuses: {status_counts}; unreadable: {n_bad})")
        _check_convention(out, verbose=True)
        multi = sum(1 for v in out.values() if v["objects"] != 1)
        if multi:
            print(f"  note: {multi} images had more than one detected iris object; "
                  f"only the first was exported.")
    return out