"""
visualize_boundaries.py
=======================
Draws the pupil (red) and limbus (green) contours exported by
ExportBoundaries.java on top of the images, so you can see whether the SDK's
segmentation is right on a given image -- e.g. the images that appear in the
"worst genuine pairs" list of test_iitd.py.

Each image can be given as a path relative to the dataset root
(e.g. 013right/013_2_3.bmp) or, if the file name is unique in the dataset,
just the file name (013_2_3.bmp).

Usage
-----
    python visualize_boundaries.py "C:\\data\\CASIA" boundaries.jsonl 013_2_3.bmp 013_2_4.bmp
    python visualize_boundaries.py "C:\\data\\IITD" boundaries.jsonl 055right/06.bmp 055right/07.bmp --out check.png

Output: boundaries_check.png (or --out), one panel per image, titled with the
file name and three numbers that help spot bad segmentations:
    pupil/iris radius ratio     (typically ~0.2-0.7)
    center offset               (pupil centre vs limbus centre, % of limbus radius; typically < ~15%)
    fraction of limbus in frame (a low value means eyelids/frame edge hide much of the contour)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from iris_encoder import resample_boundary
from neurotec_boundaries import boundary_key, load_boundaries

IMAGE_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif"}


def resolve(root: Path, name: str) -> Path:
    direct = root / name
    if direct.is_file():
        return direct
    matches = [p for p in root.rglob(Path(name).name)
               if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise SystemExit(f"Image not found under {root}: {name}")
    shown = ", ".join(m.relative_to(root).as_posix() for m in matches[:4])
    raise SystemExit(f"'{name}' is ambiguous ({len(matches)} matches, e.g. {shown}). "
                     f"Give the path relative to the dataset root.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument("boundaries", type=Path, help="JSONL written by ExportBoundaries.java")
    ap.add_argument("images", nargs="+", help="relative path or unique file name, one or more")
    ap.add_argument("--out", type=Path, default=Path("boundaries_check.png"))
    args = ap.parse_args()

    b = load_boundaries(args.boundaries, verbose=False)
    paths = [resolve(args.dataset_root, n) for n in args.images]

    cols = min(3, len(paths))
    rows = (len(paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.6 * rows), squeeze=False)
    axes = axes.ravel()

    for ax, p in zip(axes, paths):
        key = boundary_key(p, args.dataset_root)
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            ax.set_title(f"{p.name}\n[unreadable]", fontsize=9); ax.axis("off"); continue
        disp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
        h, w = disp.shape[:2]

        e = b.get(key)
        if e is None:
            ax.imshow(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{p.name}\n[no SDK boundaries for this image]", fontsize=9); ax.axis("off"); continue

        info = []
        for pts, color in ((e["inner"], (0, 0, 255)), (e["outer"], (0, 255, 0))):
            xs, ys = resample_boundary(pts, 256)
            poly = np.round(np.c_[xs, ys]).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(disp, [poly], True, color, 2)
            for x, y in pts:                                   # the raw SDK points
                cv2.circle(disp, (int(round(x)), int(round(y))), 2, (255, 255, 0), -1)
            info.append((xs, ys))

        (xp, yp), (xs_, ys_) = info
        r_in = np.hypot(xp - xp.mean(), yp - yp.mean()).mean()
        r_out = np.hypot(xs_ - xs_.mean(), ys_ - ys_.mean()).mean()
        offset = np.hypot(xp.mean() - xs_.mean(), yp.mean() - ys_.mean()) / r_out
        in_frame = np.mean((xs_ >= 0) & (xs_ < w) & (ys_ >= 0) & (ys_ < h))

        ax.imshow(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{p.name}\nratio {r_in / r_out:.2f}  offset {offset:.0%}  limbus in frame {in_frame:.0%}",
                     fontsize=9)
        ax.axis("off")

    for ax in axes[len(paths):]:
        ax.axis("off")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88 if rows == 1 else 0.93)   # titles on axis('off') panels are ignored by tight_layout
    fig.savefig(args.out, dpi=140)
    print(f"Saved {args.out}   (red = pupil, green = limbus, yellow dots = raw SDK points)")


if __name__ == "__main__":
    main()