"""
test_iitd.py
============
Batch-evaluates iris_encoder.py against the IIT Delhi (IITD) Iris Database
and reproduces the genuine/impostor Hamming-distance separation plots from
Daugman's paper (Figs. 6, 9, 10, 11).

Expected dataset layout (standard IITD release):

    <dataset_root>/
        001/
            01_L.bmp
            02_L.bmp
            ...
        002/
            01_L.bmp
            ...
        ...

Any layout where each subject has their own subfolder containing several
images of (in general) the same eye will work -- the script treats every
subfolder as one identity/class and every image inside it as one sample of
that class ("same iris" = genuine pairs; "different subfolders" = impostor
pairs).

Usage
-----
    python test_iitd.py /path/to/IITD --max-impostor-pairs 3000

    # use Neurotechnology segmentation (boundaries exported by ExportBoundaries.java)
    python test_iitd.py /path/to/IITD --boundaries boundaries.jsonl

    # fair comparison: evaluate both segmenters on exactly the same images
    python test_iitd.py /path/to/IITD --dump-used used_ours.txt
    python test_iitd.py /path/to/IITD --boundaries boundaries.jsonl --dump-used used_neuro.txt
    python test_iitd.py /path/to/IITD --restrict-to used_ours.txt used_neuro.txt
    python test_iitd.py /path/to/IITD --boundaries boundaries.jsonl --restrict-to used_ours.txt used_neuro.txt

Outputs
-------
    * Binary iriscodes (2048 bits each) written to files mirroring the
      input dataset folder structure under `<dataset_root>_iriscodes/`
      (or `--codes-dir`).
    * Printed summary statistics (mean/std of genuine & impostor HD,
      count of segmentation failures, suggested decision threshold).
    * A histogram plot saved to iris_hd_distributions.png (and shown if a
      display is available).
    * Optional single-file export via --dump-codes <file>.
"""

from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from iris_encoder import (
    IrisCode,
    SegmentationError,
    hamming_distance,
    hamming_distance_debug,
    image_to_iris_code,
    image_to_iris_code_from_boundaries,
)

IMAGE_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif"}


import re

_EYE_TAG_RE = re.compile(r"[_-](L|R|left|right)\b", re.IGNORECASE)


def _eye_tag(filename: str) -> str:
    """Extract a left/right-eye tag from a filename like '01_L.bmp' ->
    'L'. Returns '' if no tag is found (single-eye-per-folder datasets)."""
    m = _EYE_TAG_RE.search(filename)
    if not m:
        return ""
    return m.group(1).upper()[0]  # 'L' or 'R'


def discover_dataset(root: Path, split_index: Optional[int] = None,
                     class_regex: Optional[str] = None) -> Dict[str, List[Path]]:
    """Group image files into identity classes.

    A 'class' == one distinct iris (one eye of one subject). Images are
    grouped by the folder that directly contains them (searched
    recursively, so an extra nesting level such as 'IITD Database/IITD
    Database/001/' is fine), and within a folder by eye:

      * if the filename carries an eye tag ('01_L.bmp', '03-right.bmp'),
        that decides the eye;
      * else, if `split_index` is given, files whose leading number is
        <= split_index are treated as one eye ('L') and the rest as the
        other ('R') -- e.g. split_index=5 for '01-05 left, 06-10 right';
      * else the whole folder is one class (correct only if each folder
        really holds a single eye).

    Class key = folder path relative to `root` (+ '_L'/'_R' when split).

    If `class_regex` is given it overrides all of the above: every image
    under `root`, in any folder, is assigned to the class named by group 1
    of the regex matched against the file name (without extension). E.g.
    "([0-9]+)_" puts '060_1_3.bmp' and '060_2_1.bmp' in class '060'.
    """
    if class_regex is not None:
        rx = re.compile(class_regex)
        grouped: Dict[str, List[Path]] = {}
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                m = rx.match(p.stem)
                if m:
                    grouped.setdefault(m.group(1), []).append(p)
        return {k: v for k, v in grouped.items() if len(v) >= 2}

    by_dir: Dict[Path, List[Path]] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            by_dir.setdefault(p.parent, []).append(p)

    classes: Dict[str, List[Path]] = {}
    lumped: List[str] = []
    for d, files in sorted(by_dir.items()):
        rel = d.relative_to(root).as_posix()
        base = root.name if rel == "." else rel
        by_eye: Dict[str, List[Path]] = {}
        for p in files:
            tag = _eye_tag(p.name)
            if not tag and split_index is not None:
                m = re.match(r"(\d+)", p.stem)
                if m:
                    tag = "L" if int(m.group(1)) <= split_index else "R"
            by_eye.setdefault(tag, []).append(p)
        for tag, imgs in by_eye.items():
            if len(imgs) >= 2:  # need >=2 images to form a genuine pair
                key = f"{base}_{tag}" if tag else base
                classes[key] = imgs
                if not tag and len(imgs) >= 8:
                    lumped.append(key)

    if lumped and split_index is None:
        print(f"NOTE: {len(lumped)} folder(s) (e.g. '{lumped[0]}') hold 8+ untagged images and "
              f"are being treated as ONE eye each. If those folders contain both eyes, genuine/"
              f"impostor pairs will be mislabeled -- see --split-by-index and check_eye_split.py.")
    return classes


def encode_all(classes: Dict[str, List[Path]], verbose: bool = True,
               boundaries=None, root: Path = None, max_harmonics=None):
    """Encode every image; returns (encoded, used_paths) where both dicts
    share the same class->list ordering (used_paths[cls][i] is the source
    image for encoded[cls][i]), skipping images where segmentation fails
    or occlusion is too heavy.

    If `boundaries` (from neurotec_boundaries.load_boundaries) is given,
    those externally-computed pupil/limbus boundaries replace our own
    segmentation; `root` is the dataset root used to build the lookup keys."""
    encoded: Dict[str, List[IrisCode]] = {}
    used_paths: Dict[str, List[Path]] = {}
    n_ok, n_fail, n_no_boundary = 0, 0, 0
    if boundaries is not None:
        from neurotec_boundaries import boundary_key
    for cls, paths in classes.items():
        codes, kept_paths = [], []
        for p in paths:
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                n_fail += 1
                continue
            try:
                if boundaries is None:
                    code = image_to_iris_code(img)
                else:
                    entry = boundaries.get(boundary_key(p, root))
                    if entry is None:           # SDK found no iris in this image
                        n_no_boundary += 1
                        n_fail += 1
                        continue
                    code = image_to_iris_code_from_boundaries(
                        img, entry["inner"], entry["outer"], max_harmonics=max_harmonics)
            except SegmentationError:
                n_fail += 1
                continue
            if code.mask.mean() < 0.4:
                n_fail += 1
                continue
            codes.append(code)
            kept_paths.append(p)
            n_ok += 1
        if len(codes) >= 2:
            encoded[cls] = codes
            used_paths[cls] = kept_paths
    if verbose:
        print(f"Encoded {n_ok} images successfully, {n_fail} skipped "
              f"(segmentation failure or too much occlusion).")
        if boundaries is not None:
            print(f"  ({n_no_boundary} of those had no usable external segmentation)")
        print(f"{len(encoded)} identities retained (>=2 usable images each).")
    return encoded, used_paths


def genuine_distances(encoded: Dict[str, List[IrisCode]], paths: Dict[str, List[Path]],
                      max_shift: int = 16):
    """Returns (distances, debug_records). debug_records entries are
    (cls, path1, path2, hd, n_valid_bits) for later inspection."""
    dists, records = [], []
    for cls, codes in encoded.items():
        cls_paths = paths[cls]
        for (i1, c1), (i2, c2) in itertools.combinations(enumerate(codes), 2):
            hd, n_valid, _ = hamming_distance_debug(c1, c2, max_shift=max_shift)
            dists.append(hd)
            records.append((cls, cls_paths[i1].name, cls_paths[i2].name, hd, n_valid))
    return dists, records


def impostor_distances(
    encoded: Dict[str, List[IrisCode]], paths: Dict[str, List[Path]],
    max_pairs: int, seed: int = 0, max_shift: int = 16,
):
    rng = random.Random(seed)
    classes = list(encoded.keys())
    dists, records = [], []
    attempts, max_attempts = 0, max_pairs * 20
    while len(dists) < max_pairs and attempts < max_attempts and len(classes) > 1:
        attempts += 1
        c1_name, c2_name = rng.sample(classes, 2)
        i1 = rng.randrange(len(encoded[c1_name]))
        i2 = rng.randrange(len(encoded[c2_name]))
        hd, n_valid, _ = hamming_distance_debug(encoded[c1_name][i1], encoded[c2_name][i2], max_shift=max_shift)
        dists.append(hd)
        records.append((f"{c1_name} vs {c2_name}", paths[c1_name][i1].name,
                         paths[c2_name][i2].name, hd, n_valid))
    return dists, records


def summarize(name: str, dists: List[float]):
    arr = np.array(dists, dtype=float)
    arr = arr[~np.isnan(arr)]
    print(f"{name}: n={len(arr)}  mean={arr.mean():.4f}  std={arr.std():.4f}  "
          f"min={arr.min():.4f}  max={arr.max():.4f}")
    return arr


def suggest_threshold(genuine: np.ndarray, impostor: np.ndarray) -> float:
    """Simple crossover-style threshold: midpoint between genuine mean+3std
    and impostor mean-3std if they don't overlap, else the point minimizing
    total misclassification on a fine grid."""
    grid = np.linspace(0.0, 0.5, 1001)
    best_t, best_err = 0.3, np.inf
    for t in grid:
        far = (impostor < t).mean()   # false accept rate
        frr = (genuine >= t).mean()   # false reject rate
        err = far + frr
        if err < best_err:
            best_err, best_t = err, t
    return best_t


def plot_distributions(genuine: np.ndarray, impostor: np.ndarray, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 0.5, 60)
    ax.hist(impostor, bins=bins, alpha=0.6, label=f"Impostors (n={len(impostor)})", color="black")
    ax.hist(genuine, bins=bins, alpha=0.6, label=f"Authentics (n={len(genuine)})", color="tab:blue")
    ax.set_xlabel("Fractional Hamming Distance")
    ax.set_ylabel("Count")
    ax.set_title("Genuine vs. Impostor Iris-Code Hamming Distances (IITD)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main():     
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument("--max-impostor-pairs", type=int, default=3000)
    ap.add_argument("--out", type=Path, default=Path("iris_hd_distributions.png"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-by-index", type=int, default=None, metavar="N",
                    help="for untagged filenames like 01.bmp..10.bmp: treat numbers <= N as one eye "
                         "and the rest as the other (e.g. 5). Verify the layout first with "
                         "check_eye_split.py. Default: each folder is one class")
    ap.add_argument("--class-regex", type=str, default=None, metavar="REGEX",
                    help="define classes from the file name instead of the folder: group 1 of the regex "
                         "(matched at the start of the name without extension) is the class id. "
                         "e.g. ([0-9]+)_ for names like 060_1_3.bmp. Overrides folder/eye logic")
    ap.add_argument("--boundaries", type=Path, default=None,
                    help="JSONL of external pupil/limbus boundaries (ExportBoundaries.java); "
                         "if omitted, our own segmentation is used")
    ap.add_argument("--smooth-harmonics", type=int, default=None,
                    help="optionally low-pass external boundaries to this many harmonics "
                         "(trims +-0.5px integer rounding); default: exact interpolation")
    ap.add_argument("--max-shift", type=int, default=16,
                    help="maximum angular columns to shift for rotation tolerance (default: 16)")
    ap.add_argument("--dump-used", type=Path, default=None,
                    help="write the relative paths of all successfully encoded images to this file")
    ap.add_argument("--codes-dir", type=Path, default=None,
                    help="directory to write binary iriscodes mirroring input folder structure "
                         "(default: <dataset_root>_iriscodes)")
    ap.add_argument("--no-save-codes", action="store_true", default=False,
                    help="do not write binary iriscodes to disk")
    ap.add_argument("--save-mask", action="store_true", default=False,
                    help="also save noise mask bit strings (<name>_mask.txt) alongside the code files")
    ap.add_argument("--dump-codes", type=Path, default=None,
                    help="optionally write all relative paths and binary iriscodes to this single text file")
    ap.add_argument("--print-codes", action="store_true", default=False,
                    help="print binary iriscodes to stdout")
    ap.add_argument("--restrict-to", type=Path, nargs="+", default=None,
                    help="only evaluate images listed in ALL of these files (intersection)")
    args = ap.parse_args()

    if not args.dataset_root.exists():
        raise SystemExit(f"Dataset root not found: {args.dataset_root}")

    classes = discover_dataset(args.dataset_root, args.split_by_index, args.class_regex)
    if not classes:
        imgs = [p for p in args.dataset_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        top = sorted(x.name for x in args.dataset_root.iterdir())[:8]
        if not imgs:
            raise SystemExit(f"No image files ({', '.join(sorted(IMAGE_EXTS))}) found anywhere under "
                             f"{args.dataset_root}. Top-level entries: {top}. Check the path.")
        raise SystemExit(f"Found {len(imgs)} images (e.g. {imgs[0]}) but no folder holds 2+ images of "
                         f"the same eye. If eyes are split by filename tag or number, check "
                         f"--split-by-index; if you passed --class-regex, check that it matches "
                         f"the file names.")
    print(f"Discovered {len(classes)} candidate identities.")
    from collections import Counter
    sizes = Counter(len(v) for v in classes.values())
    print("Images per class: " + ", ".join(f"{n} images x {c} classes" for n, c in sorted(sizes.items())) +
          "   <- check this matches what you expect (e.g. 5 per eye in IITD, 7 per eye in CASIA-V1)")

    if args.class_regex is None:
        # Heuristic safety net: file names like 060_1_3.bmp / 060_2_1.bmp carry a subject ID. If the same
        # ID shows up in 2-4 different classes for most subjects, the folders are probably sessions (or
        # otherwise splits) of ONE eye, and pairs between them are being counted as impostors.
        owners = {}
        for k, ps in classes.items():
            for p in ps:
                m = re.match(r"([0-9]+)_", p.stem)
                if m:
                    owners.setdefault(m.group(1), set()).add(k)
        shared = {pre: ks for pre, ks in owners.items() if 2 <= len(ks) <= 4}
        if owners and len(shared) >= 0.5 * len(owners):
            ex_pre, ex_ks = sorted(shared.items())[0]
            print(f"NOTICE: {len(shared)} of {len(owners)} subject IDs found in file names appear across multiple "
                  f"folders (e.g. ID {ex_pre} in {sorted(ex_ks)}).\n"
                  f"        These represent multiple sessions of the SAME eye (e.g. CASIA-V1 sessions).\n"
                  f"        Automatically grouping by subject ID regex '([0-9]+)_' so cross-session pairs are evaluated as GENUINE.")
            classes = discover_dataset(args.dataset_root, args.split_by_index, class_regex=r"([0-9]+)_")
            sizes = Counter(len(v) for v in classes.values())
            print(f"Regrouped into {len(classes)} identities: " +
                  ", ".join(f"{n} images x {c} classes" for n, c in sorted(sizes.items())))

    if args.restrict_to:
        keep = None
        for fp in args.restrict_to:
            lines = {ln.strip() for ln in open(fp, encoding="utf-8") if ln.strip()}
            keep = lines if keep is None else (keep & lines)
        classes = {c: [p for p in ps if p.relative_to(args.dataset_root).as_posix() in keep]
                   for c, ps in classes.items()}
        classes = {c: ps for c, ps in classes.items() if len(ps) >= 2}
        print(f"Restricted to {sum(len(v) for v in classes.values())} images "
              f"in {len(classes)} classes (intersection of {len(args.restrict_to)} list(s)).")

    boundaries = None
    if args.boundaries is not None:
        from neurotec_boundaries import load_boundaries
        boundaries = load_boundaries(args.boundaries)
        print("Segmentation source: EXTERNAL boundaries (" + str(args.boundaries) + ")")
    else:
        print("Segmentation source: built-in (Hough + edge-contrast refinement)")

    encoded, used_paths = encode_all(classes, boundaries=boundaries, root=args.dataset_root,
                                     max_harmonics=args.smooth_harmonics)
    if len(encoded) < 2:
        raise SystemExit("Not enough successfully-encoded identities to evaluate.")

    if args.dump_used is not None:
        with open(args.dump_used, "w", encoding="utf-8") as fh:
            for ps in used_paths.values():
                for p in ps:
                    fh.write(p.relative_to(args.dataset_root).as_posix() + "\n")
        print(f"Wrote list of used images to {args.dump_used}")

    if not args.no_save_codes:
        out_dir = args.codes_dir if args.codes_dir is not None else (args.dataset_root.parent / f"{args.dataset_root.name}_iriscodes")
        out_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for cls in sorted(used_paths.keys()):
            for p, code in zip(used_paths[cls], encoded[cls]):
                rel = p.relative_to(args.dataset_root)
                dest = (out_dir / rel).with_suffix(".txt")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(code.bit_string(), encoding="utf-8")
                if args.save_mask:
                    dest_mask = dest.parent / f"{dest.stem}_mask.txt"
                    dest_mask.write_text(code.mask_string(), encoding="utf-8")
                count += 1
        print(f"Wrote {count} binary iriscodes to '{out_dir}' (mirroring input folder structure).")

    if args.print_codes:
        total_imgs = sum(len(v) for v in used_paths.values())
        print(f"\n--- Binary IrisCodes ({total_imgs} images, 2048 bits each) ---")
        for cls in sorted(used_paths.keys()):
            for p, code in zip(used_paths[cls], encoded[cls]):
                rel = p.relative_to(args.dataset_root).as_posix()
                print(f"{rel}: {code.bit_string()}")
        print("--- End of Binary IrisCodes ---\n")

    if args.dump_codes is not None:
        with open(args.dump_codes, "w", encoding="utf-8") as fh:
            for cls in sorted(used_paths.keys()):
                for p, code in zip(used_paths[cls], encoded[cls]):
                    rel = p.relative_to(args.dataset_root).as_posix()
                    fh.write(f"{rel}: {code.bit_string()}\n")
        print(f"Wrote binary iriscodes to {args.dump_codes}")

    gen_raw, gen_records = genuine_distances(encoded, used_paths, max_shift=args.max_shift)
    imp_raw, imp_records = impostor_distances(encoded, used_paths, args.max_impostor_pairs,
                                              seed=args.seed, max_shift=args.max_shift)

    gen = summarize("Genuine  (same iris)", gen_raw)
    imp = summarize("Impostor (different iris)", imp_raw)

    # --- Diagnostics: valid-bit-overlap counts behind each comparison ---
    gen_n = np.array([r[4] for r in gen_records])
    imp_n = np.array([r[4] for r in imp_records])
    print(f"\nValid overlapping bits per comparison (out of 2048):")
    print(f"  Genuine : min={gen_n.min()}  mean={gen_n.mean():.0f}  max={gen_n.max()}")
    print(f"  Impostor: min={imp_n.min()}  mean={imp_n.mean():.0f}  max={imp_n.max()}")

    # Flag the most suspicious impostor pairs: unrelated irises should
    # never land near the genuine range. If they do, it's almost always
    # either (a) too few valid overlapping bits (statistical noise), or
    # (b) a systematic artifact shared across images (e.g. an
    # incompletely-masked specular reflection landing at the same grid
    # position regardless of identity).
    suspicious = sorted(imp_records, key=lambda r: r[3])[:5]
    print("\nLowest-HD impostor pairs (should be nowhere near the genuine range -- "
          "inspect these with visualize_segmentation.py if they look too low):")
    for cls_pair, p1, p2, hd, n_valid in suspicious:
        print(f"  {cls_pair}: {p1} vs {p2}  HD={hd:.4f}  n_valid_bits={n_valid}")

    worst_genuine = sorted(gen_records, key=lambda r: -r[3])[:8]
    print("\nWorst (highest-HD) genuine pairs (should be well below the impostor "
          "range -- these are your likely culprits for a long genuine tail / high "
          "FRR; inspect the listed images with visualize_segmentation.py):")
    for cls, p1, p2, hd, n_valid in worst_genuine:
        print(f"  {cls}: {p1} vs {p2}  HD={hd:.4f}  n_valid_bits={n_valid}")

    overlap_lo = max(gen.min(), imp.min())
    overlap_hi = min(gen.max(), imp.max())
    print(f"\nGenuine range:  [{gen.min():.4f}, {gen.max():.4f}]")
    print(f"Impostor range: [{imp.min():.4f}, {imp.max():.4f}]")
    if overlap_hi <= overlap_lo:
        print("No overlap between distributions -- perfect separation on this run.")
    else:
        print(f"Overlap region: [{overlap_lo:.4f}, {overlap_hi:.4f}]")

    d_prime = abs(imp.mean() - gen.mean()) / np.sqrt((gen.std() ** 2 + imp.std() ** 2) / 2)
    print(f"\nDecidability index d' = {d_prime:.3f}   (Daugman 1993 reports 8.4)")

    t = suggest_threshold(gen, imp)
    far = (imp < t).mean()
    frr = (gen >= t).mean()
    print(f"\nSuggested decision threshold (min FAR+FRR on this sample): HD < {t:.3f} = accept")
    print(f"  -> empirical FAR = {far:.4%}, FRR = {frr:.4%} at this threshold")

    plot_distributions(gen, imp, args.out)


if __name__ == "__main__":
    main()