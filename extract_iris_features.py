#!/usr/bin/env python3
r"""
extract_iris_features.py
========================
Batch-extracts iris feature vectors for every image in the organised
iris dataset and writes JSON output per sample.

Matches the architecture and output conventions of the FingerprintFeatureExtractor
pipeline, formatted for subsequent FAR/FRR score computation.

Assumes the organised dataset follows one of these conventions:

  Convention A  (IIT Iris / organiseIris1.py / organiseIris2.py):
    <images_root>\<SubjectSide>\<image>.bmp
    e.g.  "IIT iris seperated\001left\01.bmp"
          "IIT iris seperated\001right\06.bmp"

  Convention B  (flat per-subject folders):
    <images_root>\Subject_<N>\<image>.bmp

Output is written to:
    <output_root>\<SubjectSide>\<image_stem>.json (metadata, binary bitstring, base64)
    <output_root>\<SubjectSide>\<image_stem>.bin  (raw binary template bytes)
    e.g.  Iris_Features\001left\01.json
          Iris_Features\001left\01.bin

Prerequisites
-------------
1. IrisFeatureExtractor.java must be compiled:
     javac -cp ".;C:\Program Files\Neurotechnology\...\Bin\Java\*" src\IrisFeatureExtractor.java
2. Neurotechnology SDK license server running with IrisExtractor or IrisClient.

Usage
-----
  # Fast in-memory batch extraction (recommended, ~1-2 min for 2,240 images):
  python extract_iris_features.py "IIT iris seperated" Iris_Features --dataset IIT_Iris --skip-existing

  # Dry run to preview discovered images:
  python extract_iris_features.py "IIT iris seperated" Iris_Features --dry-run
"""

import argparse
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Force unbuffered output for real-time progress logging
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Paths — adjust these to match your environment
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()

JAVA_CLASS_DIR = SCRIPT_DIR / "src"
SDK_JAVA_DIR = Path(
    r"C:\Program Files\Neurotechnology\Neurotec_Biometric_2025_2_SDK\Bin\Java"
)
SDK_BIN_DIR = Path(
    r"C:\Program Files\Neurotechnology\Neurotec_Biometric_2025_2_SDK\Bin\Win64_x64"
)
JDK_DIR = Path(r"C:\Program Files\Java\jdk-25.0.4.1\bin")

IMAGE_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".wsq"}


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------
def find_images(images_root: Path):
    r"""
    Yield (subject_id, sample_id, side_label, image_path) for every iris image.

    Handles two directory conventions:
      A) <images_root>/<SubjectSide>/<file>.ext
         SubjectSide like "001left", "001right", "42left", etc.
      B) <images_root>/Subject_<N>/<file>.ext
    """
    folder_re_a = re.compile(r"^(\d+)(left|right)$", re.IGNORECASE)
    folder_re_b = re.compile(r"^Subject_(\d+)$", re.IGNORECASE)

    for folder in sorted(images_root.iterdir()):
        if not folder.is_dir():
            continue

        ma = folder_re_a.match(folder.name)
        mb = folder_re_b.match(folder.name)

        if ma:
            subject_id = ma.group(1)
            side_label = ma.group(2).lower()
        elif mb:
            subject_id = mb.group(1)
            side_label = ""
        else:
            continue

        for img_file in sorted(folder.iterdir()):
            if not img_file.is_file():
                continue
            if img_file.suffix.lower() not in IMAGE_EXTS:
                continue

            sample_id = img_file.stem

            inferred_side = side_label
            if not inferred_side:
                if "L" in img_file.stem.upper():
                    inferred_side = "left"
                elif "R" in img_file.stem.upper():
                    inferred_side = "right"
                else:
                    inferred_side = "default"

            yield subject_id, sample_id, inferred_side, img_file


# ---------------------------------------------------------------------------
# Worker (per-image fallback)
# ---------------------------------------------------------------------------
def run_extraction(task):
    r"""Worker function for legacy per-image invocation."""
    (java_exe, classpath, image_path, output_path,
     subject_id, sample_id, dataset, subdataset) = task

    cmd = [
        str(java_exe),
        "-cp", classpath,
        "IrisFeatureExtractor",
        str(image_path),
        str(output_path),
        subject_id,
        sample_id,
        dataset,
        subdataset,
    ]

    env = os.environ.copy()
    env["PATH"] = str(SDK_BIN_DIR) + os.pathsep + env.get("PATH", "")

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return (
        str(image_path),
        result.returncode,
        result.stdout.strip(),
        result.stderr.strip(),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Batch iris feature extractor using Neurotec VeriEye SDK")
    parser.add_argument("images_root", type=Path,
                        help="Root folder containing organised iris images")
    parser.add_argument("output_root", type=Path,
                        help="Root folder for JSON output")
    parser.add_argument("--dataset", default="IrisDataset",
                        help="Dataset name written into every JSON (default: IrisDataset)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip images whose output JSON already exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned actions without executing")
    parser.add_argument("--legacy-workers", action="store_true",
                        help="Use multi-process per-image calls instead of fast Java batch")
    parser.add_argument("--jobs", type=int, default=4,
                        help="Number of workers if using --legacy-workers (default: 4)")
    args = parser.parse_args()

    images_root = args.images_root.resolve()
    output_root = args.output_root.resolve()

    java_exe = JDK_DIR / "java.exe"
    if not java_exe.exists():
        java_exe = Path("java")

    classpath = f"{JAVA_CLASS_DIR};{SDK_JAVA_DIR}\\*"
    class_file = JAVA_CLASS_DIR / "IrisFeatureExtractor.class"

    if not class_file.exists():
        print(f"ERROR: Compiled class not found: {class_file}", flush=True)
        print("  Compile first with:")
        print(f'  javac -cp ".;{SDK_JAVA_DIR}\\*" src\\IrisFeatureExtractor.java')
        sys.exit(1)

    if not images_root.exists():
        print(f"ERROR: Images folder not found: {images_root}", flush=True)
        sys.exit(1)

    output_root.mkdir(parents=True, exist_ok=True)

    # 1. Dry run
    if args.dry_run:
        print(f"=== DRY RUN: Scanning {images_root} ===", flush=True)
        count = 0
        for subj, samp, side, path in find_images(images_root):
            count += 1
            folder_name = f"{subj.zfill(3)}{side}" if side else f"Subject_{subj}"
            out_file = output_root / folder_name / f"{samp}.json"
            print(f"[{count:>5}] {path} -> {out_file}", flush=True)
        print(f"\nTotal discovered images: {count}", flush=True)
        return

    # 2. Fast Java In-Memory Batch Mode (Default)
    if not args.legacy_workers:
        print("Using Ultra-Fast In-Memory Java Batch Mode...", flush=True)
        cmd = [
            str(java_exe),
            "-cp", classpath,
            "IrisFeatureExtractor",
            "--batch",
            str(images_root),
            str(output_root),
            args.dataset,
        ]
        if args.skip_existing:
            cmd.append("--skip-existing")

        env = os.environ.copy()
        env["PATH"] = str(SDK_BIN_DIR) + os.pathsep + env.get("PATH", "")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        for line in proc.stdout:
            print(line, end="", flush=True)

        proc.wait()
        if proc.returncode != 0:
            print(f"Extraction failed with exit code: {proc.returncode}", flush=True)
            sys.exit(proc.returncode)
        return

    # 3. Legacy Multi-Process Worker Mode (if explicitly requested)
    tasks = []
    for subject_id, sample_id, side_label, image_path in find_images(images_root):
        folder_name = f"{subject_id.zfill(3)}{side_label}" if side_label else f"Subject_{subject_id}"
        output_dir = output_root / folder_name
        output_path = output_dir / f"{sample_id}.json"
        subdataset = side_label.capitalize() if side_label else "Default"

        if args.skip_existing and output_path.exists():
            continue

        tasks.append((
            java_exe, classpath,
            image_path, output_path,
            subject_id, sample_id,
            args.dataset, subdataset,
        ))

    total = len(tasks)
    print(f"Images to process : {total}", flush=True)
    print(f"Parallel workers  : {args.jobs}", flush=True)

    if total == 0:
        print("Nothing to do.", flush=True)
        return

    ok = failed = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_extraction, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            img_str, rc, stdout, stderr = fut.result()
            tag = f"[{i:>6}/{total}]"
            if rc == 0:
                ok += 1
                print(f"{tag} OK   {img_str}", flush=True)
            else:
                failed += 1
                print(f"{tag} FAIL (rc={rc}) {img_str}", flush=True)

    print(f"\nFinished.  OK={ok}  FAILED={failed}  TOTAL={total}", flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
