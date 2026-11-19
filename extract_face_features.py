#!/usr/bin/env python3
r"""
extract_face_features.py
========================
Batch-extracts facial feature vectors for images in a dataset (e.g. img_align_celeba)
and writes dual output per sample:
  1. JSON metadata file (<sample>.json):
     - Bounding box, head pose (yaw/pitch/roll), landmarks (68 feature points), quality
     - Binary bitstring: "feature_vector_binary" (0s and 1s)
     - Base64 encoding: "feature_vector_base64"
  2. Companion raw binary file (<sample>.bin):
     - Direct serialized template bytes for fast binary loading

Prerequisites
-------------
1. FaceFeatureExtractor.java must be compiled:
     javac -cp ".;C:\Program Files\Neurotechnology\...\Bin\Java\*" src\FaceFeatureExtractor.java
2. Neurotechnology SDK license server running with FaceExtractor or FaceClient.

Usage
-----
  # Extract first 1,000 images using 12 worker threads:
  python extract_face_features.py img_align_celeba celeba_features --dataset CelebA --max 1000 --threads 12 --skip-existing

  # Extract all images:
  python extract_face_features.py img_align_celeba celeba_features --dataset CelebA --threads 12 --skip-existing

  # Dry-run preview:
  python extract_face_features.py img_align_celeba celeba_features --max 20 --dry-run
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Force unbuffered output for real-time progress logging
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).parent.resolve()
JAVA_CLASS_DIR = SCRIPT_DIR / "src"
SDK_JAVA_DIR = Path(
    r"C:\Program Files\Neurotechnology\Neurotec_Biometric_2025_2_SDK\Bin\Java"
)
SDK_BIN_DIR = Path(
    r"C:\Program Files\Neurotechnology\Neurotec_Biometric_2025_2_SDK\Bin\Win64_x64"
)
JDK_DIR = Path(r"C:\Program Files\Java\jdk-25.0.4.1\bin")


def main():
    parser = argparse.ArgumentParser(
        description="Batch face feature extractor using Neurotec VeriLook SDK")
    parser.add_argument("images_root", type=Path,
                        help="Folder containing face images (e.g. img_align_celeba)")
    parser.add_argument("output_root", type=Path,
                        help="Folder for output JSON and BIN files")
    parser.add_argument("--dataset", default="CelebA",
                        help="Dataset name written into every JSON (default: CelebA)")
    parser.add_argument("--max", type=int, default=0,
                        help="Max number of images to process (0 = all, default: 0)")
    parser.add_argument("--threads", type=int, default=8,
                        help="Number of concurrent worker threads in Java (default: 8)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip images whose output JSON already exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="Scan and list files without extracting")
    args = parser.parse_args()

    images_root = args.images_root.resolve()
    output_root = args.output_root.resolve()

    if not images_root.exists():
        print(f"ERROR: Images folder not found: {images_root}")
        sys.exit(1)

    java_exe = JDK_DIR / "java.exe"
    if not java_exe.exists():
        java_exe = Path("java")

    classpath = f"{JAVA_CLASS_DIR};{SDK_JAVA_DIR}\\*"
    class_file = JAVA_CLASS_DIR / "FaceFeatureExtractor.class"

    if not class_file.exists():
        print(f"ERROR: Compiled class not found: {class_file}")
        print("  Compile first with:")
        print(f'  javac -cp ".;{SDK_JAVA_DIR}\\*" src\\FaceFeatureExtractor.java')
        sys.exit(1)

    output_root.mkdir(parents=True, exist_ok=True)

    # 1. Dry run
    if args.dry_run:
        print(f"=== DRY RUN: Scanning {images_root} ===")
        files = [p for p in images_root.glob("*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        files.sort()
        if args.max > 0:
            files = files[:args.max]
        for idx, f in enumerate(files[:20], 1):
            out_json = output_root / f"{f.stem}.json"
            out_bin  = output_root / f"{f.stem}.bin"
            print(f"[{idx:4d}] {f.name} -> {out_json.name} + {out_bin.name}")
        if len(files) > 20:
            print(f"... and {len(files) - 20} more images")
        print(f"Total matching images to process: {len(files)}")
        return

    # 2. Launch Fast Java In-Memory Batch Extractor
    cmd = [
        str(java_exe),
        "-cp", classpath,
        "FaceFeatureExtractor",
        "--batch",
        str(images_root),
        str(output_root),
        args.dataset,
        "--threads", str(args.threads),
    ]
    if args.max > 0:
        cmd.extend(["--max", str(args.max)])
    if args.skip_existing:
        cmd.append("--skip-existing")

    env = os.environ.copy()
    env["PATH"] = str(SDK_BIN_DIR) + os.pathsep + env.get("PATH", "")

    print(f"Starting batch extraction on: {images_root}")
    print(f"Output directory:            {output_root}")
    print(f"Worker threads:              {args.threads}")
    if args.max > 0:
        print(f"Image limit (--max):         {args.max}")
    print()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env
    )

    for line in proc.stdout:
        print(line, end="", flush=True)

    proc.wait()
    if proc.returncode != 0:
        print(f"\nExtraction exited with code: {proc.returncode}")
        sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
