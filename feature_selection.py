
import os
import json
import argparse
import numpy as np

from pathlib import Path
from itertools import combinations


# ============================================================
# 1. LOAD JSON FEATURE VECTOR
# ============================================================


def load_json_features(json_path, feature_key="feature_vector_binary"):

    with open(json_path, "r") as f:
        data = json.load(f)

    if feature_key not in data:
        raise KeyError(
            f"Key '{feature_key}' not found in {json_path}. "
            f"Available keys: {list(data.keys())}"
        )

    feature_string = data[feature_key]

    if not isinstance(feature_string, str):
        raise TypeError(
            f"Expected string in {json_path}, "
            f"got {type(feature_string).__name__}"
        )

    if not feature_string or not set(feature_string).issubset({"0", "1"}):
        raise ValueError(
            f"Invalid binary feature vector in {json_path}"
        )

    return np.fromiter(
        (int(bit) for bit in feature_string),
        dtype=np.uint8
    )

# ============================================================
# 2. LOAD DIRECTORY
# ============================================================

def load_feature_directory(directory, feature_key=None):

    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(directory)

    subjects = []

    subject_names = []

    for subject_dir in sorted(directory.iterdir()):

        if not subject_dir.is_dir():
            continue

        json_files = sorted(
            subject_dir.glob("*.json")
        )

        if not json_files:
            continue

        captures = []

        for json_file in json_files:

            features = load_json_features(
                json_file,
                feature_key
            )

            captures.append(features)

        # Ensure all captures have the same dimensions
        dimensions = {len(x) for x in captures}

        if len(dimensions) != 1:
            raise ValueError(
                f"Inconsistent feature lengths in {subject_dir}"
            )

        subjects.append(np.stack(captures))

        subject_names.append(subject_dir.name)

        print(
            f"Loaded {subject_dir.name}: "
            f"{len(captures)} captures"
        )

    if not subjects:
        raise ValueError(
            f"No valid subject folders found in {directory}"
        )

    # Ensure all subjects have equal feature dimensions
    dimensions = {x.shape[1] for x in subjects}

    if len(dimensions) != 1:
        raise ValueError(
            f"Inconsistent feature dimensions in {directory}"
        )

    return subjects, subject_names


# ============================================================
# 3. CALCULATE GENUINE AND IMPOSTOR BIT STATISTICS
# ============================================================

def bit_stats(subjects, smoothing=1.0):

    n_bits = subjects[0].shape[1]

    genuine_flips = np.zeros(
        n_bits,
        dtype=np.float64
    )

    genuine_pairs = 0

    # Genuine comparisons
    for X in subjects:

        for i, j in combinations(range(len(X)), 2):

            genuine_flips += np.logical_xor(
                X[i], X[j]
            )

            genuine_pairs += 1

    if genuine_pairs == 0:
        raise ValueError(
            "Need at least one subject with two captures."
        )

    e = (
        genuine_flips + smoothing
    ) / (
        genuine_pairs + 2 * smoothing
    )

    # Estimate impostor difference rate directly
    # using different subject groups.

    impostor_differences = np.zeros(
        n_bits,
        dtype=np.float64
    )

    impostor_pairs = 0

    for i in range(len(subjects)):

        for j in range(i + 1, len(subjects)):

            for a in subjects[i]:

                for b in subjects[j]:

                    impostor_differences += np.logical_xor(
                        a, b
                    )

                    impostor_pairs += 1

    if impostor_pairs == 0:
        raise ValueError(
            "Need at least two subjects for impostor statistics."
        )

    q = (
        impostor_differences + smoothing
    ) / (
        impostor_pairs + 2 * smoothing
    )

    e = np.clip(e, 1e-3, 1 - 1e-3)
    q = np.clip(q, 1e-3, 1 - 1e-3)

    return e, q


# ============================================================
# 4. CALCULATE BIT WEIGHTS
# ============================================================

def calculate_weights(e, q):

    numerator = (1 - e) * (1 - q)

    denominator = e * q

    alpha = 0.5 * np.log(
        numerator / denominator
    )

    alpha = np.maximum(0, alpha)

    return alpha


# ============================================================
# 5. COMPACT GLOBAL MASK
# ============================================================

def create_selection_mask(iris_alpha, face_alpha):

    alpha_all = np.concatenate([
        iris_alpha,
        face_alpha
    ])

    selected_indices = np.flatnonzero(
        alpha_all > 0
    )

    return alpha_all, selected_indices


# ============================================================
# 6. SAVE RESULTS
# ============================================================

def save_results(
    output_dir,
    iris_e,
    iris_q,
    face_e,
    face_q,
    alpha_all,
    selected_indices
):

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        output_dir / "bit_selection.npz",

        iris_e=iris_e,
        iris_q=iris_q,

        face_e=face_e,
        face_q=face_q,

        alpha_all=alpha_all,

        selected_indices=selected_indices
    )

    print("\nResults saved to:", output_dir)


# ============================================================
# 7. MAIN PIPELINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Iris and Face Feature Selection"
    )

    parser.add_argument(
        "--iris-dir",
        required=True,
        help="Directory containing iris subject folders"
    )

    parser.add_argument(
        "--face-dir",
        required=True,
        help="Directory containing face subject folders"
    )

    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory"
    )

    parser.add_argument(
        "--iris-key",
        default="feature_vector_binary",
        help="JSON key containing iris features"
    )

    parser.add_argument(
        "--face-key",
        default="feature_vector_binary",
        help="JSON key containing face features"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # LOAD IRIS
    # --------------------------------------------------------

    print("\nLoading iris features...")

    iris_subjects, iris_names = load_feature_directory(
        args.iris_dir,
        args.iris_key
    )

    # --------------------------------------------------------
    # LOAD FACE
    # --------------------------------------------------------

    print("\nLoading face features...")

    face_subjects, face_names = load_feature_directory(
        args.face_dir,
        args.face_key
    )

    # --------------------------------------------------------
    # VALIDATE SUBJECT COUNTS
    # --------------------------------------------------------

    print("\nIris subjects:", len(iris_subjects))

    print("Face subjects:", len(face_subjects))

    # --------------------------------------------------------
    # CALCULATE IRIS STATISTICS
    # --------------------------------------------------------

    print("\nCalculating iris bit statistics...")

    iris_e, iris_q = bit_stats(
        iris_subjects
    )

    # --------------------------------------------------------
    # CALCULATE FACE STATISTICS
    # --------------------------------------------------------

    print("\nCalculating face bit statistics...")

    face_e, face_q = bit_stats(
        face_subjects
    )

    # --------------------------------------------------------
    # CALCULATE WEIGHTS
    # --------------------------------------------------------

    print("\nCalculating feature weights...")

    iris_alpha = calculate_weights(
        iris_e,
        iris_q
    )

    face_alpha = calculate_weights(
        face_e,
        face_q
    )

    # --------------------------------------------------------
    # COMBINE AND SELECT
    # --------------------------------------------------------

    alpha_all, selected_indices = create_selection_mask(
        iris_alpha,
        face_alpha
    )

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    total_bits = len(alpha_all)

    selected_bits = len(selected_indices)

    print("\n========== RESULTS ==========")

    print("Total iris bits:", len(iris_alpha))

    print("Selected iris bits:",
          np.count_nonzero(iris_alpha > 0))

    print("Total face bits:", len(face_alpha))

    print("Selected face bits:",
          np.count_nonzero(face_alpha > 0))

    print("Total combined bits:", total_bits)

    print("Selected combined bits:", selected_bits)

    print("Removed bits:", total_bits - selected_bits)

    print(
        "Original ciphertext chunks:",
        int(np.ceil(total_bits / 4096))
    )

    print(
        "Selected ciphertext chunks:",
        int(np.ceil(selected_bits / 4096))
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_results(
        args.output_dir,

        iris_e,
        iris_q,

        face_e,
        face_q,

        alpha_all,

        selected_indices
    )


if __name__ == "__main__":
    main()
