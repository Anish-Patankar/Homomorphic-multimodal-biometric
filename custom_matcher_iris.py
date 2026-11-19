
from pandas.io import parsers
import os
import json
import argparse
import itertools
import random

import numpy as np
import pandas as pd

from tqdm import tqdm

def load_selection_indices(selection_file, modality):

    data = np.load(selection_file)

    if modality == "iris":

        required_keys = [
            "iris_e",
            "iris_q",
            "iris_alpha",
            "selected_indices"
        ]

        for key in required_keys:
            if key not in data:
                raise KeyError(
                    f"Missing key '{key}' in {selection_file}"
                )

        iris_length = len(data["iris_e"])

        selected = data["selected_indices"]

        if np.any(selected >= iris_length):
            raise ValueError(
                "Selected indices exceed iris dimensions."
            )

        indices = selected

    elif modality == "face":

        raise ValueError(
            "Face selection is unavailable in this iris-only file."
        )

    else:
        raise ValueError("Modality must be iris or face.")

    indices = np.sort(indices)

    if len(indices) == 0:
        raise ValueError(
            f"No selected features available for {modality}"
        )

    return indices



def apply_feature_selection(features, selected_indices):

    selected_features = {}

    for identity, samples in features.items():

        selected_samples = []

        for sample in samples:

            vector = sample["vector"]

            if len(vector) <= np.max(selected_indices):
                raise ValueError(
                    f"Feature dimension mismatch for {identity}"
                )

            selected_vector = "".join(
                vector[i] for i in selected_indices
            )

            selected_samples.append({
                "file": sample["file"],
                "vector": selected_vector
            })

        selected_features[identity] = selected_samples

    return selected_features

# =====================================================
# LOAD IRIS FEATURES
# =====================================================

def load_features(dataset_path):

    features = {}

    all_files = []

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(".json"):
                all_files.append((root, file))

    for root, file in tqdm(
        all_files,
        desc="Loading iris dataset",
        unit="files",
        dynamic_ncols=True
    ):

        json_files = [
            f for f in files if f.lower().endswith(".json")
        ]

        for file in tqdm(
            json_files,
            desc=f"Loading {os.path.basename(root) or 'dataset'}",
            leave=False
        ):

            file_path = os.path.join(root, file)

            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                vector = data["feature_vector_binary"]

                if (
                    not isinstance(vector, str)
                    or not vector
                    or set(vector) - {"0", "1"}
                ):
                    print(f"\nSkipping invalid vector: {file_path}")
                    continue

                # Immediate subfolder represents identity
                relative_path = os.path.relpath(root, dataset_path)
                identity = relative_path.split(os.sep)[0]

                if identity == ".":
                    print(f"\nSkipping file outside identity folders: {file_path}")
                    continue

                features.setdefault(identity, []).append({
                    "file": file_path,
                    "vector": vector
                })

            except (json.JSONDecodeError, KeyError, OSError) as e:
                print(f"\nError reading {file_path}: {e}")

    return features


# =====================================================
# HAMMING DISTANCE AND XOR
# =====================================================

def calculate_hamming(v1, v2):

    if len(v1) != len(v2):
        raise ValueError("Feature vectors have different lengths.")

    xor_result = "".join(
        str(int(a) ^ int(b))
        for a, b in zip(v1, v2)
    )

    hamming_distance = xor_result.count("1")

    normalized_distance = (
        hamming_distance / len(v1)
    )

    return xor_result, hamming_distance, normalized_distance


# =====================================================
# CALCULATE GENUINE AND IMPOSTER SCORES
# =====================================================

def calculate_scores(features, imposter_limit=None, seed=42):

    genuine_scores = []
    imposter_scores = []

    genuine_details = []
    imposter_details = []

    identities = sorted(features.keys())

    # -------------------------------------------------
    # GENUINE PAIRS
    # -------------------------------------------------

    print("\nCalculating genuine pairs...")

    total_genuine = sum(
        len(samples) * (len(samples) - 1) // 2
        for samples in features.values()
    )

    with tqdm(
        total=total_genuine,
        desc="Genuine comparisons",
        unit="pairs",
        dynamic_ncols=True
    ) as pbar:

        for identity in identities:

            samples = features[identity]

            for a, b in itertools.combinations(samples, 2):

                xor, hd, nhd = calculate_hamming(
                    a["vector"],
                    b["vector"]
                )

                genuine_scores.append(nhd)

                genuine_details.append({
                    "identity": identity,
                    "file1": a["file"],
                    "file2": b["file"],
                    "xor": xor,
                    "hamming_distance": hd,
                    "normalized_hamming_distance": nhd
                })

                pbar.update(1)
    # -------------------------------------------------
    # IMPOSTER PAIRS
    # -------------------------------------------------

    print("\nCalculating imposter pairs...")

    identity_pairs = list(
        itertools.combinations(identities, 2)
    )

    total_imposter_pairs = sum(
        len(features[id1]) * len(features[id2])
        for id1, id2 in identity_pairs
    )

    if imposter_limit is None:
        limit = total_imposter_pairs
    else:
        limit = min(imposter_limit, total_imposter_pairs)

    print(f"Total possible imposter pairs: {total_imposter_pairs}")
    print(f"Imposter pairs to calculate: {limit}")

    # Build the sampling population only when sampling is needed.
    rng = random.Random(seed)

    if limit < total_imposter_pairs:

        sampled_indices = sorted(
            rng.sample(range(total_imposter_pairs), limit)
        )

        selected_pairs = []
        pair_index = 0
        sample_index = 0

        for id1, id2 in identity_pairs:

            samples1 = features[id1]
            samples2 = features[id2]

            pair_count = len(samples1) * len(samples2)

            while (
                sample_index < len(sampled_indices)
                and sampled_indices[sample_index] < pair_index + pair_count
            ):

                local_index = (
                    sampled_indices[sample_index] - pair_index
                )

                a_index = local_index // len(samples2)
                b_index = local_index % len(samples2)

                selected_pairs.append((
                    id1,
                    id2,
                    samples1[a_index],
                    samples2[b_index]
                ))

                sample_index += 1

            pair_index += pair_count

    else:

        selected_pairs = (
            (id1, id2, a, b)
            for id1, id2 in identity_pairs
            for a, b in itertools.product(features[id1], features[id2])
        )

    # Calculate selected imposter pairs
    for id1, id2, a, b in tqdm(
        selected_pairs,
        total=limit,
        desc="Impostor comparisons",
        unit="pairs",
        dynamic_ncols=True
    ):

        xor, hd, nhd = calculate_hamming(
            a["vector"],
            b["vector"]
        )

        imposter_scores.append(nhd)

        imposter_details.append({
            "identity1": id1,
            "identity2": id2,
            "file1": a["file"],
            "file2": b["file"],
            "xor": xor,
            "hamming_distance": hd,
            "normalized_hamming_distance": nhd
        })

    return (
        genuine_scores,
        imposter_scores,
        genuine_details,
        imposter_details
    )


# =====================================================
# FAR AND FRR
# =====================================================

def calculate_far_frr(
    genuine_scores,
    imposter_scores,
    threshold
):

    genuine = np.asarray(genuine_scores)
    imposter = np.asarray(imposter_scores)

    # Lower Hamming distance = genuine match
    far = np.mean(imposter <= threshold)

    frr = np.mean(genuine > threshold)

    return far, frr


# =====================================================
# FIND OPTIMAL THRESHOLD AND EER
# =====================================================

def find_optimal_threshold(
    genuine_scores,
    imposter_scores
):

    all_scores = np.concatenate([
        genuine_scores,
        imposter_scores
    ])

    thresholds = np.unique(all_scores)

    results = []

    print("\nEvaluating thresholds...")

    for threshold in tqdm(
        thresholds,
        desc="Threshold evaluation",
        unit="thresholds",
        dynamic_ncols=True
    ):

        far, frr = calculate_far_frr(
            genuine_scores,
            imposter_scores,
            threshold
        )

        results.append({
            "threshold": threshold,
            "FAR": far,
            "FRR": frr,
            "difference": abs(far - frr)
        })

    results_df = pd.DataFrame(results)

    best_index = results_df["difference"].idxmin()

    best = results_df.loc[best_index]

    eer = (best["FAR"] + best["FRR"]) / 2

    return (
        best["threshold"],
        best["FAR"],
        best["FRR"],
        eer,
        results_df
    )


# =====================================================
# MAIN
# =====================================================

def main():

    parser = argparse.ArgumentParser(
        description="Iris biometric evaluation"
    )

    parser.add_argument(
        "dataset_path",
        help="Path to the dataset directory"
    )

    parser.add_argument(
        "--imposter-limit",
        type=int,
        default=None,
        help="Maximum number of imposter pairs (default: all pairs)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible imposter sampling"
    )

    parser.add_argument(
        "--selection-file",
        required=True,
        help="Path to bit_selection.npz"
    )

    parser.add_argument(
        "--modality",
        choices=["iris", "face"],
        required=True,
        help="Feature modality to evaluate"
    )

    args = parser.parse_args()

    dataset_path = args.dataset_path

    if not os.path.isdir(dataset_path):
        raise ValueError(
            f"Invalid dataset path: {dataset_path}"
        )

    if args.imposter_limit is not None and args.imposter_limit < 1:
        raise ValueError("--imposter-limit must be at least 1.")

    # -------------------------------------------------
    # LOAD DATASET
    # -------------------------------------------------

    print("\n========== LOADING DATASET ==========")

    features = load_features(dataset_path)

    print("\n========== DATASET SUMMARY ==========")

    print(f"Total identities: {len(features)}")

    for identity, samples in features.items():
        print(f"{identity}: {len(samples)} images")

    # -------------------------------------------------
    # VALIDATE FEATURE VECTORS
    # -------------------------------------------------

    lengths = {
        len(sample["vector"])
        for samples in features.values()
        for sample in samples
    }

    if not lengths:
        raise ValueError("No valid feature vectors found.")

    if len(lengths) != 1:
        raise ValueError(
            f"Inconsistent feature vector lengths: {lengths}"
        )

    feature_length = next(iter(lengths))

    print(f"\nFeature vector length: {feature_length}")

    # -------------------------------------------------
    # APPLY TOP-K FEATURE SELECTION
    # -------------------------------------------------

    selected_indices = load_selection_indices(
        args.selection_file,
        args.modality
    )

    original_feature_length = feature_length

    features = apply_feature_selection(
        features,
        selected_indices
    )

    feature_length = len(selected_indices)

    print("\n========== FEATURE SELECTION ==========")

    print("Modality:", args.modality)
    print("Original feature length:", original_feature_length)
    print("Selected feature length:", feature_length)

    print(
        "Feature reduction:",
        f"{(1 - feature_length / original_feature_length) * 100:.2f}%"
    )

    # -------------------------------------------------
    # CALCULATE SCORES
    # -------------------------------------------------

    (
        genuine,
        imposter,
        genuine_details,
        imposter_details
    ) = calculate_scores(
        features,
        imposter_limit=args.imposter_limit,
        seed=args.seed
    )

    print("\n========== COMPARISON SUMMARY ==========")

    print(f"Genuine pairs: {len(genuine)}")
    print(f"Imposter pairs: {len(imposter)}")

    if not genuine:
        raise ValueError("No genuine pairs found.")

    if not imposter:
        raise ValueError("No imposter pairs found.")

    # -------------------------------------------------
    # CALCULATE METRICS
    # -------------------------------------------------

    (
        threshold,
        far,
        frr,
        eer,
        results_df
    ) = find_optimal_threshold(
        genuine,
        imposter
    )

    # -------------------------------------------------
    # DISPLAY RESULTS
    # -------------------------------------------------

    print("\n========== FINAL RESULTS ==========")

    print(f"Feature vector length: {feature_length}")

    print(f"Genuine pairs: {len(genuine)}")
    print(f"Imposter pairs: {len(imposter)}")

    print("\nDistance statistics:")

    print(f"Average genuine distance: {np.mean(genuine):.6f}")
    print(f"Average imposter distance: {np.mean(imposter):.6f}")

    print(f"Minimum genuine distance: {np.min(genuine):.6f}")
    print(f"Maximum genuine distance: {np.max(genuine):.6f}")

    print(f"Minimum imposter distance: {np.min(imposter):.6f}")
    print(f"Maximum imposter distance: {np.max(imposter):.6f}")

    print("\nRecognition performance:")

    print(f"Optimal threshold: {threshold:.6f}")
    print(f"FAR: {far * 100:.4f}%")
    print(f"FRR: {frr * 100:.4f}%")
    print(f"EER estimate: {eer * 100:.4f}%")

    # -------------------------------------------------
    # SAVE RESULTS
    # -------------------------------------------------

    print("\nSaving results...")

    pd.DataFrame(genuine_details).to_csv(
        "genuine_comparisons_iris_reduced.csv",
        index=False
    )

    pd.DataFrame(imposter_details).to_csv(
        "imposter_comparisons_iris_reduced.csv",
        index=False
    )

    results_df.to_csv(
        "threshold_results_iris_reduced.csv",
        index=False
    )

    summary = pd.DataFrame([{
        "feature_length": feature_length,
        "genuine_pairs": len(genuine),
        "imposter_pairs": len(imposter),
        "average_genuine_distance": np.mean(genuine),
        "average_imposter_distance": np.mean(imposter),
        "optimal_threshold": threshold,
        "FAR": far,
        "FRR": frr,
        "EER": eer,
        "imposter_limit": args.imposter_limit,
        "random_seed": args.seed
    }])

    summary.to_csv(
        "evaluation_summary_iris_reduced.csv",
        index=False
    )

    print("\n========== COMPLETE ==========")

    print("Generated files:")
    print("1. genuine_comparisons_iris.csv")
    print("2. imposter_comparisons_iris.csv")
    print("3. threshold_results_iris.csv")
    print("4. evaluation_summary_iris.csv")


if __name__ == "__main__":
    main()
