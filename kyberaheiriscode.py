
"""
iris_homomorphic_evaluation.py

Homomorphic Iris Recognition Evaluation
Dataset: CASIA Iris separated_iridcode

Calculates:
    1. Genuine Hamming distances
    2. Imposter Hamming distances
    3. FAR
    4. FRR
    5. GAR
    6. Optimal EER threshold
    7. Equal Error Rate (EER)

Outputs:
    iris_hamming_results.csv
    iris_eer_results.csv
    iris_hd_distributions.png
    iris_far_frr_eer.png
"""

from pkpekeygen import kPkeKeyGen
from pkpeencrypt import kPkeEncrypt
from pkpedecrypt import kPkeDecrypt
from aheeval import aheeval

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from itertools import combinations
from collections import defaultdict

from tqdm import tqdm


# =========================================================
# CONFIGURATION
# =========================================================

DATASET_PATH = Path("IIT iris seperated_iriscodes")

IRISCODE_BITS = 2048
BLOCK_SIZE = 32
NUM_BLOCKS = IRISCODE_BITS // (BLOCK_SIZE * 8)

NUM_THRESHOLDS = 10001

# =========================================================
# KEY GENERATION
# =========================================================

print("\nGenerating cryptographic keys...")

keys = kPkeKeyGen(os.urandom(32))

pk = keys[0]
sk = keys[1]

print("Key generation completed.")


# =========================================================
# READ IRIS CODES
# =========================================================

def load_iris_codes(dataset_path):

    iris_data = defaultdict(list)

    print("\nLoading iris codes...")

    for folder in tqdm(
        sorted(dataset_path.iterdir()),
        desc="Loading iris folders",
        unit="folder"
    ):

        if not folder.is_dir():
            continue

        for file in sorted(folder.glob("*.txt")):

            with open(file, "r") as f:
                iriscode = f.read().strip()

            if len(iriscode) != IRISCODE_BITS:
                print(f"Skipping invalid length: {file}")
                continue

            if not set(iriscode).issubset({"0", "1"}):
                print(f"Skipping invalid binary code: {file}")
                continue

            iris_bytes = int(iriscode, 2).to_bytes(
                IRISCODE_BITS // 8,
                byteorder="big"
            )

            iris_data[folder.name].append({
                "name": file.name,
                "bits": iriscode,
                "bytes": iris_bytes
            })

    iris_data = {
        folder: samples
        for folder, samples in iris_data.items()
        if len(samples) >= 2
    }

    print("Number of eye folders:", len(iris_data))

    total = sum(len(v) for v in iris_data.values())

    print("Total iris codes:", total)

    return iris_data


# =========================================================
# ENCRYPT IRIS CODES
# =========================================================

def encrypt_iris_codes(iris_data):

    encrypted_data = {}

    print("\nEncrypting iris codes...")

    total_samples = sum(len(samples) for samples in iris_data.values())

    with tqdm(total=total_samples, desc="Encrypting iris codes", unit="iris") as pbar:

        for folder, samples in iris_data.items():

            encrypted_data[folder] = []

            for sample in samples:

                iris_bytes = sample["bytes"]
                encrypted_blocks = []

                for i in range(NUM_BLOCKS):

                    start = i * BLOCK_SIZE
                    end = start + BLOCK_SIZE

                    message = iris_bytes[start:end]
                    randomness = os.urandom(32)

                    ciphertext = kPkeEncrypt(pk, message, randomness)

                    encrypted_blocks.append(ciphertext)

                encrypted_data[folder].append({
                    "name": sample["name"],
                    "bits": sample["bits"],
                    "ciphertexts": encrypted_blocks
                })

                pbar.update(1)

    print("Encryption completed.")

    return encrypted_data


# =========================================================
# HOMOMORPHIC HAMMING DISTANCE
# =========================================================

def encrypted_hamming_distance(sample1, sample2):

    total_distance = 0

    for c1, c2 in zip(
        sample1["ciphertexts"],
        sample2["ciphertexts"]
    ):

        # Homomorphic XOR
        cxor = aheeval(c1, c2)

        # Decrypt XOR result
        mxor = kPkeDecrypt(sk, cxor)

        if len(mxor) != BLOCK_SIZE:
            raise ValueError(
                f"Expected {BLOCK_SIZE} decrypted bytes, "
                f"got {len(mxor)}"
            )

        # Count differing bits
        total_distance += sum(
            byte.bit_count() for byte in mxor
        )

    return total_distance


# =========================================================
# GENUINE PAIR DISTANCES
# =========================================================

def calculate_genuine_distances(encrypted_data):

    genuine_distances = []
    genuine_records = []

    total_pairs = sum(
        len(list(combinations(samples, 2)))
        for samples in encrypted_data.values()
    )

    with tqdm(
        total=total_pairs,
        desc="Genuine comparisons",
        unit="pair"
    ) as pbar:

        for folder, samples in encrypted_data.items():

            for s1, s2 in combinations(samples, 2):

                hd = encrypted_hamming_distance(s1, s2)
                normalized_hd = hd / IRISCODE_BITS

                genuine_distances.append(normalized_hd)

                genuine_records.append([
                    folder, s1["name"], s2["name"],
                    hd, normalized_hd
                ])

                pbar.update(1)

    return genuine_distances, genuine_records

# =========================================================
# IMPOSTER PAIR DISTANCES
# =========================================================

def calculate_imposter_distances(encrypted_data):

    imposter_distances = []
    imposter_records = []

    folders = sorted(encrypted_data.keys())

    total_pairs = sum(
        len(encrypted_data[f1]) * len(encrypted_data[f2])
        for f1, f2 in combinations(folders, 2)
    )

    with tqdm(
        total=total_pairs,
        desc="Imposter comparisons",
        unit="pair"
    ) as pbar:

        for folder1, folder2 in combinations(folders, 2):

            samples1 = encrypted_data[folder1]
            samples2 = encrypted_data[folder2]

            for s1 in samples1:
                for s2 in samples2:

                    hd = encrypted_hamming_distance(s1, s2)
                    normalized_hd = hd / IRISCODE_BITS

                    imposter_distances.append(normalized_hd)

                    imposter_records.append([
                        folder1, s1["name"],
                        folder2, s2["name"],
                        hd, normalized_hd
                    ])

                    pbar.update(1)

    return imposter_distances, imposter_records

# =========================================================
# FAR AND FRR
# =========================================================

def calculate_far_frr(
    genuine_distances,
    imposter_distances,
    threshold
):

    genuine = np.asarray(genuine_distances)
    imposter = np.asarray(imposter_distances)

    # Imposter accepted as genuine
    false_accepts = np.sum(imposter < threshold)

    # Genuine rejected as imposter
    false_rejects = np.sum(genuine >= threshold)

    FAR = false_accepts / len(imposter)

    FRR = false_rejects / len(genuine)

    GAR = 1 - FRR

    return FAR, FRR, GAR


# =========================================================
# FIND OPTIMAL EER THRESHOLD
# =========================================================

def find_optimal_eer(
    genuine_distances,
    imposter_distances
):

    genuine = np.asarray(genuine_distances)
    imposter = np.asarray(imposter_distances)

    if len(genuine) == 0 or len(imposter) == 0:
        raise ValueError(
            "Both genuine and imposter comparisons are required."
        )

    thresholds = np.linspace(
        min(genuine.min(), imposter.min()),
        max(genuine.max(), imposter.max()),
        NUM_THRESHOLDS
    )

    results = []

    best_threshold = None
    best_far = None
    best_frr = None

    min_difference = float("inf")

    print("\nSearching for optimal EER threshold...")

    for threshold in tqdm(
        thresholds,
        desc="Finding optimal EER",
        unit="threshold"
    ):

        far, frr, gar = calculate_far_frr(
            genuine,
            imposter,
            threshold
        )

        difference = abs(far - frr)

        results.append([
            threshold,
            far,
            frr,
            gar
        ])

        if difference < min_difference:

            min_difference = difference

            best_threshold = threshold
            best_far = far
            best_frr = frr

    eer = (best_far + best_frr) / 2

    return (
        best_threshold,
        best_far,
        best_frr,
        eer,
        results
    )


# =========================================================
# PRINT STATISTICS
# =========================================================

def print_statistics(genuine, imposter):

    print("\n========== HAMMING DISTANCE STATISTICS ==========")

    print("\nGENUINE PAIRS")

    print("Count:", len(genuine))
    print("Mean:", np.mean(genuine))
    print("Standard deviation:", np.std(genuine))
    print("Minimum:", np.min(genuine))
    print("Maximum:", np.max(genuine))

    print("\nIMPOSTER PAIRS")

    print("Count:", len(imposter))
    print("Mean:", np.mean(imposter))
    print("Standard deviation:", np.std(imposter))
    print("Minimum:", np.min(imposter))
    print("Maximum:", np.max(imposter))

    d_prime = abs(
        np.mean(imposter) - np.mean(genuine)
    ) / np.sqrt(
        (np.std(genuine) ** 2 + np.std(imposter) ** 2) / 2
    )

    print("\nDecidability index d':", d_prime)


# =========================================================
# PLOT GENUINE AND IMPOSTER DISTRIBUTIONS
# =========================================================

def plot_distributions(genuine, imposter):

    plt.figure(figsize=(9, 6))

    bins = np.linspace(0, 0.5, 60)

    plt.hist(
        imposter,
        bins=bins,
        alpha=0.6,
        color="black",
        label=f"Imposter (n={len(imposter)})"
    )

    plt.hist(
        genuine,
        bins=bins,
        alpha=0.6,
        color="blue",
        label=f"Genuine (n={len(genuine)})"
    )

    plt.xlabel("Normalized Hamming Distance")
    plt.ylabel("Frequency")

    plt.title(
        "Genuine vs Imposter Iriscode Hamming Distances"
    )

    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        "iris_hd_distributions.png",
        dpi=300
    )

    plt.close()

    print("\nSaved iris_hd_distributions.png")


# =========================================================
# PLOT FAR, FRR AND EER
# =========================================================

def plot_far_frr(results, optimal_threshold, eer):

    thresholds = [r[0] for r in results]
    far_values = [r[1] for r in results]
    frr_values = [r[2] for r in results]

    plt.figure(figsize=(9, 6))

    plt.plot(
        thresholds,
        far_values,
        color="red",
        label="FAR"
    )

    plt.plot(
        thresholds,
        frr_values,
        color="blue",
        label="FRR"
    )

    plt.axvline(
        optimal_threshold,
        color="green",
        linestyle="--",
        label=f"Optimal Threshold = {optimal_threshold:.5f}"
    )

    plt.axhline(
        eer,
        color="black",
        linestyle=":",
        label=f"EER = {eer:.5f}"
    )

    plt.xlabel("Normalized Hamming Distance Threshold")
    plt.ylabel("Error Rate")

    plt.title("FAR and FRR vs Decision Threshold")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        "iris_far_frr_eer.png",
        dpi=300
    )

    plt.close()

    print("Saved iris_far_frr_eer.png")


# =========================================================
# SAVE COMPARISON RESULTS
# =========================================================

def save_comparison_results(
    genuine_records,
    imposter_records
):

    with open(
        "iris_hamming_results_IIT_D.csv",
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Comparison Type",
            "Folder 1",
            "Image 1",
            "Folder 2",
            "Image 2",
            "Hamming Distance",
            "Normalized Hamming Distance"
        ])

        for record in genuine_records:

            folder, img1, img2, hd, nhd = record

            writer.writerow([
                "Genuine",
                folder,
                img1,
                folder,
                img2,
                hd,
                nhd
            ])

        for record in imposter_records:

            folder1, img1, folder2, img2, hd, nhd = record

            writer.writerow([
                "Imposter",
                folder1,
                img1,
                folder2,
                img2,
                hd,
                nhd
            ])

    print("\nSaved iris_hamming_results.csv")


# =========================================================
# SAVE EER RESULTS
# =========================================================

def save_eer_results(results):

    with open(
        "iris_eer_results_IIT_D.csv",
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Threshold",
            "FAR",
            "FRR",
            "GAR"
        ])

        writer.writerows(results)

    print("Saved iris_eer_results.csv")


# =========================================================
# MAIN
# =========================================================

def main():

    if not DATASET_PATH.exists():

        raise SystemExit(
            f"Dataset path not found: {DATASET_PATH}"
        )

    # Load dataset
    iris_data = load_iris_codes(DATASET_PATH)

    if len(iris_data) < 2:

        raise SystemExit(
            "At least two folders with two or more valid "
            "iris codes each are required."
        )

    # Encrypt all iris codes
    encrypted_data = encrypt_iris_codes(iris_data)

    # Genuine comparisons
    genuine_distances, genuine_records = (
        calculate_genuine_distances(encrypted_data)
    )

    # Imposter comparisons
    imposter_distances, imposter_records = (
        calculate_imposter_distances(encrypted_data)
    )

    if not genuine_distances or not imposter_distances:

        raise SystemExit(
            "Insufficient genuine or imposter comparisons."
        )

    # Convert to NumPy arrays
    genuine = np.asarray(genuine_distances)
    imposter = np.asarray(imposter_distances)

    # Print statistics
    print_statistics(genuine, imposter)

    # Find optimal EER threshold
    (
        optimal_threshold,
        FAR,
        FRR,
        EER,
        roc_results
    ) = find_optimal_eer(
        genuine,
        imposter
    )

    # Print final results
    print("\n")
    print("=" * 55)
    print("             FINAL EER RESULTS")
    print("=" * 55)

    print(f"\nOptimal Threshold : {optimal_threshold:.8f}")

    print(f"FAR               : {FAR:.8f}")
    print(f"FRR               : {FRR:.8f}")
    print(f"GAR               : {1 - FRR:.8f}")

    print(f"\nFAR (%)           : {FAR * 100:.4f}%")
    print(f"FRR (%)           : {FRR * 100:.4f}%")
    print(f"EER (%)           : {EER * 100:.4f}%")

    print("\nDecision Rule:")
    print(
        f"Accept as genuine if HD < {optimal_threshold:.8f}"
    )
    print(
        f"Reject as imposter if HD >= {optimal_threshold:.8f}"
    )

    print("=" * 55)

    # Generate plots
    plot_distributions(
        genuine,
        imposter
    )

    plot_far_frr(
        roc_results,
        optimal_threshold,
        EER
    )

    # Save CSV files
    save_comparison_results(
        genuine_records,
        imposter_records
    )

    save_eer_results(
        roc_results
    )

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
