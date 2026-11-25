
import os
import json
import time
import argparse

from pathlib import Path
from itertools import combinations
from tqdm import tqdm

from pkpekeygen import kPkeKeyGen
from pkpeencrypt import kPkeEncrypt
from pkpedecrypt import kPkeDecrypt
from aheeval import aheeval


CHUNK_SIZE = 32


def load_dataset(dataset_path):

    dataset = {}

    files = [
        (folder, file)
        for folder in Path(dataset_path).iterdir()
        if folder.is_dir()
        for file in folder.glob("*.json")
    ]

    for folder, file in tqdm(
        files,
        desc="Loading dataset",
        unit="file"
    ):

        with open(file, "r") as f:
            data = json.load(f)

        bits = data["feature_vector_binary"].strip()

        if not bits or any(bit not in "01" for bit in bits):
            raise ValueError(f"Invalid feature vector: {file}")

        dataset[(folder.name, file.name)] = bits

    return dataset


def bits_to_bytes(bits):

    return bytes(
        int(bits[i:i + 8], 2)
        for i in range(0, len(bits), 8)
    )


def encrypt_feature_vector(bits, pk):

    ciphertexts = []

    plaintext = bits_to_bytes(bits)

    for i in range(0, len(plaintext), CHUNK_SIZE):

        chunk = plaintext[i:i + CHUNK_SIZE]

        ciphertext = kPkeEncrypt(
            pk,
            chunk,
            os.urandom(32)
        )

        ciphertexts.append(ciphertext)

    return ciphertexts


def main():

    parser = argparse.ArgumentParser(
        description="Face Kyber AHE"
    )

    parser.add_argument(
        "--face-dir",
        required=True
    )

    args = parser.parse_args()

    total_start = time.perf_counter()

    dataset_path = Path(args.face_dir)

    if not dataset_path.is_dir():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    # --------------------------------
    # 1. Load dataset
    # --------------------------------

    load_start = time.perf_counter()

    dataset = load_dataset(dataset_path)

    load_time = time.perf_counter() - load_start

    print(f"\nLoaded vectors: {len(dataset)}")

    # --------------------------------
    # 2. Pad feature vectors
    # --------------------------------

    max_length = max(map(len, dataset.values()))

    padded_length = ((max_length + 255) // 256) * 256

    dataset = {
        key: bits.ljust(padded_length, "0")
        for key, bits in dataset.items()
    }

    # --------------------------------
    # 3. Generate keys
    # --------------------------------

    print("\nGenerating keys...")

    start = time.perf_counter()

    keys = kPkeKeyGen(os.urandom(32))

    pk, sk = keys[0], keys[1]

    keygen_time = time.perf_counter() - start

    # --------------------------------
    # 4. Encrypt feature vectors
    # --------------------------------

    encrypted_dataset = {}

    encryption_start = time.perf_counter()

    for key, bits in tqdm(
        dataset.items(),
        total=len(dataset),
        desc="Encrypting vectors",
        unit="vector"
    ):

        encrypted_dataset[key] = encrypt_feature_vector(
            bits,
            pk
        )

    encryption_time = time.perf_counter() - encryption_start

    # --------------------------------
    # 5. Generate genuine pairs
    # --------------------------------

    subject_files = {}

    for (subject, filename), ciphertexts in encrypted_dataset.items():

        subject_files.setdefault(subject, []).append(
            (filename, ciphertexts)
        )

    pairs = []

    for subject, files in subject_files.items():

        for (file1, ct1), (file2, ct2) in combinations(files, 2):

            pairs.append(
                (subject, file1, file2, ct1, ct2)
            )

    print(f"\nTotal genuine pairs: {len(pairs)}")

    # --------------------------------
    # 6. Homomorphic XOR evaluation
    # --------------------------------

    results = []

    evaluation_time = 0.0
    decryption_time = 0.0

    with tqdm(
        total=len(pairs),
        desc="Evaluating encrypted XOR",
        unit="pair"
    ) as pbar:

        for subject, file1, file2, ct1, ct2 in pairs:

            # Homomorphic XOR
            start = time.perf_counter()

            xor_ciphertexts = [
                aheeval(c1, c2)
                for c1, c2 in zip(ct1, ct2)
            ]

            evaluation_time += time.perf_counter() - start

            # Decrypt XOR results
            start = time.perf_counter()

            distance = 0

            for xor_ct in xor_ciphertexts:

                xor_plaintext = kPkeDecrypt(sk, xor_ct)

                distance += sum(
                    byte.bit_count()
                    for byte in xor_plaintext
                )

            decryption_time += time.perf_counter() - start

            results.append({
                "subject": subject,
                "file1": file1,
                "file2": file2,
                "hamming_distance": distance
            })

            pbar.update(1)

            pbar.set_postfix({
                "eval": f"{evaluation_time:.1f}s",
                "decrypt": f"{decryption_time:.1f}s"
            })

    # --------------------------------
    # 7. Save results
    # --------------------------------

    output_file = dataset_path / "face_gen_pair_ahe.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    # --------------------------------
    # 8. Final statistics
    # --------------------------------

    total_time = time.perf_counter() - total_start

    print("\n" + "=" * 45)
    print("          PERFORMANCE STATISTICS")
    print("=" * 45)

    print(f"Dataset loading       : {load_time:.4f} sec")
    print(f"Key generation        : {keygen_time:.4f} sec")
    print(f"Encryption            : {encryption_time:.4f} sec")
    print(f"Homomorphic XOR       : {evaluation_time:.4f} sec")
    print(f"Decryption            : {decryption_time:.4f} sec")
    print("-" * 45)
    print(f"Total execution       : {total_time:.4f} sec")

    print("-" * 45)
    print(f"Feature vectors       : {len(dataset)}")
    print(f"Genuine pairs         : {len(results)}")
    print(f"Feature vector length : {padded_length} bits")
    print(f"Chunks per vector     : {padded_length // 256}")

    if pairs:
        print(
            f"Average pair time     : "
            f"{(evaluation_time + decryption_time) / len(pairs):.4f} sec"
        )

    print("=" * 45)

    print(f"\nResults saved: {output_file}")


if __name__ == "__main__":
    main()
