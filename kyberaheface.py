from pkpekeygen import kPkeKeyGen
from pkpeencrypt import kPkeEncrypt
from pkpedecrypt import kPkeDecrypt
from aheeval import aheeval
import os
import pickle
import json
import argparse
from itertools import product, combinations


def pad_for_kyber(fv):
    rem = len(fv) % 256
    while rem>0 and rem<256:
        fv+="0"
        rem+=1
    
    return fv

def hamming_distance(v1,v2):
    return sum(bit1 != bit2 for bit1,bit2 in zip(v1,v2))

def genuine(dir_path):
    genuine_pairs = {}
    for folder in os.listdir(dir_path):
        folder_path = os.path.join(dir_path, folder)
        
        if not os.path.isdir(folder_path):
            continue

        json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
        
        for file1, file2 in combinations(json_files,2):
            fpath1 = os.path.join(folder_path,file1)
            fpath2 = os.path.join(folder_path,file2)

            with open(fpath1, "r") as f:
                data1 = json.load(f)
            
            with open(fpath2, "r") as f:
                data2 = json.load(f)

            fv1 = pad_for_kyber(data1["feature_vector_binary"])
            fv2 = pad_for_kyber(data2["feature_vector_binary"])

            genuine_pairs[(folder,file1,file2)] = (
                fv1, fv2,  hamming_distance(fv1,fv2)      
            )
    
    # with open("face_gen_pair.json","w") as f:
    #     json.dump(genuine_pairs,f,indent = 4)
    print(genuine_pairs)
    with open(os.path.join(dir_path,"face_gen_pair.txt"),"w") as f:
        for key, value in genuine_pairs.items():
            f.write(f"{key}: {value}\n")

    return 

def main():

    parser = argparse.ArgumentParser(
        description="Face Kyber AHE"
    )

    parser.add_argument(
        "--face-dir",
        required=True,
        help="Directory containing face subject folders"
    )

    args = parser.parse_args()

    keys = kPkeKeyGen(os.urandom(32))
    # print(keys)
    pk = keys[0]
    sk = keys[1]    

    DATASET_PATH = args.face_dir

    if not os.path.exists(DATASET_PATH):

        raise SystemExit(
            f"Dataset path not found: {DATASET_PATH}"
        )

    # Load dataset
    

    genuine(DATASET_PATH)


    # for _ in range(80):
    #     m1 = os.urandom(32)
    #     m2 = os.urandom(32)
    #     # m1 = bytes([0]*32)
    #     # m2 = bytes([0]*32)
    #     r1 = os.urandom(32)
    #     r2 = os.urandom(32)
    #     c1 = kPkeEncrypt(pk,m1,r1)
    #     c2 = kPkeEncrypt(pk,m2,r2)
    #     cxor = aheeval(c1,c2)
    #     mxor = kPkeDecrypt(sk,cxor)
    #     expected = bytes(a ^ b for a,b in zip(m1,m2))
    #     print("asserting")
    #     assert expected == mxor
    # print(expected.hex())
    # print(mxor.hex())
    # print(len(expected.hex()))
    # print(len(mxor.hex()))

if __name__ == "__main__":
    main()