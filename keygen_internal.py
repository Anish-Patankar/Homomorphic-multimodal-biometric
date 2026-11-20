from pkpekeygen import kPkeKeyGen
from utils.hash import H
import os

def keygen_int(d,z, kyberVariant = "ML-KEM-512"):
    keys = kPkeKeyGen(d,kyberVariant)
    encKey = keys[0]
    dkPKE = keys[1]
    decKey = dkPKE + encKey + H(encKey).digest() + z
    # print("inside internal keygen")
    print(encKey.hex())
    print(decKey.hex())

    return (encKey,decKey)
    


# keygen_int(os.urandom(32),os.urandom(32))