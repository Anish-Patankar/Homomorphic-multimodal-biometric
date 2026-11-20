import os
from encrypt_int import encrypt_int

def encrypt(ek, m, kyberVariant = "ML-KEM-512"):
    # m = os.urandom(32)
    if m == None:
        return None
    return encrypt_int(ek,m,kyberVariant)