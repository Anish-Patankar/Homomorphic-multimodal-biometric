from decrypt_int import decrypt_int

def decrypt(dk, c, kyberVariant = "ML-KEM-512"):
    K = decrypt_int(dk,c,kyberVariant)
    return K