from pkpeencrypt import kPkeEncrypt
from utils.hash import H, G

def encrypt_int(ek, m, kyberVariant = "ML-KEM-512"):
    hash = G(m + H(ek).digest()).digest()
    K = hash[:32]
    r = hash[32:]
    c = kPkeEncrypt(ek,m,r,kyberVariant)
    return (K,c)
