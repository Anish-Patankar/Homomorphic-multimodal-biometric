from pkpedecrypt import kPkeDecrypt
from pkpeencrypt import kPkeEncrypt
from utils.params import PARAMS
from utils.hash import G, J

def decrypt_int(dk, c, kyberVariant = "ML-KEM-512"):
    kyberParams = PARAMS[kyberVariant]
    k = kyberParams["k"]    
    dkPKE = dk[:384*k]
    ekPKE = dk[384*k:32+768*k]
    h = dk[768*k+32: 768*k+64]
    z = dk[768*k+64: 768*k+96]
    m = kPkeDecrypt(dkPKE,c,kyberVariant)
    hash = G(bytes(m+h)).digest()
    K = hash[:32]
    r = hash[32:]
    Kbar = J(bytes(z + c)).digest(32)
    cnew = kPkeEncrypt(ekPKE,m,r,kyberVariant)
    if c != cnew:
        K = Kbar
    
    return K