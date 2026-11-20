from utils.params import Q, N, ZETA, PARAMS
from utils.hash import G, H, J, PRF, XOF
from utils.sampling import sampleNTT, samplePolyCBD
from utils.matrixops import matrix_vector_multiply_ntt, polyvec_add_ntt
from utils.compression import ByteEncode_d
import os
from ntt import NTTContext
from fips_ntt import ntt

def kPkeKeyGen(d: bytes, kyberVariant = "ML-KEM-512"):
    kyberParams = PARAMS[kyberVariant]
    k = kyberParams["k"]
    eta1=kyberParams["eta1"]
    rhoAndSigma = G(d+(k.to_bytes(1,'big'))).digest()
    # print((d+k.to_bytes(1,'big')).hex())
    rho = rhoAndSigma[:32]
    sigma = rhoAndSigma[32:]
    # print(rho.hex())
    # print(sigma.hex())
    n=0
    A_hat=[[None for _ in range(k)] for _ in range(k)]
    # print(A_hat)
    for i in range(k):
        for j in range(k):
            A_hat[i][j] = sampleNTT(rho+j.to_bytes(1,'big')+i.to_bytes(1,'big'))
    # print(A_hat)
    s=[None for _ in range(k)]
    e=[None for _ in range(k)]
    # print(s)
    for i in range(k):
        s[i] = samplePolyCBD(eta1,PRF(sigma,n.to_bytes(1,'big'),eta1))
        # print(s)
        n+=1
    for i in range(k):
        e[i] = samplePolyCBD(eta1,PRF(sigma,n.to_bytes(1,'big'),eta1))
        n+=1
    nttContext = NTTContext(N,Q)
    s_hat = []
    # print(len(s[0]))
    for i in range(k):
        # s_hat.append(nttContext.ftt_fwd(s[i]))
        s_hat.append(ntt(s[i]))
    # print(s_hat)
    e_hat = []
    for i in range(k):
        # e_hat.append(nttContext.ftt_fwd(e[i]))
        e_hat.append(ntt(e[i]))
    
    t_hat = matrix_vector_multiply_ntt(A_hat, s_hat)
    t_hat = polyvec_add_ntt(t_hat, e_hat)

    # print("t_hat is val")
    # print(t_hat)
    # print(len(t_hat))
    ek_bytes = bytearray()
    for i in range(k):
        ek_bytes.extend(ByteEncode_d(t_hat[i],12))
    
    ek_bytes.extend(rho)    
    ekPKE = bytes(ek_bytes)
    
    dk_bytes = bytearray()
    for i in range(k):
        dk_bytes.extend(ByteEncode_d(s_hat[i],12))
    dkPKE = bytes(dk_bytes)

    # print(ekPKE.hex())
    # print(dkPKE.hex())
    return (ekPKE, dkPKE)


# kPkeKeyGen(os.urandom(32),"ML-KEM-512")
