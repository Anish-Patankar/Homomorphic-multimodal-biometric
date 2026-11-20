from utils.params import Q, N, ZETA, PARAMS
from utils.hash import G, H, J, PRF, XOF
from utils.sampling import sampleNTT, samplePolyCBD
from utils.matrixops import matrix_vector_multiply_ntt, polyvec_add_ntt, transpose, poly_add, transpose_vec, matrix_vector_multiply, poly_sub, poly_dot_product_ntt
from utils.compression import ByteEncode_d, ByteDecode_d, DecompressPoly, compress, CompressPoly
import os
from ntt import NTTContext
from fips_ntt import ntt, ntt_inv


def kPkeDecrypt(dkPKE, c, kyberVariant = "ML-KEM-512"):
    kyberParams = PARAMS[kyberVariant]
    k=kyberParams["k"]
    eta1=kyberParams["eta1"]
    eta2=kyberParams["eta2"]
    du = kyberParams["du"]
    dv = kyberParams["dv"]
    c1 = c[:32*du*k]
    c2 = c[32*du*k:]
    u = []
    for i in range(k):
        u.append(DecompressPoly(ByteDecode_d(c1[i*32*du:(i+1)*32*du],du),du))
    v = DecompressPoly(ByteDecode_d(c2,dv),dv)
    s_hat = []
    for i in range(k):
        s_hat.append(ByteDecode_d(dkPKE[384*i : 384*(i+1)],12))
    nttContext = NTTContext(N,Q)

    u_hat = []
    for i in range(k):
        # u_hat.append(nttContext.ftt_fwd(u[i]))
        u_hat.append(ntt(u[i]))

    # s_hatxu_hat = matrix_vector_multiply(transpose_vec(s_hat),u_hat)
    # sxu = nttContext.ftt_inv(s_hatxu_hat[0])
    s_hatxu_hat = poly_dot_product_ntt(s_hat, u_hat, k)
    # sxu = ntt_inv(s_hatxu_hat[0])
    sxu = ntt_inv(s_hatxu_hat)

    w = poly_sub(v,sxu)
    m = ByteEncode_d(CompressPoly(w,1),1)
    return m



