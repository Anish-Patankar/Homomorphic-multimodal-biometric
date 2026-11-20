from utils.params import Q, N, PARAMS
from utils.hash import PRF
from utils.sampling import sampleNTT, samplePolyCBD
from utils.matrixops import matrix_vector_multiply_ntt, polyvec_add_ntt, transpose, poly_add, transpose_vec, matrix_vector_multiply, poly_dot_product_ntt
from utils.compression import ByteEncode_d, ByteDecode_d, DecompressPoly, CompressPoly
from ntt import NTTContext
from fips_ntt import ntt, ntt_inv

def kPkeEncrypt(ekPKE, m, r, kyberVariant = "ML-KEM-512"):
    kyberParams = PARAMS[kyberVariant]
    k=kyberParams["k"]
    eta1=kyberParams["eta1"]
    eta2=kyberParams["eta2"]
    n=0
    t_hat = []
    for i in range(k):
        t_hat.append(ByteDecode_d(ekPKE[i*384:(i+1)*384],12))
    # print(type(ekPKE))
    # print(ekPKE)
    # print(len(ekPKE))
    rho = ekPKE[(i*k)*384:]
    # print("size of rho is")
    # print(len(rho))
    A_hat=[[None for _ in range(k)] for _ in range(k)]
    for i in range(k):
        for j in range(k):
            A_hat[i][j] = sampleNTT(rho+j.to_bytes(1,'big')+i.to_bytes(1,'big'))
    
    y=[None for _ in range(k)]
    e1=[None for _ in range(k)]
    # print(s)
    for i in range(k):
        y[i] = samplePolyCBD(eta1,PRF(r,n.to_bytes(1,'big'),eta1))
        n+=1
    for i in range(k):
        e1[i] = samplePolyCBD(eta2,PRF(r,n.to_bytes(1,'big'),eta2))
        n+=1
    e2 = samplePolyCBD(eta2,PRF(r,n.to_bytes(1,'big'),eta2))
    nttContext = NTTContext(N,Q)
    y_hat = []
    for i in range(k):
        # y_hat.append(nttContext.ftt_fwd(y[i]))
        y_hat.append(ntt(y[i]))

    A_hatxy_hat = matrix_vector_multiply_ntt(transpose(A_hat),y_hat)
    Axy = []
    for i in range(k):
        # Axy.append(nttContext.ftt_inv(A_hatxy_hat[i]))
        Axy.append(ntt_inv(A_hatxy_hat[i]))
    # print(Axy)
    u= polyvec_add_ntt(Axy,e1)
    # for i in range(k):
        # u.append(Axy[i]e1[i])
    
    mu = DecompressPoly(ByteDecode_d(m,1),1)

    # t_hatxy_hat = matrix_vector_multiply(transpose_vec(t_hat),y_hat)
    t_dot_y = poly_dot_product_ntt(t_hat, y_hat, k)
    v = poly_add(poly_add(ntt_inv(t_dot_y), e2), mu)
    # v= poly_add(poly_add(nttContext.ftt_inv(t_hatxy_hat[0]), e2), mu)
    # v= poly_add(poly_add(ntt_inv(t_hatxy_hat[0]), e2), mu)
    c1 = bytearray()
    c2 = bytearray()
    for i in range(k):
        c1.extend(ByteEncode_d(CompressPoly(u[i],kyberParams["du"]),kyberParams["du"]))
    c2.extend(ByteEncode_d(CompressPoly(v,kyberParams["dv"]),kyberParams["dv"]))
    # print("Cipher text is")
    # print(c1)
    # print(c2)
    c = bytes(c1 + c2)
    # print(len(c))
    return c
