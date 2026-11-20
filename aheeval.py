from utils.compression import ByteDecode_d, ByteEncode_d, CompressPoly, DecompressPoly
from utils.matrixops import poly_add
from utils.params import PARAMS

def aheeval(c1,c2, kyberVariant = "ML-KEM-512"):
    kyberParams = PARAMS[kyberVariant]
    k=kyberParams["k"]
    eta1=kyberParams["eta1"]
    eta2=kyberParams["eta2"]
    du = kyberParams["du"]
    dv = kyberParams["dv"]
    
    c11 = c1[:32*du*k]
    c12 = c1[32*du*k:]
    c21 = c2[:32*du*k]
    c22 = c2[32*du*k:]
    
    u1 = []
    for i in range(k):
        u1.append(DecompressPoly(ByteDecode_d(c11[i*32*du:(i+1)*32*du],du),du))
    v1 = DecompressPoly(ByteDecode_d(c12,dv),dv)

    u2 = []
    for i in range(k):
        u2.append(DecompressPoly(ByteDecode_d(c21[i*32*du:(i+1)*32*du],du),du))
    v2 = DecompressPoly(ByteDecode_d(c22,dv),dv)

    uadd = []
    for i in range(k):
        uadd.append(poly_add(u1[i],u2[i]))
    
    vadd = poly_add(v1,v2)

    cadd1 = bytearray()
    cadd2 = bytearray()
    for i in range(k):
        cadd1.extend(ByteEncode_d(CompressPoly(uadd[i],kyberParams["du"]),kyberParams["du"]))
    cadd2.extend(ByteEncode_d(CompressPoly(vadd,kyberParams["dv"]),kyberParams["dv"]))

    c = bytes(cadd1 + cadd2)
    return c