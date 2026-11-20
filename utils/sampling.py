from .params import Q , N
from .hash import XOF
import os

def sampleNTT(B):
    xof = XOF(B)
    coeffs = []
    while len(coeffs)<N:
        C = xof.read(3)
        d1 = C[0] | ((C[1] & 0x0F) << 8) 
        d2 = (C[1] >> 4) | (C[2] << 4)
        val1 = d1 & 0xFFF
        val2 = d2 & 0xFFF

        if val1 < Q:
            coeffs.append(val1)
        
        if len(coeffs) < N and val2 < Q:
            coeffs.append(val2)
    
    return coeffs

def samplePolyCBD(eta, B):    
    coeffs = []
    # print("values are")
    # print(N)
    for i in range(N):
        a = 0 
        b = 0 
        
        for j in range(eta):
            byte_idx = (2*i*eta + j) // 8
            bit_idx = (2*i*eta + j) % 8
            bit = (B[byte_idx] >> bit_idx) & 1
            a += bit
            
            byte_idx = ((2*i+1)*eta + j) // 8
            bit_idx = ((2*i+1)*eta + j) % 8
            bit = (B[byte_idx] >> bit_idx) & 1
            b += bit
        
        coeffs.append(a - b)
    
    return coeffs


# print(sampleNTT(os.urandom(32)))
# print(samplePolyCBD(3,os.urandom(684)))