from pkpekeygen import kPkeKeyGen
from pkpeencrypt import kPkeEncrypt
from pkpedecrypt import kPkeDecrypt
from aheeval import aheeval
import os

keys = kPkeKeyGen(os.urandom(32))
# print(keys)
pk = keys[0]
sk = keys[1]
for _ in range(80):
    m1 = os.urandom(32)
    m2 = os.urandom(32)
    # m1 = bytes([0]*32)
    # m2 = bytes([0]*32)
    r1 = os.urandom(32)
    r2 = os.urandom(32)
    c1 = kPkeEncrypt(pk,m1,r1)
    c2 = kPkeEncrypt(pk,m2,r2)
    cxor = aheeval(c1,c2)
    mxor = kPkeDecrypt(sk,cxor)
    expected = bytes(a ^ b for a,b in zip(m1,m2))
    print("asserting")
    assert expected == mxor
print(expected.hex())
print(mxor.hex())
print(len(expected.hex()))
print(len(mxor.hex()))