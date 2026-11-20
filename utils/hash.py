import hashlib

def G(val):
    return hashlib.sha3_512(val)

def H(val):
    return hashlib.sha3_256(val)

def J(val:bytes):
    shake256 = hashlib.shake_256()
    shake256.update(val)
    return shake256

def PRF(s:bytes,b:bytes,eta):
   shake256 =  J(s+b)
   return shake256.digest(64*eta)

class XOF:
    # inputData = rho+bytes([i,j])
    def __init__(self, data):
        self.shake = hashlib.shake_128(data)
        self.pos = 0

    def read(self, n):
        out = self.shake.digest(self.pos + n)
        chunk = out[self.pos:self.pos + n]
        self.pos += n
        return chunk

