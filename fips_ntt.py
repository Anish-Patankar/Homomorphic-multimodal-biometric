import random

from utils.params import ZETAS

Q    = 3329
ZETA = 17  # primitive 512th root of unity mod Q

def _bit_rev_7(x: int) -> int:
    """Reverse the 7 least-significant bits of x."""
    result = 0
    for _ in range(7):
        result = (result << 1) | (x & 1)
        x >>= 1
    return result

# Precompute twiddle factors: ζ^brv7(i) mod Q for i = 0..127
ZETAS = [pow(ZETA, _bit_rev_7(i), Q) for i in range(128)]

def ntt(f: list) -> list:
    """FIPS 203 Algorithm 9 — forward NTT."""
    f = list(f)
    k = 1
    length = 128
    while length >= 2:
        start = 0
        while start < 256:
            zeta = ZETAS[k]
            k += 1
            for j in range(start, start + length):
                t = (zeta * f[j + length]) % Q
                f[j + length] = (f[j] - t) % Q
                f[j] = (f[j] + t) % Q
            start += 2 * length
        length //= 2
    return f

def ntt_inv(f: list) -> list:
    """FIPS 203 Algorithm 10 — inverse NTT."""
    f = list(f)
    k = 127
    length = 2
    while length <= 128:
        start = 0
        while start < 256:
            zeta = ZETAS[k]
            k -= 1
            for j in range(start, start + length):
                t             = f[j]
                f[j]          = (t + f[j + length]) % Q
                f[j + length] = (zeta * (f[j + length] - t)) % Q
            start += 2 * length
        length *= 2
    inv128 = pow(128, Q - 2, Q)   # 128^{-1} mod q = 3303
    return [(c * inv128) % Q for c in f]

# random.seed(42)
# f = [random.randint(0, Q-1) for _ in range(256)]
# assert ntt_inv(ntt(f)) == f, "Round-trip FAILED"
# print("Round-trip: PASS")

# # Check ZETAS[1] = 17 (ζ^brv7(1) = ζ^64 = 17^64 mod 3329)
# assert ZETAS[1] == pow(17, 64, 3329)
# print("ZETAS[1]:", ZETAS[1])   # should be 1729

# # Check inv128
# assert (128 * pow(128, Q-2, Q)) % Q == 1
# print("inv128:", pow(128, Q-2, Q))   # should be 3303