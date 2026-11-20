from .params import N, INV_N, Q, QINV, ZETA

def _bit_rev_7(x: int) -> int:
    result = 0
    for _ in range(7):
        result = (result << 1) | (x & 1)
        x >>= 1
    return result

ZETAS = [pow(ZETA, _bit_rev_7(i), Q) for i in range(128)]


def transpose(mat):
    rows = len(mat)
    if rows == 0:
        return mat
    cols = len(mat[0])
    matTranspose = []
    for j in range(cols):
        colTranspose = []
        for i in range(rows):
            colTranspose.append(mat[i][j])
        matTranspose.append(colTranspose)
    
    return matTranspose


def transpose_vec(vec):
    rows = len(vec)
    if rows == 0:
        return vec
    vecTranspose =[]
    vecTranspose.append(vec)
    return vecTranspose

def poly_add(a, b):
    return [(a[i] + b[i]) % Q for i in range(N)]

def poly_sub(a, b):
    return [((a[i] - b[i]) % Q) for i in range(N)]

def montgomery_reduce(a: int) -> int:
    u = (a * QINV) & 0xFFFF
    t = (a + u * Q) >> 16

    if t >= Q:
        t -= Q

    return t

# def basemul(a, b, zeta):
#     r = [0, 0]
    
#     r[0] = montgomery_reduce(a[1] * b[1]) % Q
#     r[0] = montgomery_reduce(r[0] * zeta) % Q
#     r[0] += montgomery_reduce(a[0] * b[0]) % Q
#     r[0] = r[0] % Q
    
#     r[1] = montgomery_reduce(a[0] * b[1]) % Q
#     r[1] += montgomery_reduce(a[1] * b[0]) % Q
#     r[1] = r[1] % Q

#     return r

def basemul(a, b, zeta):
    r = [0, 0]
    r[0] = (a[0]*b[0] + a[1]*b[1]*zeta) % Q
    r[1] = (a[0]*b[1] + a[1]*b[0]) % Q
    return r


def basemul_poly(a, b) :
    r = [0] * N
    
    for i in range(N // 4):
    
        result1 = basemul(
            [a[4*i], a[4*i + 1]],
            [b[4*i], b[4*i + 1]],
            ZETAS[64 + i]
        )
        r[4*i] = result1[0]
        r[4*i + 1] = result1[1]
        
    
        result2 = basemul(
            [a[4*i + 2], a[4*i + 3]],
            [b[4*i + 2], b[4*i + 3]],
            -ZETAS[64 + i]  
        )
        r[4*i + 2] = result2[0]
        r[4*i + 3] = result2[1]
    
    return r



def matrix_vector_multiply(A, B):
    
    k = len(B)
    t_hat = []
    
    
    result = [0] * N
        
    for j in range(k):
        product = basemul_poly(A[0][j], B[j])
         
        for coeff_idx in range(N):
            result[coeff_idx] = (result[coeff_idx] + product[coeff_idx]) % Q
        
    t_hat.append(result)
    
    return t_hat

def matrix_vector_multiply_ntt(A_hat, s_hat):
    k = len(s_hat)
    t_hat = []
    
    for i in range(k):
        result = [0] * N
        
        for j in range(k):
            product = basemul_poly(A_hat[i][j], s_hat[j])
            
            for coeff_idx in range(N):
                result[coeff_idx] = ( result[coeff_idx] + product[coeff_idx] ) % Q
        
        t_hat.append(result)
    
    return t_hat




def polyvec_add_ntt(a, b):

    k = len(a)
    result = []
    
    for i in range(k):
        poly_sum = []
        for j in range(N):
            poly_sum.append((a[i][j] + b[i][j]) % Q)
        result.append(poly_sum)
    
    return result


def poly_dot_product_ntt(s_hat, u_hat, k):
    result = [0] * N
    for i in range(k):
        product = basemul_poly(s_hat[i], u_hat[i])
        result = [(result[j] + product[j]) % Q for j in range(N)]
    return result


# def poly_add(a, b):
#     return [a[i] + b[i] for i in range(N)]