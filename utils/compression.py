from .params import Q, N

def compress(x,d):
    x = x % Q
    y = ((x << d) + Q // 2) // Q
    y = y & ((1 << d) - 1)
    return y

def decompress(y,d):
    x = ((y * Q) + (1 << (d - 1))) >> d
    return x

def CompressPoly(poly, d):
    return [compress(c, d) for c in poly]


def DecompressPoly(poly, d):
    return [decompress(c, d) for c in poly]


def ByteEncode_d(poly, d):
    output_size = (N * d) // 8
    result = bytearray(output_size)
    
    bit_pos = 0
    for coeff in poly:
        val = coeff & ((1 << d) - 1)
        
        byte_pos = bit_pos // 8
        bit_offset = bit_pos % 8
       
       
        remaining_bits = d
        while remaining_bits > 0:
            bits_in_current_byte = min(8 - bit_offset, remaining_bits)
            mask = (1 << bits_in_current_byte) - 1
            
            result[byte_pos] |= ((val & mask) << bit_offset)
            
            val >>= bits_in_current_byte
            remaining_bits -= bits_in_current_byte
            byte_pos += 1
            bit_offset = 0
        
        bit_pos += d
    
    return bytes(result)


def ByteDecode_d(data, d):
    # expected_size = (N * d) // 8    
    poly = []
    bit_pos = 0
    
    for _ in range(N):
        val = 0
        byte_pos = bit_pos // 8
        bit_offset = bit_pos % 8
        
        remaining_bits = d
        shift = 0
        
        while remaining_bits > 0:
            bits_in_current_byte = min(8 - bit_offset, remaining_bits)
            mask = (1 << bits_in_current_byte) - 1
            
            bits = (data[byte_pos] >> bit_offset) & mask
            val |= (bits << shift)
            
            shift += bits_in_current_byte
            remaining_bits -= bits_in_current_byte
            byte_pos += 1
            bit_offset = 0
        
        if d == 12:
            poly.append(val % Q)
        else:
            poly.append(val % (1 << d))
        
        bit_pos += d
    
    return poly




