from keygen_internal import keygen_int
import os


def keygen(kyberVariant = "ML-KEM-512"):
    d = os.urandom(32)
    z = os.urandom(32)
    # z = bytes.fromhex("1F8CB39E9E30BC458A0DC5408884B1187FB217018DF760FA57317703B844A0A9")
    # d = bytes.fromhex("47B893474672BA92E4B12EE44FB32953AF8E8503B5FB471D1614FB8A021A660A")
    # z = bytes.fromhex("B9958575128F97022525266064F9422E83F75FBDCD0F2BC2648655F0E31A17E5")
    # d = bytes.fromhex("3DEF73C558E1F5F2BFB4F6EAA36EE200CDF54580958CC2176282628FF508A016")
    # z = bytes.fromhex("c6f5785a6f2b42e843228be53eb768d64c6f9d4355ae95f083e51ed57c437310")
    # d = bytes.fromhex("e1e3206875e67d7e81353774fe9025035b9b41a4a9f6ec00b91c600442fd717d")
    if d == None or z == None:
        return None
    
    keys = keygen_int(d,z, kyberVariant)
    return keys


# print(keygen())
# keygen()