import numpy as np
import math
from typing import List, Tuple

def ints_to_bits(arr, bits_per_symbol: int) -> List[int]:
    out = []
    for a in arr:
        for b in range(bits_per_symbol-1, -1, -1):
            out.append((a >> b) & 1)
    return out

def bits_to_bytes(bits: List[int]) -> List[int]:
    bits = list(bits)
    if len(bits) % 8 != 0:
        bits += [0] * (8 - (len(bits) % 8))
    out = []
    for i in range(0, len(bits), 8):
        byte = 0
        for b in bits[i:i+8]:
            byte = (byte << 1) | int(b)
        out.append(byte)
    return out

def bits_entropy_stats(bits: List[int]) -> Tuple[float, float, float, float]:
    """
    Calcula métricas del flujo de bits:
    - p0: probabilidad de 0
    - p1: probabilidad de 1
    - H: entropía (bits/bit)
    - var: varianza sobre {0,1}
    """
    arr = np.array(bits, dtype=np.uint8)
    p1 = arr.mean()
    p0 = 1 - p1

    def hb(p):
        if p <= 0 or p >= 1:
            return 0.0
        return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

    H = hb(p1)
    var = arr.var()
    return p0, p1, H, var
