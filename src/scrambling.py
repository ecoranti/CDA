from typing import List

class LFSR:
    def __init__(self, seed: int = 0b1010110011, taps=(9, 6), bitwidth: int = 10):
        assert 1 <= bitwidth <= 32
        self.state = seed & ((1 << bitwidth) - 1)
        self.taps = taps
        self.bitwidth = bitwidth

    def step(self) -> int:
        fb = 0
        for t in self.taps:
            fb ^= (self.state >> t) & 1
        out = self.state & 1
        self.state = (self.state >> 1) | (fb << (self.bitwidth - 1))
        return out

    def prbs(self, n: int) -> List[int]:
        return [self.step() for _ in range(n)]

def scramble(bits: List[int], seed: int = 0b1010110011, taps=(9, 6), bitwidth: int = 10) -> List[int]:
    l = LFSR(seed=seed, taps=taps, bitwidth=bitwidth)
    mask = l.prbs(len(bits))
    return [int(b) ^ int(m) for b, m in zip(bits, mask)]
