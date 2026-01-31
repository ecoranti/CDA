"""Lab 2 – Transmisor digital con modulación y pulso de Nyquist (RRC).

Genera una secuencia binaria aleatoria, mapea a símbolos complejos IQ según la
modulación seleccionada, conforma el pulso usando un filtro Root Raised Cosine y
exporta figuras y archivo IQ compatible con SDR (float32 intercalado I/Q).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _ensure_dirs(out_dir: str) -> Path:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p / "figures").mkdir(parents=True, exist_ok=True)
    return p


def bits_random(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n, dtype=np.int8)


def map_bpsk(bits: np.ndarray) -> np.ndarray:
    return 2 * bits.astype(np.float32) - 1.0  # {0,1} -> {-1,+1}


def map_qpsk(bits: np.ndarray) -> np.ndarray:
    # Gray mapping: 00:-1-1j, 01:-1+1j, 11:+1+1j, 10:+1-1j (normalized)
    if len(bits) % 2 != 0:
        bits = np.hstack([bits, 0])
    b0 = bits[0::2]
    b1 = bits[1::2]
    I = 2 * b0.astype(np.float32) - 1.0
    Q = 2 * b1.astype(np.float32) - 1.0
    # Map to Gray
    # 00 -> (-1,-1), 01 -> (-1, +1), 11 -> (+1, +1), 10 -> (+1, -1)
    # As constructed above already follows Gray if we consider (b0,b1)
    s = (I + 1j * Q) / np.sqrt(2.0)
    return s.astype(np.complex64)


def map_16qam(bits: np.ndarray) -> np.ndarray:
    # 16-QAM Gray on I and Q: 2 bits per axis -> levels {-3,-1,+1,+3}/sqrt(10)
    if len(bits) % 4 != 0:
        bits = np.hstack([bits, np.zeros(4 - (len(bits) % 4), dtype=np.int8)])
    b = bits.reshape(-1, 4)
    def axis(vh: np.ndarray) -> np.ndarray:
        # vh: 2-bit vector per symbol, Gray mapping 00->-3, 01->-1, 11->+1, 10->+3
        g = vh[:, 0] * 2 + vh[:, 1]
        # map: 0->-3, 1->-1, 3->+1, 2->+3
        levels = np.array([-3, -1, +3, +1], dtype=np.int8)  # index 0,1,2,3 -> level
        return levels[g]
    I = axis(b[:, 0:2]).astype(np.float32)
    Q = axis(b[:, 2:4]).astype(np.float32)
    s = (I + 1j * Q) / np.sqrt(10.0)
    return s.astype(np.complex64)


def rrc_taps(sps: int, alpha: float, span_symbols: int) -> np.ndarray:
    """Root Raised Cosine pulse (unit symbol period Ts=1)."""
    N = span_symbols * sps
    t = np.arange(-N/2, N/2 + 1) / sps  # time in symbol periods
    h = np.zeros_like(t, dtype=np.float64)
    for i, tt in enumerate(t):
        if abs(tt) < 1e-12:
            h[i] = 1.0 + alpha * (4/np.pi - 1)
        elif abs(abs(4*alpha*tt) - 1.0) < 1e-12:
            # special case at t = ±1/(4α)
            h[i] = (alpha/np.sqrt(2)) * (
                (1 + 2/np.pi) * np.sin(np.pi/(4*alpha)) + (1 - 2/np.pi) * np.cos(np.pi/(4*alpha))
            )
        else:
            num = np.sin(np.pi*tt*(1 - alpha)) + 4*alpha*tt * np.cos(np.pi*tt*(1 + alpha))
            den = np.pi*tt * (1 - (4*alpha*tt)**2)
            h[i] = num / den
    # Normalize energy so that sum of h^2 = 1 (approximate unity energy)
    h = h / np.sqrt(np.sum(h*h) + 1e-12)
    return h.astype(np.float64)


def upsample(symbols: np.ndarray, sps: int) -> np.ndarray:
    x = np.zeros(len(symbols) * sps, dtype=np.complex128)
    x[::sps] = symbols.astype(np.complex128)
    return x


def shape_signal(symbols: np.ndarray, sps: int, alpha: float, span_symbols: int) -> Tuple[np.ndarray, np.ndarray]:
    h = rrc_taps(sps, alpha, span_symbols)
    x_up = upsample(symbols, sps)
    y = np.convolve(x_up, h, mode="same")
    return y.astype(np.complex64), h.astype(np.float32)


def constellation_at_samples(y: np.ndarray, sps: int, h_len: int) -> np.ndarray:
    # Choose a sampling phase: center of filter
    gd = h_len // 2
    start = gd % sps
    samples = y[start::sps]
    return samples


def save_figures(out_dir: Path, y: np.ndarray, h: np.ndarray, sps: int, mod_name: str):
    figdir = out_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    # 1) Señal en el tiempo (I y Q)
    t = np.arange(len(y)) / sps
    plt.figure()
    plt.plot(t, y.real, label='I')
    plt.plot(t, y.imag, label='Q')
    plt.xlabel('Tiempo [símbolos]')
    plt.ylabel('Amplitud')
    plt.title(f'Señal IQ (mod={mod_name}, sps={sps})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir / "l2_time_iq.png", dpi=140)
    plt.close()

    # 2) Espectro (módulo de FFT)
    N = 4096
    Y = np.fft.fftshift(np.fft.fft(y, n=N))
    f = np.linspace(-0.5, 0.5, N, endpoint=False)
    P = 20*np.log10(np.abs(Y)/np.max(np.abs(Y)) + 1e-12)
    plt.figure()
    plt.plot(f, P)
    plt.xlabel('Frecuencia normalizada [ciclos/símbolo]')
    plt.ylabel('Magnitud [dB]')
    plt.title('Espectro de la señal IQ')
    plt.tight_layout()
    plt.savefig(figdir / "l2_spectrum.png", dpi=140)
    plt.close()

    # 3) Constelación (muestras en instantes de símbolo)
    const = constellation_at_samples(y, sps, len(h))
    plt.figure()
    plt.scatter(const.real, const.imag, s=8, alpha=0.6)
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.title('Constelación (muestras a Ts)')
    plt.gca().set_aspect('equal', 'box')
    plt.tight_layout()
    plt.savefig(figdir / "l2_constellation.png", dpi=140)
    plt.close()

    # 4) Pulso RRC (respuesta temporal)
    t_h = np.arange(len(h)) / sps - (len(h)//2)/sps
    plt.figure()
    plt.plot(t_h, h)
    plt.xlabel('Tiempo [símbolos]')
    plt.ylabel('h_RRC')
    plt.title('Pulso formador RRC')
    plt.tight_layout()
    plt.savefig(figdir / "l2_rrc_pulse.png", dpi=140)
    plt.close()


def save_sdr_bin(out_dir: Path, y: np.ndarray, base_name: str):
    # float32 interleaved I/Q
    iq = np.empty(2 * len(y), dtype=np.float32)
    iq[0::2] = y.real.astype(np.float32)
    iq[1::2] = y.imag.astype(np.float32)
    (out_dir / f"{base_name}.bin").write_bytes(iq.tobytes())


@dataclass
class RunResult:
    out_dir: str
    num_symbols: int
    sps: int
    alpha: float
    modulation: str


def run(
    modulation: str = "QPSK",
    bits_len: int = 2000,
    sps: int = 8,
    alpha: float = 0.25,
    span_symbols: int = 8,
    out_dir: str = "outputs_ui/lab2",
    seed: int = 0,
) -> RunResult:
    outp = _ensure_dirs(out_dir)
    bits = bits_random(bits_len, seed=seed)

    mod = modulation.upper()
    if mod == "BPSK":
        symbols = map_bpsk(bits)
        symbols = symbols.astype(np.complex64)  # real -> complex
    elif mod == "QPSK":
        symbols = map_qpsk(bits)
    elif mod in ("16QAM", "16-QAM", "QAM16"):
        symbols = map_16qam(bits)
        mod = "16QAM"
    else:
        raise ValueError(f"Modulación no soportada: {modulation}")

    y, h = shape_signal(symbols, sps=sps, alpha=alpha, span_symbols=span_symbols)

    save_figures(outp, y, h, sps=sps, mod_name=mod)
    save_sdr_bin(outp, y, base_name=f"lab2_{mod}_sps{sps}_a{alpha}")

    return RunResult(out_dir=str(outp), num_symbols=len(symbols), sps=sps, alpha=alpha, modulation=mod)

