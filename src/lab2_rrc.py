from __future__ import annotations
import os
import json
from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _save_json(obj: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_iq_float32(iq: np.ndarray, path: str) -> None:
    iq = np.asarray(iq, dtype=np.complex64)
    inter = np.empty(iq.size * 2, dtype=np.float32)
    inter[0::2] = iq.real.astype(np.float32)
    inter[1::2] = iq.imag.astype(np.float32)
    inter.tofile(path)


def rrc_taps(alpha: float, sps: int, span: int) -> np.ndarray:
    if not (0 <= alpha <= 1):
        raise ValueError("alpha debe estar en [0, 1].")
    n = np.arange(-span * sps, span * sps + 1, dtype=np.float64)
    t = n / float(sps)
    taps = np.zeros_like(t, dtype=np.float64)
    pi_t = np.pi * t
    four_alpha_t = 4 * alpha * t
    eps = 1e-12
    num = np.sin(np.pi * t * (1 - alpha)) + four_alpha_t * np.cos(np.pi * t * (1 + alpha))
    den = pi_t * (1 - (four_alpha_t ** 2))
    mask = (np.abs(den) > eps)
    taps[mask] = num[mask] / den[mask]
    m0 = (np.abs(t) <= eps)
    if np.any(m0):
        taps[m0] = 1 - alpha + 4 * alpha / np.pi
    if alpha > 0:
        tm = 1.0 / (4.0 * alpha)
        m1 = np.isclose(np.abs(t), tm, atol=1e-12)
        if np.any(m1):
            val = (alpha / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) +
                (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
            )
            taps[m1] = val
    taps = taps / np.sqrt(np.sum(taps ** 2) + 1e-12)
    return taps.astype(np.float64)


def map_bits_to_symbols(bits: np.ndarray, mod: str = "BPSK") -> np.ndarray:
    bits = np.asarray(bits).astype(np.uint8).ravel()
    mod = mod.upper()
    if mod == "BPSK":
        s = 2 * bits.astype(np.int8) - 1
        return s.astype(np.float64) + 0j
    elif mod == "QPSK":
        if bits.size % 2 != 0:
            bits = np.pad(bits, (0, 1), constant_values=0)
        b0 = bits[0::2]
        b1 = bits[1::2]
        I = np.where(b0 == 0, 1.0, -1.0)
        Q = np.where(b1 == 0, 1.0, -1.0)
        c = (I + 1j * Q) / np.sqrt(2.0)
        return c.astype(np.complex128)
    else:
        raise ValueError("Modulación no soportada. Use 'BPSK' o 'QPSK'.")


def upsample_and_filter(symbols: np.ndarray, sps: int, taps: np.ndarray) -> np.ndarray:
    L = symbols.size
    x_up = np.zeros(L * sps, dtype=np.complex128)
    x_up[::sps] = symbols
    y = np.convolve(x_up, taps.astype(np.complex128), mode="full")
    delay = (taps.size - 1) // 2
    y = y[delay: delay + L * sps]
    p = np.mean(np.abs(y) ** 2) + 1e-12
    y = y / np.sqrt(p)
    return y.astype(np.complex64)


def plot_rrc(taps: np.ndarray, sps: int, out_path: str) -> None:
    t = np.arange(-(taps.size // 2), taps.size // 2 + 1) / float(sps)
    if t.size > taps.size:
        t = t[:-1]
    plt.figure()
    plt.plot(t, taps)
    plt.axhline(0, color='gray', lw=0.6)
    plt.axvline(0, color='gray', lw=0.6)
    plt.grid(True, alpha=0.25)
    plt.xlabel("Tiempo [símbolos]")
    plt.ylabel("h_RRC[n]")
    plt.title("Pulso raíz de coseno alzado (RRC)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_time_iq(iq: np.ndarray, sps: int, out_path: str, nsamples: int = 6000) -> None:
    Ns = min(nsamples, iq.size)
    t = np.arange(Ns) / float(sps)
    plt.figure()
    plt.plot(t, iq.real[:Ns], label="I")
    plt.plot(t, iq.imag[:Ns], label="Q", alpha=0.85)
    plt.axhline(0, color='gray', lw=0.6)
    plt.grid(True, alpha=0.25)
    plt.xlabel("Tiempo [símbolos]")
    plt.ylabel("Amplitud")
    plt.title("Señal en el tiempo (I/Q)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_constellation(iq: np.ndarray, sps: int, out_path: str, npoints: int = 2000) -> None:
    sym = iq[::sps]
    if sym.size > npoints:
        rng = np.random.default_rng(0)
        idx = rng.choice(sym.size, size=npoints, replace=False)
        sym = sym[idx]
    plt.figure()
    plt.scatter(sym.real, sym.imag, s=8)
    # Ejes cartesianos en 0 para referencia
    plt.axhline(0, color='gray', lw=0.6)
    plt.axvline(0, color='gray', lw=0.6)
    # Límites simétricos con pequeño margen
    mx = float(np.max(np.abs(np.concatenate([sym.real, sym.imag])))) if sym.size else 1.0
    lim = mx * 1.2 if mx > 0 else 1.0
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.title("Constelación")
    plt.axis("equal")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_spectrum(iq: np.ndarray, sps: int, out_path: str) -> None:
    fs = float(sps)
    f, Pxx = welch(iq, fs=fs, nperseg=2048, return_onesided=False, scaling="density")
    Pxx = np.fft.fftshift(Pxx)
    f = np.fft.fftshift(f)
    PdB = 10 * np.log10(np.maximum(Pxx, 1e-18))
    plt.figure()
    plt.plot(f, PdB)
    plt.axvline(0, color='gray', lw=0.6)
    plt.grid(True, which='both', alpha=0.25)
    plt.xlabel("Frecuencia [ciclos/símbolo]")
    plt.ylabel("Densidad espectral [dB/Hz]")
    plt.title("Espectro (Welch)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_eye(iq: np.ndarray, sps: int, out_path: str, span_symbols: int = 2, max_traces: int = 300) -> None:
    """Dibuja diagrama de ojo para I y Q, superponiendo ventanas de span_symbols símbolos.
    """
    seg = span_symbols * sps
    if seg <= 1:
        seg = 2 * sps
    # Construir segmentos desplazados cada símbolo
    traces_I = []
    traces_Q = []
    for start in range(0, len(iq) - seg, sps):
        s = iq[start:start+seg]
        traces_I.append(s.real)
        traces_Q.append(s.imag)
    if not traces_I:
        return
    # Submuestrear si hay demasiadas
    step = max(1, len(traces_I) // max_traces)
    traces_I = traces_I[::step]
    traces_Q = traces_Q[::step]
    t = np.arange(seg) / float(sps)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6.4, 3.8))
    plt.subplot(1, 2, 1)
    for tr in traces_I:
        plt.plot(t, tr, color='#2563eb', alpha=0.25, linewidth=0.8)
    for k in range(0, span_symbols+1):
        plt.axvline(k, color='gray', lw=0.5, alpha=0.5)
    plt.axhline(0, color='gray', lw=0.6)
    plt.grid(True, alpha=0.25)
    plt.title('Eye (I)')
    plt.xlabel('Tiempo [símbolos]')
    plt.ylabel('Amplitud')
    plt.subplot(1, 2, 2)
    for tr in traces_Q:
        plt.plot(t, tr, color='#0ea5e9', alpha=0.25, linewidth=0.8)
    for k in range(0, span_symbols+1):
        plt.axvline(k, color='gray', lw=0.5, alpha=0.5)
    plt.axhline(0, color='gray', lw=0.6)
    plt.grid(True, alpha=0.25)
    plt.title('Eye (Q)')
    plt.xlabel('Tiempo [símbolos]')
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_bits_and_iq(bits: np.ndarray, y: np.ndarray, sps: int, modulation: str, out_path: str, max_symbols: int = 120) -> None:
    """Grafica, alineados en tiempo simbólico, un segmento de bits (arriba) y la señal IQ (abajo).
    Para BPSK se usan 1 bit/símbolo; para QPSK, 2 bits/símbolo.
    """
    if bits is None or len(bits) == 0 or y is None or len(y) == 0:
        return
    mod = (modulation or "QPSK").upper()
    bps = 1 if mod == "BPSK" else 2
    total_symbols = len(y) // sps
    usable_symbols = min(max_symbols, total_symbols, (len(bits) + bps - 1) // bps)
    if usable_symbols <= 0:
        return
    n_samp = usable_symbols * sps
    seg = y[:n_samp]

    nb = usable_symbols * bps
    bseg = np.array(bits[:nb], dtype=np.uint8)

    t_iq = np.arange(n_samp) / float(sps)
    t_bits = np.arange(nb + 1) / float(bps)
    bvals = np.concatenate([bseg, bseg[-1:]]) if nb > 0 else np.zeros(1)

    plt.figure(figsize=(6.4, 4.8))
    plt.subplot(2, 1, 1)
    plt.step(t_bits, bvals, where='post')
    for k in range(0, usable_symbols + 1):
        plt.axvline(k, color='gray', lw=0.4, alpha=0.5)
    plt.ylim(-0.2, 1.2)
    plt.ylabel('Bit')
    plt.title('Bits (Lab1) y señal IQ (Lab2)')
    plt.grid(True, alpha=0.25)

    plt.subplot(2, 1, 2)
    plt.plot(t_iq, seg.real, label='I')
    plt.plot(t_iq, seg.imag, label='Q', alpha=0.85)
    for k in range(0, usable_symbols + 1):
        plt.axvline(k, color='gray', lw=0.4, alpha=0.5)
    plt.xlabel('Tiempo [símbolos]')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()

@dataclass
class Lab2Params:
    out_dir: str
    n_bits: int = 2000
    modulation: str = "QPSK"
    sps: int = 8
    rolloff: float = 0.25
    span: int = 8
    seed: int = 0
    eye_span: int = 2
    eye_traces: int = 300


def run_lab2(params: Lab2Params, bits: np.ndarray | None = None) -> Dict[str, str]:
    _ensure_dir(params.out_dir)
    if bits is None:
        rng = np.random.default_rng(params.seed)
        bits = rng.integers(0, 2, size=params.n_bits, dtype=np.uint8)
    syms = map_bits_to_symbols(bits, params.modulation)
    taps = rrc_taps(params.rolloff, params.sps, params.span)
    iq = upsample_and_filter(syms, params.sps, taps)
    paths = {
        "params": os.path.join(params.out_dir, "params.json"),
        "iq_bin": os.path.join(params.out_dir, "iq.bin"),
        "rrc_impulse_png": os.path.join(params.out_dir, "rrc_impulse.png"),
        "iq_time_png": os.path.join(params.out_dir, "iq_time.png"),
        "constellation_png": os.path.join(params.out_dir, "constellation.png"),
        "spectrum_png": os.path.join(params.out_dir, "spectrum.png"),
        "eye_png": os.path.join(params.out_dir, "eye_diagram.png"),
        "bits_iq_transition_png": os.path.join(params.out_dir, "bits_iq_transition.png"),
    }
    # Guardar parámetros (si se pasaron bits explícitos, reflejar su longitud)
    p = asdict(params)
    try:
        p["n_bits"] = int(bits.size)  # type: ignore[attr-defined]
    except Exception:
        pass
    _save_json(p, paths["params"])
    save_iq_float32(iq, paths["iq_bin"])
    plot_rrc(taps, params.sps, paths["rrc_impulse_png"])
    plot_time_iq(iq, params.sps, paths["iq_time_png"])
    plot_constellation(iq, params.sps, paths["constellation_png"])
    plot_spectrum(iq, params.sps, paths["spectrum_png"])
    eye_span = max(1, int(params.eye_span))
    eye_tr = max(10, int(params.eye_traces))
    plot_eye(iq, params.sps, paths["eye_png"], span_symbols=eye_span, max_traces=eye_tr)
    try:
        plot_bits_and_iq(bits, iq, params.sps, params.modulation, paths["bits_iq_transition_png"])
    except Exception:
        pass
    return paths


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("Lab2 – Modulación digital + RRC")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_bits", type=int, default=2000)
    ap.add_argument("--mod", choices=["BPSK", "QPSK"], default="QPSK")
    ap.add_argument("--sps", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--span", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    params = Lab2Params(
        out_dir=args.out,
        n_bits=args.n_bits,
        modulation=args.mod,
        sps=args.sps,
        rolloff=args.alpha,
        span=args.span,
        seed=args.seed,
    )
    out = run_lab2(params)
    print("Archivos generados:")
    for k, v in out.items():
        print(f"- {k}: {v}")
