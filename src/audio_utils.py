import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample_poly
from typing import Tuple
import math


# µ-law companding (no uniform quantization)
def mu_law_compand(x: np.ndarray, mu: int = 255) -> np.ndarray:
    """
    Aplica companding µ-law a una señal en [-1,1].
    y = sign(x) * log(1 + mu*|x|) / log(1 + mu)
    """
    x = np.clip(x, -1.0, 1.0).astype(np.float32)
    return np.sign(x) * (np.log1p(mu * np.abs(x)) / np.log1p(mu))

def mu_law_expand(y: np.ndarray, mu: int = 255) -> np.ndarray:
    """
    Inversa del companding µ-law, devuelve señal en [-1,1].
    x = sign(y) * (1/mu) * ( (1+mu)^{|y|} - 1 )
    Implementado de forma numéricamente estable con expm1/log1p.
    """
    y = np.clip(y, -1.0, 1.0).astype(np.float32)
    return np.sign(y) * (np.expm1(np.log1p(mu) * np.abs(y)) / mu)

def load_wav_mono(path: str, target_fs: int = None) -> Tuple[np.ndarray, int]:
    """
    Carga un WAV y devuelve (x, fs) en mono y normalizado a [-1, 1].
    Si target_fs no es None y difiere de fs, re-muestrea con resample_poly.
    """
    fs, x = wavfile.read(path)

    # Convertir a float32 [-1, 1]
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float32) / 2147483648.0
    elif x.dtype in (np.float32, np.float64):
        x = x.astype(np.float32)
    else:
        x = x.astype(np.float32)
        m = np.max(np.abs(x)) + 1e-9
        x = x / m

    # A mono si viene estéreo
    if x.ndim == 2:
        x = x.mean(axis=1)

    # Re-muestreo si se pide target_fs
    if target_fs and target_fs != fs:
        from fractions import Fraction
        frac = Fraction(target_fs, fs).limit_denominator()
        x = resample_poly(x, frac.numerator, frac.denominator)
        fs = target_fs

    # Normalizar por seguridad
    x = x / (np.max(np.abs(x)) + 1e-9)
    return x, fs

def uniform_quantize(x: np.ndarray, bits: int = 8, xmin: float = -1.0, xmax: float = 1.0):
    """
    Cuantificación uniforme mid-rise a L = 2**bits niveles en [xmin, xmax].
    Devuelve (q, L) donde q son índices enteros en [0, L-1].
    """
    L = 2 ** bits
    x_clip = np.clip(x, xmin, xmax)
    q = np.floor((x_clip - xmin) / (xmax - xmin) * L).astype(int)
    q = np.clip(q, 0, L - 1)
    return q, L


# µ-law non-uniform quantization
def mu_law_quantize(x: np.ndarray, bits: int = 8, mu: int = 255, xmin: float = -1.0, xmax: float = 1.0):
    """
    Cuantificación NO uniforme mediante companding µ-law seguido de cuantización uniforme.
    Pasos:
      1) Companding: y = mu_law_compand(x)
      2) Cuantización uniforme de y en [-1,1] con 'bits' bits.
    Devuelve (q, L) donde q son índices enteros en [0, L-1].
    """
    y = mu_law_compand(x, mu=mu)
    return uniform_quantize(y, bits=bits, xmin=xmin, xmax=xmax)

def save_signal_plot(x: np.ndarray, fs: int, title: str, fname: str):
    """Guarda la señal en el tiempo (una figura por gráfico, sin estilos de color explícitos)."""
    t = np.arange(len(x)) / fs
    plt.figure()
    plt.plot(t, x)
    plt.axhline(0, color='gray', lw=0.6)
    plt.grid(True, alpha=0.25)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=140)
    plt.close()

def save_hist_amplitudes(x: np.ndarray, title: str, fname: str, bins: int = 50):
    """Histograma de amplitudes (una figura, sin estilos de color explícitos)."""
    plt.figure()
    plt.hist(x, bins=bins)
    plt.axvline(0, color='gray', lw=0.6)
    plt.grid(True, alpha=0.25)
    plt.xlabel('Amplitud')
    plt.ylabel('Ocurrencias')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=140)
    plt.close()

def plot_hist_bits(bits, title, fname, as_probability: bool = False):
    """Histograma de bits (0/1) como conteo o probabilidad."""
    bits = np.array(bits, dtype=np.uint8)
    n = max(len(bits), 1)
    counts = np.array([np.sum(bits == 0), np.sum(bits == 1)], dtype=float)
    vals = counts / n if as_probability else counts
    plt.figure()
    plt.bar([0, 1], vals)
    plt.xticks([0, 1], ['0', '1'])
    plt.xlabel('Bit')
    plt.ylabel('Probabilidad' if as_probability else 'Ocurrencias')
    plt.title(title)
    if as_probability:
        plt.axhline(0.5, color='gray', lw=0.6, ls='--', alpha=0.6)
    plt.grid(True, axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(fname, dpi=140)
    plt.close()

def plot_hist_bits_compare(bits1, label1, bits2, label2, title, fname, as_probability: bool = True):
    """
    Gráfico comparativo de histogramas de bits (0/1) para dos secuencias.
    Muestra barras agrupadas para 0 y 1.
    """
    b1 = np.array(bits1, dtype=np.uint8)
    b2 = np.array(bits2, dtype=np.uint8)

    n1 = max(len(b1), 1)
    n2 = max(len(b2), 1)
    c1 = np.array([np.sum(b1 == 0), np.sum(b1 == 1)], dtype=float)
    c2 = np.array([np.sum(b2 == 0), np.sum(b2 == 1)], dtype=float)

    if as_probability:
        c1 = c1 / n1
        c2 = c2 / n2

    idx = np.arange(2)
    width = 0.35

    plt.figure()
    plt.bar(idx - width/2, c1, width, label=label1)
    plt.bar(idx + width/2, c2, width, label=label2)
    plt.xticks(idx, ['0', '1'])
    plt.xlabel('Bit')
    plt.ylabel('Probabilidad' if as_probability else 'Ocurrencias')
    plt.title(title)
    plt.legend()
    if as_probability:
        plt.axhline(0.5, color='gray', lw=0.6, ls='--', alpha=0.6)
    plt.grid(True, axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(fname, dpi=140)
    plt.close()

def plot_entropy_evolution(bit_series_list, labels, fname, step: int = 10000):
    """
    Grafica la evolución de la entropía binaria H(p) calculada por tramos de tamaño 'step'
    para una o más series de bits. Cada serie se grafica como una curva.
    """
    def hb(p: float) -> float:
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)

    plt.figure()
    for bits, label in zip(bit_series_list, labels):
        b = np.array(bits, dtype=np.uint8)
        if len(b) == 0:
            xs, ys = [0], [0.0]
        else:
            steps = max(step, 1)
            xs, ys = [], []
            for i in range(0, len(b), steps):
                chunk = b[i:i+steps]
                p1 = float(chunk.mean()) if len(chunk) > 0 else 0.0
                xs.append(i + len(chunk))
                ys.append(hb(p1))
        plt.plot(xs, ys, label=label)

    plt.xlabel('Cantidad de bits procesados')
    plt.ylabel('Entropía H(p) [bits/bit]')
    plt.title('Evolución de la entropía')
    plt.legend()
    plt.axhline(1.0, color='gray', lw=0.6, ls='--', alpha=0.6)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(fname, dpi=140)
    plt.close()
