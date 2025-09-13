import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample_poly
from typing import Tuple

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

def save_signal_plot(x: np.ndarray, fs: int, title: str, fname: str):
    """Guarda la señal en el tiempo (una figura por gráfico, sin estilos de color explícitos)."""
    t = np.arange(len(x)) / fs
    plt.figure()
    plt.plot(t, x)
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
    plt.xlabel('Amplitud')
    plt.ylabel('Frecuencia')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=140)
    plt.close()

def plot_hist_bits(bits, title, fname):
    """Histograma de bits (conteo de 0/1)."""
    bits = np.array(bits, dtype=np.uint8)
    counts = [np.sum(bits == 0), np.sum(bits == 1)]
    plt.figure()
    plt.bar([0, 1], counts)
    plt.xticks([0, 1], ['0', '1'])
    plt.xlabel('Bit')
    plt.ylabel('Frecuencia')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=140)
    plt.close()
