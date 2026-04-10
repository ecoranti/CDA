"""Utilidades de audio para la etapa de Formateo.

Este módulo concentra únicamente las operaciones que usamos en el flujo actual:

- carga de WAV mono normalizado en `[-1, 1]`
- cuantización uniforme
- companding y cuantización no uniforme `A-Law`
- gráficos auxiliares para inspección de señal y bits

Decisión de diseño:
    para simplificar el laboratorio y alinearlo con el estándar habitual en
    Argentina, se conserva solamente `A-Law` como opción no uniforme.
"""

from typing import Optional, Tuple
import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

matplotlib.use("Agg")


A_LAW_STANDARD = 87.6
DEFAULT_SIGNAL_RANGE = (-1.0, 1.0)


def a_law_compand(x: np.ndarray, A: float = A_LAW_STANDARD) -> np.ndarray:
    """Aplica companding A-Law a una señal normalizada.

    La entrada se satura a `[-1, 1]` y la salida queda en el mismo rango.
    Usamos `A = 87.6`, valor estándar de G.711 para esta variante.
    """
    x = np.clip(x, *DEFAULT_SIGNAL_RANGE).astype(np.float64)
    abs_x = np.abs(x)
    ln_A = np.log(A)
    norm = 1.0 + ln_A

    companded = np.where(
        abs_x <= 1.0 / A,
        A * abs_x / norm,
        (1.0 + np.log(A * abs_x + 1e-15)) / norm,
    )
    return (np.sign(x) * companded).astype(np.float32)


def a_law_expand(y: np.ndarray, A: float = A_LAW_STANDARD) -> np.ndarray:
    """Reconstruye una señal previamente comprimida con A-Law."""
    y = np.clip(y, *DEFAULT_SIGNAL_RANGE).astype(np.float64)
    abs_y = np.abs(y)
    ln_A = np.log(A)
    norm = 1.0 + ln_A

    expanded = np.where(
        abs_y <= 1.0 / norm,
        abs_y * norm / A,
        np.exp(abs_y * norm - 1.0) / A,
    )
    return (np.sign(y) * expanded).astype(np.float32)

def load_wav_mono(path: str, target_fs: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Carga un WAV, lo pasa a mono y lo normaliza en `[-1, 1]`.

    Si `target_fs` está definido y difiere de la frecuencia original,
    remuestrea usando `resample_poly`.
    """
    fs, x = wavfile.read(path)

    # Convertir a float32 en escala unitaria cuando el WAV viene en PCM entero.
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

    # Si el archivo trae más de un canal, promediamos para trabajar en mono.
    if x.ndim == 2:
        x = x.mean(axis=1)

    # Remuestreo racional para mantener buena calidad numérica.
    if target_fs and target_fs != fs:
        from fractions import Fraction
        frac = Fraction(target_fs, fs).limit_denominator()
        x = resample_poly(x, frac.numerator, frac.denominator)
        fs = target_fs

    # Re-normalización final por seguridad, independientemente del formato de origen.
    x = x / (np.max(np.abs(x)) + 1e-9)
    return x, fs


def uniform_quantize(
    x: np.ndarray,
    bits: int = 8,
    xmin: float = DEFAULT_SIGNAL_RANGE[0],
    xmax: float = DEFAULT_SIGNAL_RANGE[1],
):
    """Cuantiza de forma uniforme con esquema mid-rise.

    Retorna:
        q: índices enteros en `[0, L-1]`
        L: cantidad total de niveles, con `L = 2**bits`
    """
    L = 2 ** bits
    x_clip = np.clip(x, xmin, xmax)
    q = np.floor((x_clip - xmin) / (xmax - xmin) * L).astype(int)
    q = np.clip(q, 0, L - 1)
    return q, L


def a_law_quantize(
    x: np.ndarray,
    bits: int = 8,
    A: float = A_LAW_STANDARD,
    xmin: float = DEFAULT_SIGNAL_RANGE[0],
    xmax: float = DEFAULT_SIGNAL_RANGE[1],
):
    """Cuantiza en dos pasos: companding A-Law y cuantización uniforme.

    Esto nos permite mantener más resolución efectiva para amplitudes bajas,
    que es justamente el objetivo del cuantizador no uniforme en voz.
    """
    y = a_law_compand(x, A=A)
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

def save_signal_quantized_compare(
    x: np.ndarray,
    xhat: np.ndarray,
    fs: int,
    title: str,
    fname: str,
    max_ms: float = 40.0,
):
    """Compara un tramo corto con actividad real de la señal y su reconstrucción."""
    x = np.asarray(x, dtype=np.float32)
    xhat = np.asarray(xhat, dtype=np.float32)
    n = min(len(x), len(xhat))
    if n == 0:
        return
    win = min(n, max(1, int(fs * max_ms / 1000.0)))
    if win >= n:
        start = 0
    else:
        # Elegimos la ventana de mayor energía RMS para evitar mostrar silencios
        power = x[:n] * x[:n]
        kernel = np.ones(win, dtype=np.float32)
        energy = np.convolve(power, kernel, mode="valid")
        start = int(np.argmax(energy)) if energy.size else 0
    stop = start + win
    t = np.arange(win) / float(fs)
    plt.figure()
    plt.plot(t, x[start:stop], label="Original", lw=1.2, color="#1f77b4")
    plt.step(t, xhat[start:stop], where="mid", label="Reconstruida", lw=1.0, alpha=0.9, color="#ff7f0e")
    plt.grid(True, alpha=0.25)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    plt.title(title)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    plt.tight_layout()
    plt.savefig(fname, dpi=140)
    plt.close()

def save_quantizer_characteristic(
    x_in: np.ndarray,
    x_out: np.ndarray,
    title: str,
    fname: str,
):
    """Grafica la característica entrada/salida del cuantizador reconstruido."""
    x_in = np.asarray(x_in, dtype=np.float32)
    x_out = np.asarray(x_out, dtype=np.float32)
    n = min(len(x_in), len(x_out))
    if n == 0:
        return
    order = np.argsort(x_in[:n])
    xin = x_in[:n][order]
    xout = x_out[:n][order]
    plt.figure()
    plt.plot(xin, xin, ls="--", lw=0.9, color="gray", label="Ideal y=x")
    plt.plot(xin, xout, lw=1.2, label="Cuantizador")
    plt.grid(True, alpha=0.25)
    plt.xlabel("Entrada normalizada")
    plt.ylabel("Salida reconstruida")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=140)
    plt.close()

def _select_low_level_window(x: np.ndarray, fs: int, max_ms: float = 40.0) -> tuple[int, int]:
    """Selecciona una ventana con energía baja-media pero no nula para comparar cuantizadores."""
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    if n == 0:
        return 0, 0
    win = min(n, max(1, int(fs * max_ms / 1000.0)))
    if win >= n:
        return 0, win
    power = x * x
    energy = np.convolve(power, np.ones(win, dtype=np.float32), mode="valid") / float(win)
    nz = energy[energy > 1e-8]
    if nz.size == 0:
        return 0, win
    target = float(np.quantile(nz, 0.35))
    idx = int(np.argmin(np.abs(energy - target)))
    return idx, idx + win

def save_quantizer_low_level_compare(
    x: np.ndarray,
    xhat_uniform: np.ndarray,
    xhat_alaw: np.ndarray,
    fs: int,
    title: str,
    fname: str,
    max_ms: float = 40.0,
):
    """Compara ambos cuantizadores sobre un tramo de amplitud baja-media."""
    x = np.asarray(x, dtype=np.float32)
    xu = np.asarray(xhat_uniform, dtype=np.float32)
    xa = np.asarray(xhat_alaw, dtype=np.float32)
    n = min(len(x), len(xu), len(xa))
    if n == 0:
        return
    start, stop = _select_low_level_window(x[:n], fs, max_ms=max_ms)
    t = np.arange(stop - start) / float(fs)
    plt.figure()
    plt.plot(t, x[start:stop], label="Original", lw=1.2, color="#1f77b4")
    plt.step(t, xu[start:stop], where="mid", label="Uniforme", lw=1.0, color="#ff7f0e")
    plt.step(t, xa[start:stop], where="mid", label="A-law", lw=1.0, color="#2ca02c")
    plt.grid(True, alpha=0.25)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    plt.title(title)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    plt.tight_layout()
    plt.savefig(fname, dpi=140)
    plt.close()

def save_quantization_error_compare(
    x: np.ndarray,
    xhat_uniform: np.ndarray,
    xhat_alaw: np.ndarray,
    fs: int,
    title: str,
    fname: str,
    max_ms: float = 40.0,
):
    """Compara el error de cuantización de uniforme y A-law sobre el mismo tramo."""
    x = np.asarray(x, dtype=np.float32)
    xu = np.asarray(xhat_uniform, dtype=np.float32)
    xa = np.asarray(xhat_alaw, dtype=np.float32)
    n = min(len(x), len(xu), len(xa))
    if n == 0:
        return
    start, stop = _select_low_level_window(x[:n], fs, max_ms=max_ms)
    t = np.arange(stop - start) / float(fs)
    eu = xu[start:stop] - x[start:stop]
    ea = xa[start:stop] - x[start:stop]
    plt.figure()
    plt.axhline(0, color="gray", lw=0.8, ls="--")
    plt.plot(t, eu, label="Error uniforme", lw=1.0, color="#ff7f0e")
    plt.plot(t, ea, label="Error A-law", lw=1.0, color="#2ca02c")
    plt.grid(True, alpha=0.25)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Error")
    plt.title(title)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
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
