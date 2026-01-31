"""Utilidades para evaluar MSE y SQNR de cuantización uniforme y µ-law.

Este módulo expone funciones usadas por src/main.py:
    - mse(x, y)
    - sqnr_db(x, y)
    - eval_uniform(x, bits)
    - eval_mulaw(x, mu, bits)
"""

from typing import Tuple
import numpy as np

from .audio_utils import (
    uniform_quantize,
    mu_law_compand,
    mu_law_expand,
)


def _reconstruct_midrise(q: np.ndarray, bits: int, xmin: float = -1.0, xmax: float = 1.0) -> np.ndarray:
    """Reconstruye niveles del cuantizador mid-rise a partir de índices q en [0, 2^bits-1]."""
    L = 2 ** bits
    q = np.asarray(q, dtype=np.int64)
    step = (xmax - xmin) / L
    # nivel representativo: centro del bin
    return xmin + (q.astype(np.float32) + 0.5) * step


def eval_uniform(x: np.ndarray, bits: int = 8, xmin: float = -1.0, xmax: float = 1.0) -> np.ndarray:
    """Cuantiza x de forma uniforme (mid-rise) y devuelve la señal reconstruida xhat."""
    x = np.asarray(x, dtype=np.float32)
    q, _ = uniform_quantize(x, bits=bits, xmin=xmin, xmax=xmax)
    xhat = _reconstruct_midrise(q, bits=bits, xmin=xmin, xmax=xmax)
    return xhat.astype(np.float32)


def eval_mulaw(x: np.ndarray, mu: int = 255, bits: int = 8, xmin: float = -1.0, xmax: float = 1.0) -> np.ndarray:
    """Cuantiza x con companding µ-law + cuantización uniforme y devuelve xhat expandida."""
    x = np.asarray(x, dtype=np.float32)
    y = mu_law_compand(x, mu=mu)
    q, _ = uniform_quantize(y, bits=bits, xmin=xmin, xmax=xmax)
    yhat = _reconstruct_midrise(q, bits=bits, xmin=xmin, xmax=xmax)
    xhat = mu_law_expand(yhat, mu=mu)
    return xhat.astype(np.float32)


def mse(x: np.ndarray, y: np.ndarray) -> float:
    """Error cuadrático medio entre x e y."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    diff = x - y
    return float(np.mean(diff * diff))


def sqnr_db(x: np.ndarray, y: np.ndarray) -> float:
    """SQNR en dB = 10*log10(P_signal / MSE)."""
    x = np.asarray(x, dtype=np.float32)
    m = mse(x, y)
    ps = float(np.mean(x * x)) + 1e-12
    return 10.0 * float(np.log10(ps / (m + 1e-20)))
