from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.special import erfc

from . import lab2_rrc
from .audio_utils import a_law_expand, load_wav_mono, save_signal_quantized_compare
from .bits_utils import bits_to_bytes
from .scrambling import scramble

matplotlib.use("Agg")


@dataclass
class Lab3Params:
    out_dir: str
    n_bits: int = 10000
    modulation: str = "QPSK"
    m_order: int = 4
    sps: int = 8
    rolloff: float = 0.25
    span: int = 8
    ebn0_start: float = 0.0
    ebn0_end: float = 12.0
    ebn0_step: float = 2.0
    trials_per_ebn0: int = 20
    theory_points: int = 300
    seed: int = 0
    channel_mode: str = "awgn"
    use_rx_rrc: bool = True
    timing_offset_ts: float = 0.0


def _validate_m_order(m_order: int) -> int:
    M = int(m_order)
    if M < 2:
        raise ValueError("M debe ser >= 2")
    if M & (M - 1):
        raise ValueError("M debe ser potencia de 2")
    return M


def _bps(modulation: str, m_order: int = 4) -> int:
    mod = (modulation or "QPSK").upper()
    if mod == "BPSK":
        return 1
    if mod == "QPSK":
        return 2
    if mod in {"MPSK", "M-PSK"}:
        M = _validate_m_order(m_order)
        return int(np.log2(M))
    raise ValueError(f"Modulación no soportada: {modulation}")


def _n_symbols_from_bits(n_bits: int, modulation: str, m_order: int = 4) -> int:
    bps = _bps(modulation, m_order=m_order)
    return (int(n_bits) + bps - 1) // bps


def _mod_label(modulation: str, m_order: int = 4) -> str:
    mod = (modulation or "QPSK").upper()
    if mod in {"MPSK", "M-PSK"}:
        return f"{_validate_m_order(m_order)}-PSK"
    return mod


def add_noise(
    signal: np.ndarray,
    ebn0_db: float,
    sps: int,
    bits_per_symbol: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Agrega AWGN complejo respetando la relación Eb/N0."""
    rx, _ = add_noise_with_noise(signal, ebn0_db, sps, bits_per_symbol, rng=rng)
    return rx


def add_noise_with_noise(
    signal: np.ndarray,
    ebn0_db: float,
    sps: int,
    bits_per_symbol: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve señal con ruido y el vector de ruido aplicado (conocido en simulación)."""
    if rng is None:
        rng = np.random.default_rng()
    p_signal = float(np.mean(np.abs(signal) ** 2))
    esn0_db = float(ebn0_db) + 10.0 * np.log10(float(bits_per_symbol))
    snr_db = esn0_db - 10.0 * np.log10(float(sps))
    snr_linear = 10.0 ** (snr_db / 10.0)
    p_noise = p_signal / snr_linear
    sigma = np.sqrt(p_noise / 2.0)
    noise = sigma * (rng.standard_normal(signal.size) + 1j * rng.standard_normal(signal.size))
    return signal + noise, noise


def estimate_ebn0_db_from_signal_noise(
    signal: np.ndarray,
    noise: np.ndarray,
    sps: int,
    bits_per_symbol: int,
) -> tuple[float, float]:
    """Estima SNR y Eb/N0 desde señal+ruido conocidos (modo laboratorio)."""
    ps = float(np.mean(np.abs(signal) ** 2))
    pn = float(np.mean(np.abs(noise) ** 2))
    snr_lin = ps / max(pn, 1e-18)
    ebn0_lin = snr_lin * float(sps) / float(bits_per_symbol)
    snr_db = 10.0 * np.log10(max(snr_lin, 1e-18))
    ebn0_db = 10.0 * np.log10(max(ebn0_lin, 1e-18))
    return snr_db, ebn0_db


def matched_filter(rx_signal: np.ndarray, sps: int, alpha: float, span: int) -> tuple[np.ndarray, np.ndarray]:
    taps = lab2_rrc.rrc_taps(alpha, sps, span)
    y = np.convolve(rx_signal, taps, mode="full")
    return y, taps


def downsample(
    filtered_signal: np.ndarray,
    sps: int,
    delay_samples: int,
    n_symbols: int,
    timing_offset_ts: float = 0.0,
) -> np.ndarray:
    # Permite muestreo desfasado fraccionalmente respecto de Ts para estudiar degradación temporal.
    idxs = sampling_indices(len(filtered_signal), sps, delay_samples, n_symbols, timing_offset_ts)
    if idxs.size == 0:
        return np.array([], dtype=np.complex64)
    n = np.arange(len(filtered_signal), dtype=np.float64)
    y_r = np.interp(idxs, n, filtered_signal.real)
    y_i = np.interp(idxs, n, filtered_signal.imag)
    return (y_r + 1j * y_i).astype(np.complex64)


def sampling_indices(
    signal_len: int,
    sps: int,
    delay_samples: int,
    n_symbols: int,
    timing_offset_ts: float = 0.0,
) -> np.ndarray:
    """Índices fraccionales de muestreo kT_s usados por el downsampling."""
    offset_samples = float(timing_offset_ts) * float(sps)
    idxs = delay_samples + offset_samples + np.arange(n_symbols, dtype=np.float64) * float(sps)
    valid = (idxs >= 0.0) & (idxs <= (signal_len - 1))
    return idxs[valid]


def demap_qpsk(samples: np.ndarray) -> np.ndarray:
    bits_0 = np.where(samples.real > 0, 0, 1)
    bits_1 = np.where(samples.imag > 0, 0, 1)
    out = np.empty(2 * len(samples), dtype=np.uint8)
    out[0::2] = bits_0
    out[1::2] = bits_1
    return out


def demap_bpsk(samples: np.ndarray) -> np.ndarray:
    return np.where(samples.real > 0, 1, 0).astype(np.uint8)


def demap_mpsk(samples: np.ndarray, m_order: int) -> np.ndarray:
    M = _validate_m_order(m_order)
    k = int(np.log2(M))
    if samples.size == 0:
        return np.zeros(0, dtype=np.uint8)
    angles = np.angle(samples.astype(np.complex128))
    angles = np.mod(angles, 2.0 * np.pi)
    idx = np.mod(np.round(angles * M / (2.0 * np.pi)).astype(np.int64), M)
    out = np.zeros(samples.size * k, dtype=np.uint8)
    for b in range(k):
        shift = k - 1 - b
        out[b::k] = ((idx >> shift) & 1).astype(np.uint8)
    return out


def calculate_ber(tx_bits: np.ndarray, rx_bits: np.ndarray) -> tuple[float, int]:
    L = min(len(tx_bits), len(rx_bits))
    if L == 0:
        return 0.0, 0
    tx = tx_bits[:L]
    rx = rx_bits[:L]
    n_errors = int(np.sum(tx != rx))
    return float(n_errors) / float(L), n_errors


def _reconstruct_midrise(q: np.ndarray, bits: int, xmin: float = -1.0, xmax: float = 1.0) -> np.ndarray:
    L = 2 ** bits
    q = np.asarray(q, dtype=np.int64)
    step = (xmax - xmin) / L
    return xmin + (q.astype(np.float32) + 0.5) * step


def _bits_to_ints(bits: np.ndarray, bits_per_word: int) -> np.ndarray:
    arr = np.asarray(bits, dtype=np.uint8).ravel()
    n_words = arr.size // bits_per_word
    if n_words <= 0:
        return np.zeros(0, dtype=np.int64)
    arr = arr[: n_words * bits_per_word].reshape(n_words, bits_per_word)
    weights = (2 ** np.arange(bits_per_word - 1, -1, -1, dtype=np.int64)).reshape(1, -1)
    return (arr.astype(np.int64) * weights).sum(axis=1)


def _save_wav_from_unit_signal(x: np.ndarray, fs: int, path: Path) -> None:
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    y = np.round(x * 32767.0).astype(np.int16)
    wavfile.write(str(path), int(fs), y)


def _recover_audio_from_bits(bits_rx: np.ndarray, lab1_meta: dict | None, out_dir: Path) -> dict[str, str]:
    if not lab1_meta:
        return {}
    source = (lab1_meta.get("source") or "audio").lower()
    if source != "audio":
        return {}

    fs = int(lab1_meta.get("fs") or 16000)
    n_bits = int(lab1_meta.get("n_bits") or 8)
    quantizer = (lab1_meta.get("quantizer") or "alaw").lower()
    lfsr_seed = int(lab1_meta.get("lfsr_seed") or int("0b1010110011", 2))
    taps_meta = lab1_meta.get("lfsr_taps") or [9, 6]
    lfsr_taps = tuple(int(t) for t in taps_meta)
    lfsr_bitwidth = int(lab1_meta.get("lfsr_bitwidth") or 10)
    audio_path = lab1_meta.get("audio")

    bits_descr = np.asarray(
        scramble(np.asarray(bits_rx, dtype=np.uint8).ravel().tolist(), seed=lfsr_seed, taps=lfsr_taps, bitwidth=lfsr_bitwidth),
        dtype=np.uint8,
    )
    q_idx = _bits_to_ints(bits_descr, n_bits)
    if q_idx.size == 0:
        return {}

    yhat = _reconstruct_midrise(q_idx, bits=n_bits, xmin=-1.0, xmax=1.0).astype(np.float32)
    if quantizer == "alaw":
        xhat = a_law_expand(yhat)
    else:
        xhat = yhat
    xhat = np.asarray(xhat, dtype=np.float32)

    paths: dict[str, str] = {}
    rx_wav = out_dir / "audio_rx.wav"
    _save_wav_from_unit_signal(xhat, fs, rx_wav)
    paths["audio_rx_wav"] = str(rx_wav)

    if audio_path:
        try:
            x_ref, _ = load_wav_mono(str(audio_path), target_fs=fs)
            n = min(len(x_ref), len(xhat))
            x_ref = x_ref[:n].astype(np.float32)
            x_cmp = xhat[:n].astype(np.float32)
            ref_wav = out_dir / "audio_tx_ref.wav"
            _save_wav_from_unit_signal(x_ref, fs, ref_wav)
            paths["audio_tx_ref_wav"] = str(ref_wav)
            cmp_png = out_dir / "audio_compare_rx.png"
            save_signal_quantized_compare(
                x_ref,
                x_cmp,
                fs,
                "Audio original vs recuperado",
                str(cmp_png),
            )
            paths["audio_compare_rx_png"] = str(cmp_png)
        except Exception:
            pass
    return paths


def _recover_text_from_bits(bits_rx: np.ndarray, lab1_meta: dict | None, out_dir: Path) -> dict[str, str]:
    if not lab1_meta:
        return {}
    source = (lab1_meta.get("source") or "audio").lower()
    if source != "text":
        return {}

    lfsr_seed = int(lab1_meta.get("lfsr_seed") or int("0b1010110011", 2))
    taps_meta = lab1_meta.get("lfsr_taps") or [9, 6]
    lfsr_taps = tuple(int(t) for t in taps_meta)
    lfsr_bitwidth = int(lab1_meta.get("lfsr_bitwidth") or 10)
    text_path = lab1_meta.get("text")

    bits_descr = np.asarray(
        scramble(np.asarray(bits_rx, dtype=np.uint8).ravel().tolist(), seed=lfsr_seed, taps=lfsr_taps, bitwidth=lfsr_bitwidth),
        dtype=np.uint8,
    )
    if bits_descr.size == 0:
        return {}

    raw_bytes = bytes(bits_to_bytes(bits_descr.tolist()))
    text_rx = raw_bytes.decode("utf-8", errors="replace")

    paths: dict[str, str] = {}
    rx_txt = out_dir / "text_rx.txt"
    rx_txt.write_text(text_rx, encoding="utf-8")
    paths["text_rx_txt"] = str(rx_txt)

    if text_path:
        try:
            ref_text = Path(text_path).read_text(encoding="utf-8")
            ref_txt = out_dir / "text_tx_ref.txt"
            ref_txt.write_text(ref_text, encoding="utf-8")
            paths["text_tx_ref_txt"] = str(ref_txt)
        except Exception:
            pass
    return paths


def _recover_source_from_bits(bits_rx: np.ndarray, lab1_meta: dict | None, out_dir: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    paths.update(_recover_audio_from_bits(bits_rx, lab1_meta, out_dir))
    paths.update(_recover_text_from_bits(bits_rx, lab1_meta, out_dir))
    return paths


def theoretical_ber_mpsk(
    ebn0_db_arr: np.ndarray | list[float],
    modulation: str,
    m_order: int = 4,
) -> np.ndarray:
    ebn0_lin = 10 ** (np.array(ebn0_db_arr, dtype=np.float64) / 10.0)
    mod = (modulation or "QPSK").upper()
    if mod in {"BPSK", "QPSK"}:
        return 0.5 * erfc(np.sqrt(ebn0_lin))
    if mod in {"MPSK", "M-PSK"}:
        M = _validate_m_order(m_order)
        if M == 2:
            return 0.5 * erfc(np.sqrt(ebn0_lin))
        k = int(np.log2(M))
        # Aproximación de BER para M-PSK coherente (mapeo Gray aproximado).
        arg = np.sqrt(2.0 * k * ebn0_lin) * np.sin(np.pi / float(M))
        q_arg = 0.5 * erfc(arg / np.sqrt(2.0))
        ber = (2.0 / float(k)) * q_arg
        return np.clip(ber, 1e-12, 0.5)
    raise ValueError(f"Modulación no soportada para BER teórica: {modulation}")


def _plot_mf_impulse_and_freq(taps: np.ndarray, sps: int, impulse_path: Path, freq_path: Path) -> None:
    t = np.arange(-(len(taps) // 2), len(taps) // 2 + 1) / float(sps)
    if t.size > taps.size:
        t = t[:-1]
    plt.figure()
    plt.plot(t, taps)
    plt.axhline(0, color="gray", lw=0.6)
    plt.axvline(0, color="gray", lw=0.6)
    plt.grid(True, alpha=0.25)
    plt.xlabel("Tiempo [simbolos]")
    plt.ylabel("h_MF[n]")
    plt.title("Filtro acoplado - respuesta impulsiva")
    plt.tight_layout()
    plt.savefig(impulse_path, dpi=140)
    plt.close()

    nfft = 4096
    H = np.fft.fftshift(np.fft.fft(taps, n=nfft))
    f = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / float(sps)))
    HdB = 20.0 * np.log10(np.maximum(np.abs(H), 1e-12))
    plt.figure()
    plt.plot(f, HdB)
    plt.axvline(0, color="gray", lw=0.6)
    plt.grid(True, alpha=0.25)
    plt.xlabel("Frecuencia [ciclos/simbolo]")
    plt.ylabel("|H_MF(f)| [dB]")
    plt.title("Filtro acoplado - respuesta en frecuencia")
    plt.tight_layout()
    plt.savefig(freq_path, dpi=140)
    plt.close()


def _plot_rx_frontend_impulse_and_freq(
    taps: np.ndarray,
    sps: int,
    impulse_path: Path,
    freq_path: Path,
    use_rx_rrc: bool,
) -> None:
    if use_rx_rrc:
        _plot_mf_impulse_and_freq(taps, sps, impulse_path, freq_path)
        return

    plt.figure()
    plt.stem(np.arange(len(taps)), taps.real, basefmt=" ", linefmt="C0-", markerfmt="C0o")
    plt.grid(True, alpha=0.25)
    plt.xlabel("Indice de muestra")
    plt.ylabel("h[n]")
    plt.title("Frente Rx sin filtro acoplado")
    plt.tight_layout()
    plt.savefig(impulse_path, dpi=140)
    plt.close()

    f = np.linspace(-0.5 * sps, 0.5 * sps, 512, endpoint=False)
    HdB = np.zeros_like(f)
    plt.figure()
    plt.plot(f, HdB)
    plt.axvline(0, color="gray", lw=0.6)
    plt.grid(True, alpha=0.25)
    plt.xlabel("Frecuencia [ciclos/simbolo]")
    plt.ylabel("|H(f)| [dB]")
    plt.title("Frente Rx sin filtro acoplado")
    plt.tight_layout()
    plt.savefig(freq_path, dpi=140)
    plt.close()


def _plot_tx_rx_constellations(
    tx_syms: np.ndarray,
    rx_syms: np.ndarray,
    out_path: Path,
    ebn0: float,
    channel_mode: str = "awgn",
    edge_trim: int = 4,
    viz_scale: float = 1.0,
) -> None:
    if edge_trim > 0:
        if tx_syms.size > 2 * edge_trim:
            tx_plot = tx_syms[edge_trim:-edge_trim]
        else:
            tx_plot = tx_syms
        if rx_syms.size > 2 * edge_trim:
            rx_plot = rx_syms[edge_trim:-edge_trim]
        else:
            rx_plot = rx_syms
    else:
        tx_plot = tx_syms
        rx_plot = rx_syms
    if viz_scale > 0:
        rx_plot = rx_plot / float(viz_scale)

    all_vals = np.concatenate([tx_plot.real, tx_plot.imag, rx_plot.real, rx_plot.imag])
    m = float(np.max(np.abs(all_vals))) if all_vals.size else 1.0
    lim = max(1.0, 1.15 * m)
    plt.figure(figsize=(8.5, 3.8))
    plt.subplot(1, 2, 1)
    plt.scatter(tx_plot.real, tx_plot.imag, s=8, alpha=0.6)
    plt.axhline(0, color="gray", lw=0.6)
    plt.axvline(0, color="gray", lw=0.6)
    plt.grid(True, alpha=0.25)
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Constelacion Tx")
    plt.xlabel("I")
    plt.ylabel("Q")

    plt.subplot(1, 2, 2)
    plt.scatter(rx_plot.real, rx_plot.imag, s=8, alpha=0.6)
    plt.axhline(0, color="gray", lw=0.6)
    plt.axvline(0, color="gray", lw=0.6)
    plt.grid(True, alpha=0.25)
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.gca().set_aspect("equal", adjustable="box")
    mode = (channel_mode or "awgn").lower()
    if mode == "ideal":
        plt.title("Constelacion Rx (canal ideal)")
    else:
        plt.title(f"Constelacion Rx (Eb/N0={ebn0:.1f} dB)")
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _plot_rx_constellation(
    rx_syms: np.ndarray,
    out_path: Path,
    channel_mode: str = "awgn",
    ebn0: float = 0.0,
    edge_trim: int = 4,
    viz_scale: float = 1.0,
) -> None:
    if edge_trim > 0 and rx_syms.size > 2 * edge_trim:
        rx_plot = rx_syms[edge_trim:-edge_trim]
    else:
        rx_plot = rx_syms
    if rx_plot.size == 0:
        rx_plot = rx_syms
    if viz_scale > 0:
        rx_plot = rx_plot / float(viz_scale)

    all_vals = np.concatenate([rx_plot.real, rx_plot.imag]) if rx_plot.size else np.array([0.0])
    m = float(np.max(np.abs(all_vals))) if all_vals.size else 1.0
    lim = max(1.0, 1.15 * m)

    plt.figure(figsize=(4.6, 4.0))
    plt.scatter(rx_plot.real, rx_plot.imag, s=8, alpha=0.6)
    plt.axhline(0, color="gray", lw=0.6)
    plt.axvline(0, color="gray", lw=0.6)
    plt.grid(True, alpha=0.25)
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.gca().set_aspect("equal", adjustable="box")
    mode = (channel_mode or "awgn").lower()
    if mode == "ideal":
        plt.title("Constelacion Rx (canal ideal)")
    else:
        plt.title(f"Constelacion Rx (Eb/N0={ebn0:.1f} dB)")
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _plot_rx_decision(
    samples: np.ndarray,
    modulation: str,
    out_path: Path,
    max_sym: int = 220,
    use_rx_rrc: bool = True,
    viz_scale: float = 1.0,
) -> None:
    seg = samples[:max_sym]
    if seg.size == 0:
        return
    if viz_scale > 0:
        seg = seg / float(viz_scale)
    mod = modulation.upper()
    x = np.arange(seg.size)
    plt.figure(figsize=(8, 4.2))
    if mod == "BPSK":
        plt.plot(x, seg.real, marker=".", linestyle="-", alpha=0.75, label="I muestreada")
        plt.axhline(0, color="red", lw=1.0, linestyle="--", label="Umbral ML/MAP")
        plt.ylabel("I")
    else:
        plt.subplot(2, 1, 1)
        plt.plot(x, seg.real, marker=".", linestyle="-", alpha=0.75, label="I muestreada")
        plt.axhline(0, color="red", lw=1.0, linestyle="--", label="Umbral ML/MAP")
        plt.ylabel("I")
        plt.grid(True, alpha=0.25)
        plt.legend(loc="upper right")
        plt.subplot(2, 1, 2)
        plt.plot(x, seg.imag, marker=".", linestyle="-", alpha=0.75, color="#0ea5e9", label="Q muestreada")
        plt.axhline(0, color="red", lw=1.0, linestyle="--", label="Umbral ML/MAP")
        plt.ylabel("Q")
    plt.xlabel("Indice de simbolo")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper right")
    if use_rx_rrc:
        plt.suptitle("Salida del filtro acoplado y decision ML/MAP")
    else:
        plt.suptitle("Salida Rx sin filtro acoplado y decision ML/MAP")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _plot_downsampling(
    rx_filtered: np.ndarray,
    sample_idxs: np.ndarray,
    sampled_syms: np.ndarray,
    sps: int,
    out_path: Path,
    max_samples: int = 800,
    viz_scale: float = 1.0,
) -> None:
    if rx_filtered.size == 0 or sample_idxs.size == 0:
        return

    n_end = min(int(max_samples), int(rx_filtered.size))
    n = np.arange(n_end, dtype=np.float64)
    if viz_scale > 0:
        i_sig = (rx_filtered.real[:n_end] / float(viz_scale))
        q_sig = (rx_filtered.imag[:n_end] / float(viz_scale))
    else:
        i_sig = rx_filtered.real[:n_end]
        q_sig = rx_filtered.imag[:n_end]

    # Muestras dentro de la ventana graficada
    mask = (sample_idxs >= 0.0) & (sample_idxs < float(n_end))
    idx_win = sample_idxs[mask]
    sym_win = sampled_syms[: idx_win.size]
    if viz_scale > 0:
        sym_win = sym_win / float(viz_scale)
    if idx_win.size == 0:
        return

    plt.figure(figsize=(8.2, 4.5))
    plt.plot(n, i_sig, label="z_I[n]", color="#2563eb", lw=1.0)
    plt.plot(n, q_sig, label="z_Q[n]", color="#f97316", lw=1.0, alpha=0.9)
    plt.plot(idx_win, sym_win.real, "o", ms=3.8, color="#1d4ed8", label="Muestras I @ kT_s")
    plt.plot(idx_win, sym_win.imag, "o", ms=3.8, color="#ea580c", label="Muestras Q @ kT_s")
    for xv in idx_win:
        plt.axvline(xv, color="#94a3b8", lw=0.6, alpha=0.25)
    plt.axhline(0, color="gray", lw=0.6)
    plt.grid(True, alpha=0.25)
    plt.xlabel("Índice de muestra n")
    plt.ylabel("Amplitud")
    plt.title(f"Downsampling en Rx (1 muestra/símbolo, sps={sps})")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _estimate_rx_viz_scale(
    tx_syms: np.ndarray,
    rx_syms: np.ndarray,
    edge_trim: int = 8,
) -> float:
    """Escala para visualización: alinea potencia media Rx con Tx sin afectar la detección."""
    if tx_syms.size == 0 or rx_syms.size == 0:
        return 1.0
    if edge_trim > 0:
        tx = tx_syms[edge_trim:-edge_trim] if tx_syms.size > 2 * edge_trim else tx_syms
        rx = rx_syms[edge_trim:-edge_trim] if rx_syms.size > 2 * edge_trim else rx_syms
    else:
        tx, rx = tx_syms, rx_syms
    if tx.size == 0 or rx.size == 0:
        return 1.0
    p_tx = float(np.mean(np.abs(tx) ** 2))
    p_rx = float(np.mean(np.abs(rx) ** 2))
    if p_tx <= 0.0 or p_rx <= 0.0:
        return 1.0
    scale = np.sqrt(p_rx / p_tx)
    if not np.isfinite(scale) or scale <= 0:
        return 1.0
    return float(scale)


def _plot_ber_point(
    ber: float,
    ebn0_val: float,
    out_path: Path,
    modulation: str,
    m_order: int = 4,
) -> None:
    eb_min, eb_max = -2.0, max(12.0, ebn0_val + 2.0)
    eb_arr = np.linspace(eb_min, eb_max, 200)
    ber_theo = theoretical_ber_mpsk(eb_arr, modulation=modulation, m_order=m_order)
    mod_lbl = _mod_label(modulation, m_order=m_order)
    plt.figure()
    plt.semilogy(eb_arr, ber_theo, "k-", label=f"Teorica ({mod_lbl})")
    plt.semilogy([ebn0_val], [ber], "ro", markersize=8, label="Simulada")
    plt.grid(True, which="both", alpha=0.3)
    plt.xlabel("Eb/N0 [dB]")
    plt.ylabel("BER")
    plt.title(f"Punto de operacion (Eb/N0={ebn0_val:.1f} dB)")
    plt.legend()
    plt.ylim(bottom=1e-6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _run_chain(
    bits_tx: np.ndarray,
    tx_signal: np.ndarray,
    modulation: str,
    ebn0: float,
    sps: int,
    rolloff: float,
    span: int,
    seed: int,
    m_order: int = 4,
    channel_mode: str = "awgn",
    use_rx_rrc: bool = True,
    timing_offset_ts: float = 0.0,
) -> dict:
    mod = modulation.upper()
    rng = np.random.default_rng(seed)
    bps = _bps(mod, m_order=m_order)
    mode = (channel_mode or "awgn").lower()
    if mode == "ideal":
        rx_noisy = np.asarray(tx_signal, dtype=np.complex64)
        noise = np.zeros_like(rx_noisy)
        snr_est_db = float("inf")
        ebn0_est_db = float("inf")
    else:
        rx_noisy, noise = add_noise_with_noise(tx_signal, ebn0, sps, bps, rng=rng)
        snr_est_db, ebn0_est_db = estimate_ebn0_db_from_signal_noise(tx_signal, noise, sps, bps)
    if use_rx_rrc:
        rx_filtered, taps_rx = matched_filter(rx_noisy, sps, rolloff, span)
        delay_rx = (len(taps_rx) - 1) // 2
    else:
        rx_filtered = np.asarray(rx_noisy, dtype=np.complex64)
        taps_rx = np.array([1.0], dtype=np.float64)
        delay_rx = 0
    n_syms_expected = _n_symbols_from_bits(len(bits_tx), mod, m_order=m_order)
    sample_idxs = sampling_indices(
        len(rx_filtered),
        sps,
        delay_rx,
        n_syms_expected,
        timing_offset_ts=timing_offset_ts,
    )
    rx_syms = downsample(
        rx_filtered,
        sps,
        delay_rx,
        n_syms_expected,
        timing_offset_ts=timing_offset_ts,
    )
    if mod == "QPSK":
        bits_rx = demap_qpsk(rx_syms)
    elif mod in {"MPSK", "M-PSK"}:
        bits_rx = demap_mpsk(rx_syms, m_order=m_order)
    else:
        bits_rx = demap_bpsk(rx_syms)
    bits_rx = bits_rx[: len(bits_tx)]
    ber, n_errors = calculate_ber(bits_tx, bits_rx)
    return {
        "rx_filtered": rx_filtered,
        "rx_syms": rx_syms,
        "bits_rx": bits_rx,
        "ber": ber,
        "n_errors": n_errors,
        "taps_rx": taps_rx,
        "snr_est_db": snr_est_db,
        "ebn0_est_db": ebn0_est_db,
        "rx_channel": rx_noisy,
        "sample_idxs": sample_idxs,
    }


def _render_single_outputs(
    out_dir: Path,
    tx_syms: np.ndarray,
    rx_syms: np.ndarray,
    rx_filtered: np.ndarray,
    taps_rx: np.ndarray,
    sample_idxs: np.ndarray,
    ebn0: float,
    ber: float,
    sps: int,
    modulation: str,
    m_order: int = 4,
    channel_mode: str = "awgn",
    use_rx_rrc: bool = True,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    p = out_dir
    p.mkdir(parents=True, exist_ok=True)

    rx_viz_scale = _estimate_rx_viz_scale(tx_syms, rx_syms, edge_trim=8)
    rx_filtered_viz = rx_filtered / rx_viz_scale if rx_viz_scale > 0 else rx_filtered

    lab2_rrc.plot_time_iq(rx_filtered_viz, sps, str(p / "rx_time.png"), nsamples=800)
    paths["rx_time_png"] = str(p / "rx_time.png")

    lab2_rrc.plot_eye(rx_filtered_viz, sps, str(p / "rx_eye.png"))
    paths["rx_eye_png"] = str(p / "rx_eye.png")

    _plot_rx_constellation(
        rx_syms,
        p / "rx_constellation.png",
        channel_mode=channel_mode,
        ebn0=ebn0,
        viz_scale=rx_viz_scale,
    )
    paths["rx_constellation_png"] = str(p / "rx_constellation.png")

    _plot_ber_point(ber, ebn0, p / "ber_point.png", modulation=modulation, m_order=m_order)
    paths["ber_point_png"] = str(p / "ber_point.png")

    _plot_tx_rx_constellations(
        tx_syms,
        rx_syms,
        p / "tx_rx_constellations.png",
        ebn0,
        channel_mode=channel_mode,
        viz_scale=rx_viz_scale,
    )
    paths["tx_rx_constellations_png"] = str(p / "tx_rx_constellations.png")

    _plot_rx_frontend_impulse_and_freq(
        taps_rx,
        sps,
        p / "mf_impulse.png",
        p / "mf_freq.png",
        use_rx_rrc=use_rx_rrc,
    )
    paths["mf_impulse_png"] = str(p / "mf_impulse.png")
    paths["mf_freq_png"] = str(p / "mf_freq.png")

    _plot_rx_decision(
        rx_syms,
        modulation,
        p / "rx_decision.png",
        use_rx_rrc=use_rx_rrc,
        viz_scale=rx_viz_scale,
    )
    paths["rx_decision_png"] = str(p / "rx_decision.png")
    _plot_downsampling(
        rx_filtered,
        sample_idxs,
        rx_syms,
        sps,
        p / "rx_downsampling.png",
        viz_scale=rx_viz_scale,
    )
    paths["rx_downsampling_png"] = str(p / "rx_downsampling.png")

    return paths


def _generate_diag_outputs_from_chain(
    out_dir: Path,
    bits_tx: np.ndarray,
    modulation: str,
    m_order: int,
    sps: int,
    ebn0: float,
    chain: dict,
    channel_mode: str = "awgn",
    use_rx_rrc: bool = True,
) -> dict[str, str]:
    tx_syms = lab2_rrc.map_bits_to_symbols(bits_tx, modulation, m_order=m_order)
    return _render_single_outputs(
        out_dir=out_dir,
        tx_syms=tx_syms,
        rx_syms=chain["rx_syms"],
        rx_filtered=chain["rx_filtered"],
        taps_rx=chain["taps_rx"],
        sample_idxs=chain["sample_idxs"],
        ebn0=float(ebn0),
        ber=float(chain["ber"]),
        sps=sps,
        modulation=modulation,
        m_order=m_order,
        channel_mode=channel_mode,
        use_rx_rrc=use_rx_rrc,
    )


def _run_ber_curve(
    *,
    out_dir: Path,
    bits_tx: np.ndarray,
    tx_signal: np.ndarray,
    modulation: str,
    m_order: int,
    sps: int,
    rolloff: float,
    span: int,
    ebn0_start: float,
    ebn0_end: float,
    ebn0_step: float,
    trials_per_ebn0: int,
    theory_points: int,
    seed: int,
    channel_mode: str = "awgn",
    use_rx_rrc: bool = True,
    timing_offset_ts: float = 0.0,
    cancel_cb=None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    mod = modulation.upper()
    syms_tx = lab2_rrc.map_bits_to_symbols(bits_tx, mod, m_order=m_order)
    ebn0_vals = np.arange(ebn0_start, ebn0_end + 0.1 * ebn0_step, ebn0_step)
    trials = max(1, int(trials_per_ebn0))

    results_ebn0: list[float] = []
    results_ber: list[float] = []
    results_ber_std: list[float] = []
    results_ber_ci95: list[float] = []
    results_ebn0_est_mean: list[float] = []
    results_ebn0_est_std: list[float] = []

    for i, ebn0 in enumerate(ebn0_vals):
        if cancel_cb and cancel_cb():
            raise InterruptedError("Cancelado por el usuario")
        total_errors = 0
        total_bits = 0
        ebn0_estimates = []
        ber_trials = []
        for t in range(trials):
            if cancel_cb and cancel_cb():
                raise InterruptedError("Cancelado por el usuario")
            chain = _run_chain(
                bits_tx=bits_tx,
                tx_signal=tx_signal,
                modulation=mod,
                ebn0=float(ebn0),
                sps=sps,
                rolloff=rolloff,
                span=span,
                seed=seed + (1000 * i) + t + 1,
                m_order=m_order,
                channel_mode=channel_mode,
                use_rx_rrc=use_rx_rrc,
                timing_offset_ts=timing_offset_ts,
            )
            total_errors += int(chain["n_errors"])
            total_bits += int(len(bits_tx))
            ebn0_estimates.append(float(chain["ebn0_est_db"]))
            ber_trials.append(float(chain["ber"]))

        ber = float(total_errors) / max(float(total_bits), 1.0)
        ber_std = float(np.std(ber_trials)) if len(ber_trials) > 1 else 0.0
        ber_ci95 = 1.96 * ber_std / np.sqrt(max(len(ber_trials), 1))
        results_ebn0.append(float(ebn0))
        results_ber.append(ber)
        results_ber_std.append(ber_std)
        results_ber_ci95.append(float(ber_ci95))
        results_ebn0_est_mean.append(float(np.mean(ebn0_estimates)))
        results_ebn0_est_std.append(float(np.std(ebn0_estimates)))
        print(
            f"Eb/N0 = {float(ebn0):5.1f} dB | BER = {ber:.2e} | "
            f"Eb/N0_est={results_ebn0_est_mean[-1]:.2f}±{results_ebn0_est_std[-1]:.2f} dB | "
            f"BER_std={ber_std:.2e} | trials={trials} (TxSyms={len(syms_tx)})"
        )

    n_theory = max(2, int(theory_points))
    # Si hay un solo punto de Eb/N0, abrir una ventana local para poder ver la curva teórica.
    if np.isclose(float(ebn0_end), float(ebn0_start)):
        eb_plot_min = float(ebn0_start) - 1.5
        eb_plot_max = float(ebn0_end) + 1.5
    else:
        eb_plot_min = float(min(ebn0_start, ebn0_end))
        eb_plot_max = float(max(ebn0_start, ebn0_end))
    ebn0_fine = np.linspace(eb_plot_min, eb_plot_max, n_theory)
    ber_theory = theoretical_ber_mpsk(ebn0_fine, modulation=mod, m_order=m_order)
    mod_lbl = _mod_label(mod, m_order=m_order)
    x = np.array(results_ebn0, dtype=np.float64)
    y = np.array(results_ber, dtype=np.float64)
    y_ci = np.array(results_ber_ci95, dtype=np.float64)
    plt.figure()
    plt.semilogy(
        ebn0_fine,
        ber_theory,
        color="black",
        linestyle="-",
        linewidth=2.0,
        zorder=2,
        label=f"Teorica ({mod_lbl}, {n_theory} pts)",
    )
    plt.semilogy(
        x,
        y,
        color="#2563eb",
        linestyle="--",
        linewidth=1.2,
        marker="o",
        markersize=4.5,
        zorder=3,
        label=f"Simulada media ({mod}, N={trials})",
    )
    # Referencia puntual teórica exactamente en los mismos Eb/N0 de simulación.
    y_theo_pts = theoretical_ber_mpsk(x, modulation=mod, m_order=m_order)
    plt.semilogy(
        x,
        y_theo_pts,
        linestyle="none",
        marker="x",
        markersize=6.0,
        markeredgewidth=1.2,
        color="black",
        zorder=4,
        label="Teórica @ puntos simulados",
    )
    if trials > 1:
        y_low = np.maximum(y - y_ci, 1e-8)
        y_high = np.maximum(y + y_ci, 1e-8)
        plt.fill_between(x, y_low, y_high, color="#3b82f6", alpha=0.18, label="IC 95% (Monte Carlo)")
        plt.errorbar(x, y, yerr=y_ci, fmt="none", ecolor="#1d4ed8", alpha=0.6, capsize=3, lw=0.8)
    plt.grid(True, which="both", alpha=0.3)
    plt.xlabel("Eb/N0 [dB]")
    plt.ylabel("BER (Bit Error Rate)")
    plt.title(f"Curva BER vs Eb/N0 - {mod}")
    # Autoescala robusta en Y (log): evita que el punto quede pegado al borde superior.
    y_all = np.concatenate([ber_theory, y, y_theo_pts])
    y_all = y_all[np.isfinite(y_all) & (y_all > 0)]
    if y_all.size:
        ymin_data = float(np.min(y_all))
        ymax_data = float(np.max(y_all))
        # margen de media década hacia abajo y arriba
        y_min_plot = 10 ** np.floor(np.log10(ymin_data) - 0.5)
        y_max_plot = 10 ** np.ceil(np.log10(ymax_data) + 0.5)
        y_min_plot = max(1e-8, y_min_plot)
        y_max_plot = min(1.0, y_max_plot)
        if y_max_plot <= y_min_plot:
            y_max_plot = min(1.0, y_min_plot * 10.0)
        plt.ylim(y_min_plot, y_max_plot)
    # En X, usar rango real del gráfico (expandido en single-point)
    plt.xlim(eb_plot_min, eb_plot_max)
    plt.legend()
    plot_path = out_dir / "ber_curve.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=140)
    plt.close()

    csv_path = out_dir / "ber_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "EbN0_Target_dB",
                "BER_Sim",
                "BER_Theory",
                "BER_Std_MonteCarlo",
                "BER_CI95_MonteCarlo",
                "EbN0_Est_Mean_dB",
                "EbN0_Est_Std_dB",
                "Trials",
            ]
        )
        for e, b, bs, bci, em, es in zip(
            results_ebn0,
            results_ber,
            results_ber_std,
            results_ber_ci95,
            results_ebn0_est_mean,
            results_ebn0_est_std,
        ):
            b_theo = float(theoretical_ber_mpsk([e], modulation=mod, m_order=m_order)[0])
            writer.writerow([e, b, b_theo, bs, bci, em, es, trials])

    return {
        "out_dir": str(out_dir),
        "ber_plot": str(plot_path),
        "ber_csv": str(csv_path),
        "ber_data": list(zip(results_ebn0, results_ber)),
        "ber_std_data": list(zip(results_ber_std, results_ber_ci95)),
        "ebn0_est_data": list(zip(results_ebn0_est_mean, results_ebn0_est_std)),
        "trials_per_ebn0": trials,
        "theory_points": n_theory,
        "modulation": mod,
        "m_order": int(m_order),
        "sps": int(sps),
        "rolloff": float(rolloff),
        "span": int(span),
        "use_rx_rrc": bool(use_rx_rrc),
        "timing_offset_ts": float(timing_offset_ts),
    }


def run_simulation(params: Lab3Params) -> dict:
    print(f"Iniciando simulacion Canal y Rx. Mod={params.modulation}, n_bits={params.n_bits}...")
    mod = params.modulation.upper()
    rng_bits = np.random.default_rng(params.seed)
    bits_tx = rng_bits.integers(0, 2, size=params.n_bits, dtype=np.uint8)
    syms_tx = lab2_rrc.map_bits_to_symbols(bits_tx, mod, m_order=params.m_order)
    taps_tx = lab2_rrc.rrc_taps(params.rolloff, params.sps, params.span)
    tx_signal = lab2_rrc.upsample_and_filter(syms_tx, params.sps, taps_tx)
    res = _run_ber_curve(
        out_dir=Path(params.out_dir),
        bits_tx=bits_tx,
        tx_signal=tx_signal,
        modulation=mod,
        m_order=params.m_order,
        sps=params.sps,
        rolloff=params.rolloff,
        span=params.span,
        ebn0_start=params.ebn0_start,
        ebn0_end=params.ebn0_end,
        ebn0_step=params.ebn0_step,
        trials_per_ebn0=params.trials_per_ebn0,
        theory_points=params.theory_points,
        seed=params.seed,
        channel_mode=params.channel_mode,
        use_rx_rrc=params.use_rx_rrc,
        timing_offset_ts=params.timing_offset_ts,
        cancel_cb=None,
    )
    # Empaquete diagnóstico para el informe (figuras de Rx, MF y constelaciones).
    eb_diag = float((params.ebn0_start + params.ebn0_end) / 2.0)
    chain_diag = _run_chain(
        bits_tx=bits_tx,
        tx_signal=tx_signal,
        modulation=mod,
        ebn0=eb_diag,
        sps=params.sps,
        rolloff=params.rolloff,
        span=params.span,
        seed=params.seed + 99991,
        m_order=params.m_order,
        channel_mode=params.channel_mode,
        use_rx_rrc=params.use_rx_rrc,
        timing_offset_ts=params.timing_offset_ts,
    )
    diag_paths = _generate_diag_outputs_from_chain(
        out_dir=Path(params.out_dir),
        bits_tx=bits_tx,
        modulation=mod,
        m_order=params.m_order,
        sps=params.sps,
        ebn0=eb_diag,
        chain=chain_diag,
        channel_mode=params.channel_mode,
        use_rx_rrc=params.use_rx_rrc,
    )
    res["diag_ebn0_db"] = eb_diag
    res["diag_paths"] = diag_paths
    print(f"Resultados guardados en {params.out_dir}")
    return res


def run_single(params: Lab3Params, bits: np.ndarray | None = None, lab1_meta: dict | None = None) -> dict:
    outp = Path(params.out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    mod = params.modulation.upper()
    if bits is None:
        rng = np.random.default_rng(params.seed)
        bits_tx = rng.integers(0, 2, size=params.n_bits, dtype=np.uint8)
    else:
        bits_tx = np.asarray(bits, dtype=np.uint8).ravel()
    syms_tx = lab2_rrc.map_bits_to_symbols(bits_tx, mod, m_order=params.m_order)
    taps_tx = lab2_rrc.rrc_taps(params.rolloff, params.sps, params.span)
    tx_signal = lab2_rrc.upsample_and_filter(syms_tx, params.sps, taps_tx)
    ebn0_val = float(params.ebn0_start)

    chain = _run_chain(
        bits_tx=bits_tx,
        tx_signal=tx_signal,
        modulation=mod,
        ebn0=ebn0_val,
        sps=params.sps,
        rolloff=params.rolloff,
        span=params.span,
        seed=params.seed + 11,
        m_order=params.m_order,
        channel_mode=params.channel_mode,
        use_rx_rrc=params.use_rx_rrc,
        timing_offset_ts=params.timing_offset_ts,
    )

    fig_paths = _render_single_outputs(
        out_dir=outp,
        tx_syms=syms_tx,
        rx_syms=chain["rx_syms"],
        rx_filtered=chain["rx_filtered"],
        taps_rx=chain["taps_rx"],
        sample_idxs=chain["sample_idxs"],
        ebn0=ebn0_val,
        ber=float(chain["ber"]),
        sps=params.sps,
        modulation=mod,
        m_order=params.m_order,
        channel_mode=params.channel_mode,
        use_rx_rrc=params.use_rx_rrc,
    )
    recovered_paths = _recover_source_from_bits(chain["bits_rx"], lab1_meta, outp)
    fig_paths.update(recovered_paths)
    return {
        "ebn0": ebn0_val,
        "ebn0_target_db": ebn0_val,
        "ebn0_est_db": float(chain["ebn0_est_db"]),
        "snr_est_db": float(chain["snr_est_db"]),
        "ebn0_delta_db": float(chain["ebn0_est_db"]) - ebn0_val,
        "modulation": mod,
        "m_order": int(params.m_order),
        "ber": float(chain["ber"]),
        "n_errors": int(chain["n_errors"]),
        "n_bits": int(len(bits_tx)),
        "paths": fig_paths,
        "ok": True,
    }


def _load_bits_from_lab2_dir(lab2_path: Path, meta: dict) -> tuple[np.ndarray, str]:
    bits_file = None
    for candidate in ("bits_formateo.bin", "bits_from_lab1.bin", "bits_tx.bin", "bits.bin"):
        c = lab2_path / candidate
        if c.exists():
            bits_file = c
            break
    if bits_file is None:
        raise FileNotFoundError(f"No se encontro archivo de bits en {lab2_path}")

    raw = np.fromfile(bits_file, dtype=np.uint8)
    if raw.size == 0:
        raise ValueError(f"Archivo de bits vacio: {bits_file}")

    n_effective = int(meta.get("n_bits_effective") or meta.get("n_bits") or 0)
    uniq = set(np.unique(raw).tolist())
    if uniq.issubset({0, 1}):
        bits = raw.astype(np.uint8)
        if n_effective > 0:
            bits = bits[:n_effective]
        return bits, f"raw_bits:{bits_file.name}"

    # Compatibilidad retroactiva: archivo empaquetado en bytes.
    unpacked = np.unpackbits(raw)
    if n_effective > 0:
        unpacked = unpacked[:n_effective]
    return unpacked.astype(np.uint8), f"packed_bits:{bits_file.name}"


def run_from_file(
    lab2_dir: str,
    ebn0: float,
    out_dir: str = "outputs/lab3_integrated",
    channel_mode: str = "awgn",
    use_rx_rrc: bool = True,
    timing_offset_ts: float = 0.0,
) -> dict:
    lab2_path = Path(lab2_dir)
    params_file = lab2_path / "params.json"
    iq_file = lab2_path / "iq.bin"
    if not iq_file.exists():
        iq_file = lab2_path / "iq_tx.bin"
    if not params_file.exists():
        raise FileNotFoundError(f"No se encontro params.json en {lab2_dir}")
    if not iq_file.exists():
        raise FileNotFoundError(f"No se encontro iq.bin/iq_tx.bin en {lab2_dir}")

    with open(params_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    l2_conf = meta.get("lab2", meta)
    mod = (l2_conf.get("modulation", "QPSK") or "QPSK").upper()
    m_order = int(l2_conf.get("m_order") or (2 if mod == "BPSK" else 4))
    sps = int(l2_conf.get("sps", 8))
    alpha = float(l2_conf.get("rolloff", 0.25))
    span = int(l2_conf.get("span", 8))

    raw_iq = np.fromfile(iq_file, dtype=np.float32)
    if raw_iq.size % 2 != 0:
        raise ValueError("iq.bin invalido: numero impar de muestras float32")
    tx_signal = raw_iq[0::2] + 1j * raw_iq[1::2]
    bits_tx, bits_mode = _load_bits_from_lab2_dir(lab2_path, meta)
    tx_syms = lab2_rrc.map_bits_to_symbols(bits_tx, mod, m_order=m_order)

    chain = _run_chain(
        bits_tx=bits_tx,
        tx_signal=tx_signal,
        modulation=mod,
        ebn0=float(ebn0),
        sps=sps,
        rolloff=alpha,
        span=span,
        seed=7,
        m_order=m_order,
        channel_mode=channel_mode,
        use_rx_rrc=use_rx_rrc,
        timing_offset_ts=timing_offset_ts,
    )

    out_path = Path(out_dir)
    fig_paths = _render_single_outputs(
        out_dir=out_path,
        tx_syms=tx_syms,
        rx_syms=chain["rx_syms"],
        rx_filtered=chain["rx_filtered"],
        taps_rx=chain["taps_rx"],
        sample_idxs=chain["sample_idxs"],
        ebn0=float(ebn0),
        ber=float(chain["ber"]),
        sps=sps,
        modulation=mod,
        m_order=m_order,
        channel_mode=channel_mode,
        use_rx_rrc=use_rx_rrc,
    )
    recovered_paths = _recover_source_from_bits(chain["bits_rx"], meta.get("lab1"), out_path)
    fig_paths.update(recovered_paths)

    return {
        "ebn0": float(ebn0),
        "ebn0_target_db": float(ebn0),
        "ebn0_est_db": float(chain["ebn0_est_db"]),
        "snr_est_db": float(chain["snr_est_db"]),
        "ebn0_delta_db": float(chain["ebn0_est_db"]) - float(ebn0),
        "modulation": mod,
        "m_order": int(m_order),
        "ber": float(chain["ber"]),
        "n_errors": int(chain["n_errors"]),
        "n_bits": int(len(bits_tx)),
        "paths": fig_paths,
        "ok": True,
        "source": "file_integration",
        "channel_mode": channel_mode,
        "use_rx_rrc": bool(use_rx_rrc),
        "timing_offset_ts": float(timing_offset_ts),
        "debug": {
            "tx_10": bits_tx[:10].tolist(),
            "rx_10": chain["bits_rx"][:10].tolist(),
            "len_tx": int(len(bits_tx)),
            "len_rx": int(len(chain["bits_rx"])),
            "n_errors": int(chain["n_errors"]),
            "bits_mode": bits_mode,
        },
    }


def run_simulation_from_file(
    lab2_dir: str,
    out_dir: str,
    ebn0_start: float,
    ebn0_end: float,
    ebn0_step: float,
    trials_per_ebn0: int = 20,
    theory_points: int = 300,
    seed: int = 0,
    channel_mode: str = "awgn",
    use_rx_rrc: bool = True,
    timing_offset_ts: float = 0.0,
    cancel_cb=None,
) -> dict:
    lab2_path = Path(lab2_dir)
    params_file = lab2_path / "params.json"
    iq_file = lab2_path / "iq.bin"
    if not iq_file.exists():
        iq_file = lab2_path / "iq_tx.bin"
    if not params_file.exists():
        raise FileNotFoundError(f"No se encontro params.json en {lab2_dir}")
    if not iq_file.exists():
        raise FileNotFoundError(f"No se encontro iq.bin/iq_tx.bin en {lab2_dir}")

    with open(params_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    l2_conf = meta.get("lab2", meta)
    mod = (l2_conf.get("modulation", "QPSK") or "QPSK").upper()
    m_order = int(l2_conf.get("m_order") or (2 if mod == "BPSK" else 4))
    sps = int(l2_conf.get("sps", 8))
    alpha = float(l2_conf.get("rolloff", 0.25))
    span = int(l2_conf.get("span", 8))

    raw_iq = np.fromfile(iq_file, dtype=np.float32)
    if raw_iq.size % 2 != 0:
        raise ValueError("iq.bin invalido: numero impar de muestras float32")
    tx_signal = raw_iq[0::2] + 1j * raw_iq[1::2]
    bits_tx, bits_mode = _load_bits_from_lab2_dir(lab2_path, meta)

    res = _run_ber_curve(
        out_dir=Path(out_dir),
        bits_tx=bits_tx,
        tx_signal=tx_signal,
        modulation=mod,
        m_order=m_order,
        sps=sps,
        rolloff=alpha,
        span=span,
        ebn0_start=ebn0_start,
        ebn0_end=ebn0_end,
        ebn0_step=ebn0_step,
        trials_per_ebn0=trials_per_ebn0,
        theory_points=theory_points,
        seed=seed,
        channel_mode=channel_mode,
        use_rx_rrc=use_rx_rrc,
        timing_offset_ts=timing_offset_ts,
        cancel_cb=cancel_cb,
    )
    eb_diag = float((ebn0_start + ebn0_end) / 2.0)
    chain_diag = _run_chain(
        bits_tx=bits_tx,
        tx_signal=tx_signal,
        modulation=mod,
        ebn0=eb_diag,
        sps=sps,
        rolloff=alpha,
        span=span,
        seed=seed + 99991,
        m_order=m_order,
        channel_mode=channel_mode,
        use_rx_rrc=use_rx_rrc,
        timing_offset_ts=timing_offset_ts,
    )
    diag_paths = _generate_diag_outputs_from_chain(
        out_dir=Path(out_dir),
        bits_tx=bits_tx,
        modulation=mod,
        m_order=m_order,
        sps=sps,
        ebn0=eb_diag,
        chain=chain_diag,
        channel_mode=channel_mode,
        use_rx_rrc=use_rx_rrc,
    )
    res["diag_ebn0_db"] = eb_diag
    res["diag_paths"] = diag_paths
    res["source"] = "lab2_file_chain"
    res["lab2_path"] = str(lab2_path)
    res["bits_mode"] = bits_mode
    res["m_order"] = int(m_order)
    res["channel_mode"] = channel_mode
    res["use_rx_rrc"] = bool(use_rx_rrc)
    res["timing_offset_ts"] = float(timing_offset_ts)
    recovered_paths = _recover_source_from_bits(chain_diag["bits_rx"], meta.get("lab1"), Path(out_dir))
    if recovered_paths:
        res["recovered_paths"] = recovered_paths
    return res


def main():
    ap = argparse.ArgumentParser("Canal y Rx - Demodulacion Digital")
    ap.add_argument("--out", default="outputs/lab3")
    ap.add_argument("--n_bits", type=int, default=100000)
    ap.add_argument("--mod", default="QPSK", choices=["BPSK", "QPSK", "MPSK"])
    ap.add_argument("--M", type=int, default=4, help="Orden M para MPSK (potencia de 2)")
    ap.add_argument("--eb_start", type=float, default=0.0)
    ap.add_argument("--eb_end", type=float, default=10.0)
    ap.add_argument("--eb_step", type=float, default=1.0)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--theory_points", type=int, default=300)
    ap.add_argument("--sps", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--span", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--disable_rx_rrc", action="store_true")
    ap.add_argument("--timing_offset_ts", type=float, default=0.0)
    args = ap.parse_args()

    p = Lab3Params(
        out_dir=args.out,
        n_bits=args.n_bits,
        modulation=args.mod,
        m_order=args.M,
        sps=args.sps,
        rolloff=args.alpha,
        span=args.span,
        ebn0_start=args.eb_start,
        ebn0_end=args.eb_end,
        ebn0_step=args.eb_step,
        trials_per_ebn0=args.trials,
        theory_points=args.theory_points,
        seed=args.seed,
        use_rx_rrc=not args.disable_rx_rrc,
        timing_offset_ts=args.timing_offset_ts,
    )
    run_simulation(p)


if __name__ == "__main__":
    main()
