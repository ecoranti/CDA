from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

from . import lab2_rrc

matplotlib.use("Agg")


@dataclass
class Lab3Params:
    out_dir: str
    n_bits: int = 10000
    modulation: str = "QPSK"
    sps: int = 8
    rolloff: float = 0.25
    span: int = 8
    ebn0_start: float = 0.0
    ebn0_end: float = 12.0
    ebn0_step: float = 2.0
    trials_per_ebn0: int = 20
    seed: int = 0


def _bps(modulation: str) -> int:
    mod = (modulation or "QPSK").upper()
    if mod == "BPSK":
        return 1
    if mod == "QPSK":
        return 2
    raise ValueError(f"Modulación no soportada: {modulation}")


def _n_symbols_from_bits(n_bits: int, modulation: str) -> int:
    bps = _bps(modulation)
    return (int(n_bits) + bps - 1) // bps


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


def downsample(filtered_signal: np.ndarray, sps: int, delay_samples: int, n_symbols: int) -> np.ndarray:
    idxs = np.arange(delay_samples, len(filtered_signal), sps)
    idxs = idxs[:n_symbols]
    return filtered_signal[idxs]


def demap_qpsk(samples: np.ndarray) -> np.ndarray:
    bits_0 = np.where(samples.real > 0, 0, 1)
    bits_1 = np.where(samples.imag > 0, 0, 1)
    out = np.empty(2 * len(samples), dtype=np.uint8)
    out[0::2] = bits_0
    out[1::2] = bits_1
    return out


def demap_bpsk(samples: np.ndarray) -> np.ndarray:
    return np.where(samples.real > 0, 1, 0).astype(np.uint8)


def calculate_ber(tx_bits: np.ndarray, rx_bits: np.ndarray) -> tuple[float, int]:
    L = min(len(tx_bits), len(rx_bits))
    if L == 0:
        return 0.0, 0
    tx = tx_bits[:L]
    rx = rx_bits[:L]
    n_errors = int(np.sum(tx != rx))
    return float(n_errors) / float(L), n_errors


def theoretical_ber_bpsk_qpsk(ebn0_db_arr: np.ndarray | list[float]) -> np.ndarray:
    ebn0_lin = 10 ** (np.array(ebn0_db_arr, dtype=np.float64) / 10.0)
    return 0.5 * erfc(np.sqrt(ebn0_lin))


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


def _plot_tx_rx_constellations(tx_syms: np.ndarray, rx_syms: np.ndarray, out_path: Path, ebn0: float) -> None:
    all_vals = np.concatenate([tx_syms.real, tx_syms.imag, rx_syms.real, rx_syms.imag])
    m = float(np.max(np.abs(all_vals))) if all_vals.size else 1.0
    lim = max(1.0, 1.15 * m)
    plt.figure(figsize=(8.5, 3.8))
    plt.subplot(1, 2, 1)
    plt.scatter(tx_syms.real, tx_syms.imag, s=8, alpha=0.6)
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
    plt.scatter(rx_syms.real, rx_syms.imag, s=8, alpha=0.6)
    plt.axhline(0, color="gray", lw=0.6)
    plt.axvline(0, color="gray", lw=0.6)
    plt.grid(True, alpha=0.25)
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(f"Constelacion Rx (Eb/N0={ebn0:.1f} dB)")
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _plot_rx_decision(samples: np.ndarray, modulation: str, out_path: Path, max_sym: int = 220) -> None:
    seg = samples[:max_sym]
    if seg.size == 0:
        return
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
    plt.suptitle("Salida del filtro acoplado y decision ML/MAP")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _plot_ber_point(ber: float, ebn0_val: float, out_path: Path) -> None:
    eb_min, eb_max = -2.0, max(12.0, ebn0_val + 2.0)
    eb_arr = np.linspace(eb_min, eb_max, 200)
    ber_theo = theoretical_ber_bpsk_qpsk(eb_arr)
    plt.figure()
    plt.semilogy(eb_arr, ber_theo, "k-", label="Teorica (BPSK/QPSK)")
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
) -> dict:
    mod = modulation.upper()
    rng = np.random.default_rng(seed)
    bps = _bps(mod)
    rx_noisy, noise = add_noise_with_noise(tx_signal, ebn0, sps, bps, rng=rng)
    snr_est_db, ebn0_est_db = estimate_ebn0_db_from_signal_noise(tx_signal, noise, sps, bps)
    rx_filtered, taps_rx = matched_filter(rx_noisy, sps, rolloff, span)
    delay_rx = (len(taps_rx) - 1) // 2
    n_syms_expected = _n_symbols_from_bits(len(bits_tx), mod)
    rx_syms = downsample(rx_filtered, sps, delay_rx, n_syms_expected)
    if mod == "QPSK":
        bits_rx = demap_qpsk(rx_syms)
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
    }


def _render_single_outputs(
    out_dir: Path,
    tx_syms: np.ndarray,
    rx_syms: np.ndarray,
    rx_filtered: np.ndarray,
    taps_rx: np.ndarray,
    ebn0: float,
    ber: float,
    sps: int,
    modulation: str,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    p = out_dir
    p.mkdir(parents=True, exist_ok=True)

    lab2_rrc.plot_time_iq(rx_filtered, sps, str(p / "rx_time.png"), nsamples=800)
    paths["rx_time_png"] = str(p / "rx_time.png")

    lab2_rrc.plot_eye(rx_filtered, sps, str(p / "rx_eye.png"))
    paths["rx_eye_png"] = str(p / "rx_eye.png")

    lab2_rrc.plot_constellation(rx_syms, 1, str(p / "rx_constellation.png"))
    paths["rx_constellation_png"] = str(p / "rx_constellation.png")

    _plot_ber_point(ber, ebn0, p / "ber_point.png")
    paths["ber_point_png"] = str(p / "ber_point.png")

    _plot_tx_rx_constellations(tx_syms, rx_syms, p / "tx_rx_constellations.png", ebn0)
    paths["tx_rx_constellations_png"] = str(p / "tx_rx_constellations.png")

    _plot_mf_impulse_and_freq(taps_rx, sps, p / "mf_impulse.png", p / "mf_freq.png")
    paths["mf_impulse_png"] = str(p / "mf_impulse.png")
    paths["mf_freq_png"] = str(p / "mf_freq.png")

    _plot_rx_decision(rx_syms, modulation, p / "rx_decision.png")
    paths["rx_decision_png"] = str(p / "rx_decision.png")

    return paths


def _generate_diag_outputs_from_chain(
    out_dir: Path,
    bits_tx: np.ndarray,
    modulation: str,
    sps: int,
    ebn0: float,
    chain: dict,
) -> dict[str, str]:
    tx_syms = lab2_rrc.map_bits_to_symbols(bits_tx, modulation)
    return _render_single_outputs(
        out_dir=out_dir,
        tx_syms=tx_syms,
        rx_syms=chain["rx_syms"],
        rx_filtered=chain["rx_filtered"],
        taps_rx=chain["taps_rx"],
        ebn0=float(ebn0),
        ber=float(chain["ber"]),
        sps=sps,
        modulation=modulation,
    )


def _run_ber_curve(
    *,
    out_dir: Path,
    bits_tx: np.ndarray,
    tx_signal: np.ndarray,
    modulation: str,
    sps: int,
    rolloff: float,
    span: int,
    ebn0_start: float,
    ebn0_end: float,
    ebn0_step: float,
    trials_per_ebn0: int,
    seed: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    mod = modulation.upper()
    syms_tx = lab2_rrc.map_bits_to_symbols(bits_tx, mod)
    ebn0_vals = np.arange(ebn0_start, ebn0_end + 0.1 * ebn0_step, ebn0_step)
    trials = max(1, int(trials_per_ebn0))

    results_ebn0: list[float] = []
    results_ber: list[float] = []
    results_ber_std: list[float] = []
    results_ber_ci95: list[float] = []
    results_ebn0_est_mean: list[float] = []
    results_ebn0_est_std: list[float] = []

    for i, ebn0 in enumerate(ebn0_vals):
        total_errors = 0
        total_bits = 0
        ebn0_estimates = []
        ber_trials = []
        for t in range(trials):
            chain = _run_chain(
                bits_tx=bits_tx,
                tx_signal=tx_signal,
                modulation=mod,
                ebn0=float(ebn0),
                sps=sps,
                rolloff=rolloff,
                span=span,
                seed=seed + (1000 * i) + t + 1,
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

    ebn0_fine = np.linspace(ebn0_start, ebn0_end, 300)
    ber_theory = theoretical_ber_bpsk_qpsk(ebn0_fine)
    x = np.array(results_ebn0, dtype=np.float64)
    y = np.array(results_ber, dtype=np.float64)
    y_ci = np.array(results_ber_ci95, dtype=np.float64)
    plt.figure()
    plt.semilogy(ebn0_fine, ber_theory, "k-", label="Teorica (BPSK/QPSK)")
    plt.semilogy(x, y, "bo--", label=f"Simulada media ({mod}, N={trials})")
    if trials > 1:
        y_low = np.maximum(y - y_ci, 1e-8)
        y_high = np.maximum(y + y_ci, 1e-8)
        plt.fill_between(x, y_low, y_high, color="#3b82f6", alpha=0.18, label="IC 95% (Monte Carlo)")
        plt.errorbar(x, y, yerr=y_ci, fmt="none", ecolor="#1d4ed8", alpha=0.6, capsize=3, lw=0.8)
    plt.grid(True, which="both", alpha=0.3)
    plt.xlabel("Eb/N0 [dB]")
    plt.ylabel("BER (Bit Error Rate)")
    plt.title(f"Curva BER vs Eb/N0 - {mod}")
    plt.legend()
    plt.ylim(bottom=1e-6)
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
            writer.writerow([e, b, float(theoretical_ber_bpsk_qpsk([e])[0]), bs, bci, em, es, trials])

    return {
        "out_dir": str(out_dir),
        "ber_plot": str(plot_path),
        "ber_csv": str(csv_path),
        "ber_data": list(zip(results_ebn0, results_ber)),
        "ber_std_data": list(zip(results_ber_std, results_ber_ci95)),
        "ebn0_est_data": list(zip(results_ebn0_est_mean, results_ebn0_est_std)),
        "trials_per_ebn0": trials,
        "modulation": mod,
    }


def run_simulation(params: Lab3Params) -> dict:
    print(f"Iniciando simulacion Lab3. Mod={params.modulation}, n_bits={params.n_bits}...")
    mod = params.modulation.upper()
    rng_bits = np.random.default_rng(params.seed)
    bits_tx = rng_bits.integers(0, 2, size=params.n_bits, dtype=np.uint8)
    syms_tx = lab2_rrc.map_bits_to_symbols(bits_tx, mod)
    taps_tx = lab2_rrc.rrc_taps(params.rolloff, params.sps, params.span)
    tx_signal = lab2_rrc.upsample_and_filter(syms_tx, params.sps, taps_tx)
    res = _run_ber_curve(
        out_dir=Path(params.out_dir),
        bits_tx=bits_tx,
        tx_signal=tx_signal,
        modulation=mod,
        sps=params.sps,
        rolloff=params.rolloff,
        span=params.span,
        ebn0_start=params.ebn0_start,
        ebn0_end=params.ebn0_end,
        ebn0_step=params.ebn0_step,
        trials_per_ebn0=params.trials_per_ebn0,
        seed=params.seed,
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
    )
    diag_paths = _generate_diag_outputs_from_chain(
        out_dir=Path(params.out_dir),
        bits_tx=bits_tx,
        modulation=mod,
        sps=params.sps,
        ebn0=eb_diag,
        chain=chain_diag,
    )
    res["diag_ebn0_db"] = eb_diag
    res["diag_paths"] = diag_paths
    print(f"Resultados guardados en {params.out_dir}")
    return res


def run_single(params: Lab3Params, bits: np.ndarray | None = None) -> dict:
    outp = Path(params.out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    mod = params.modulation.upper()
    if bits is None:
        rng = np.random.default_rng(params.seed)
        bits_tx = rng.integers(0, 2, size=params.n_bits, dtype=np.uint8)
    else:
        bits_tx = np.asarray(bits, dtype=np.uint8).ravel()
    syms_tx = lab2_rrc.map_bits_to_symbols(bits_tx, mod)
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
    )

    fig_paths = _render_single_outputs(
        out_dir=outp,
        tx_syms=syms_tx,
        rx_syms=chain["rx_syms"],
        rx_filtered=chain["rx_filtered"],
        taps_rx=chain["taps_rx"],
        ebn0=ebn0_val,
        ber=float(chain["ber"]),
        sps=params.sps,
        modulation=mod,
    )
    return {
        "ebn0": ebn0_val,
        "ebn0_target_db": ebn0_val,
        "ebn0_est_db": float(chain["ebn0_est_db"]),
        "snr_est_db": float(chain["snr_est_db"]),
        "ebn0_delta_db": float(chain["ebn0_est_db"]) - ebn0_val,
        "ber": float(chain["ber"]),
        "n_errors": int(chain["n_errors"]),
        "n_bits": int(len(bits_tx)),
        "paths": fig_paths,
        "ok": True,
    }


def _load_bits_from_lab2_dir(lab2_path: Path, meta: dict) -> tuple[np.ndarray, str]:
    bits_file = None
    for candidate in ("bits_from_lab1.bin", "bits.bin"):
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


def run_from_file(lab2_dir: str, ebn0: float, out_dir: str = "outputs/lab3_integrated") -> dict:
    lab2_path = Path(lab2_dir)
    params_file = lab2_path / "params.json"
    iq_file = lab2_path / "iq.bin"
    if not params_file.exists():
        raise FileNotFoundError(f"No se encontro params.json en {lab2_dir}")
    if not iq_file.exists():
        raise FileNotFoundError(f"No se encontro iq.bin en {lab2_dir}")

    with open(params_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    l2_conf = meta.get("lab2", meta)
    mod = (l2_conf.get("modulation", "QPSK") or "QPSK").upper()
    sps = int(l2_conf.get("sps", 8))
    alpha = float(l2_conf.get("rolloff", 0.25))
    span = int(l2_conf.get("span", 8))

    raw_iq = np.fromfile(iq_file, dtype=np.float32)
    if raw_iq.size % 2 != 0:
        raise ValueError("iq.bin invalido: numero impar de muestras float32")
    tx_signal = raw_iq[0::2] + 1j * raw_iq[1::2]
    bits_tx, bits_mode = _load_bits_from_lab2_dir(lab2_path, meta)
    tx_syms = lab2_rrc.map_bits_to_symbols(bits_tx, mod)

    chain = _run_chain(
        bits_tx=bits_tx,
        tx_signal=tx_signal,
        modulation=mod,
        ebn0=float(ebn0),
        sps=sps,
        rolloff=alpha,
        span=span,
        seed=7,
    )

    out_path = Path(out_dir)
    fig_paths = _render_single_outputs(
        out_dir=out_path,
        tx_syms=tx_syms,
        rx_syms=chain["rx_syms"],
        rx_filtered=chain["rx_filtered"],
        taps_rx=chain["taps_rx"],
        ebn0=float(ebn0),
        ber=float(chain["ber"]),
        sps=sps,
        modulation=mod,
    )

    return {
        "ebn0": float(ebn0),
        "ebn0_target_db": float(ebn0),
        "ebn0_est_db": float(chain["ebn0_est_db"]),
        "snr_est_db": float(chain["snr_est_db"]),
        "ebn0_delta_db": float(chain["ebn0_est_db"]) - float(ebn0),
        "ber": float(chain["ber"]),
        "n_errors": int(chain["n_errors"]),
        "n_bits": int(len(bits_tx)),
        "paths": fig_paths,
        "ok": True,
        "source": "file_integration",
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
    seed: int = 0,
) -> dict:
    lab2_path = Path(lab2_dir)
    params_file = lab2_path / "params.json"
    iq_file = lab2_path / "iq.bin"
    if not params_file.exists():
        raise FileNotFoundError(f"No se encontro params.json en {lab2_dir}")
    if not iq_file.exists():
        raise FileNotFoundError(f"No se encontro iq.bin en {lab2_dir}")

    with open(params_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    l2_conf = meta.get("lab2", meta)
    mod = (l2_conf.get("modulation", "QPSK") or "QPSK").upper()
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
        sps=sps,
        rolloff=alpha,
        span=span,
        ebn0_start=ebn0_start,
        ebn0_end=ebn0_end,
        ebn0_step=ebn0_step,
        trials_per_ebn0=trials_per_ebn0,
        seed=seed,
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
    )
    diag_paths = _generate_diag_outputs_from_chain(
        out_dir=Path(out_dir),
        bits_tx=bits_tx,
        modulation=mod,
        sps=sps,
        ebn0=eb_diag,
        chain=chain_diag,
    )
    res["diag_ebn0_db"] = eb_diag
    res["diag_paths"] = diag_paths
    res["source"] = "lab2_file_chain"
    res["lab2_path"] = str(lab2_path)
    res["bits_mode"] = bits_mode
    return res


def main():
    ap = argparse.ArgumentParser("Lab3 - Demodulacion Digital")
    ap.add_argument("--out", default="outputs/lab3")
    ap.add_argument("--n_bits", type=int, default=100000)
    ap.add_argument("--mod", default="QPSK", choices=["BPSK", "QPSK"])
    ap.add_argument("--eb_start", type=float, default=0.0)
    ap.add_argument("--eb_end", type=float, default=10.0)
    ap.add_argument("--eb_step", type=float, default=1.0)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--sps", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--span", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    p = Lab3Params(
        out_dir=args.out,
        n_bits=args.n_bits,
        modulation=args.mod,
        sps=args.sps,
        rolloff=args.alpha,
        span=args.span,
        ebn0_start=args.eb_start,
        ebn0_end=args.eb_end,
        ebn0_step=args.eb_step,
        trials_per_ebn0=args.trials,
        seed=args.seed,
    )
    run_simulation(p)


if __name__ == "__main__":
    main()
