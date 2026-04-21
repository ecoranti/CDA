import argparse
import os
from .audio_utils import (
    load_wav_mono,
    uniform_quantize,
    a_law_quantize,
    save_signal_plot,
    save_hist_amplitudes,
    save_signal_quantized_compare,
    save_quantizer_characteristic,
    save_quantizer_low_level_compare,
    save_quantization_error_compare,
    plot_hist_bits,
    plot_hist_bits_compare,
    plot_entropy_evolution,
)
from .bits_utils import ints_to_bits, bits_to_bytes, bits_entropy_stats
from .scrambling import scramble
from .report import write_markdown, save_metrics_csv
from . import sqnr_eval


def ensure_dirs(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    figdir = os.path.join(out_dir, "figures")
    os.makedirs(figdir, exist_ok=True)
    return figdir


def process_audio(
    audio_path: str,
    figdir: str,
    fs_target: int = 16000,
    n_bits: int = 8,
    quantizer: str = "alaw",
    lfsr_seed: int = 0b1010110011,
    lfsr_taps=(9, 6),
    lfsr_bitwidth: int = 10,
    hist_bins: int = 50,
    entropy_step: int = 10000,
):
    # 1) Cargar audio, convertir a mono y normalizar
    x, fs = load_wav_mono(audio_path, target_fs=fs_target)

    # 2) Gráficos previos
    save_signal_plot(x, fs, "Fuente A: señal 'voz' (tiempo)", os.path.join(figdir, "A_signal_time.png"))
    save_hist_amplitudes(x, "Fuente A: histograma de amplitudes", os.path.join(figdir, "A_signal_hist.png"), bins=hist_bins)

    rows_local = []
    xhat_uniform = sqnr_eval.eval_uniform(x, bits=n_bits)
    xhat_alaw = sqnr_eval.eval_alaw(x, A=87.6, bits=n_bits)

    def run_pipeline_for_quantizer(q_name: str, q_type: str):
        # 3) Cuantificación -> símbolos (0..2^n_bits-1) -> bits
        if q_type == "alaw":
            qA, LA = a_law_quantize(x, bits=n_bits, A=87.6, xmin=-1, xmax=1)
            prefix = "alaw"
            xhat = xhat_alaw
        else:
            qA, LA = uniform_quantize(x, bits=n_bits, xmin=-1, xmax=1)
            prefix = "uniform"
            xhat = xhat_uniform

        bitsA = ints_to_bits(qA, n_bits)

        # nombres de archivo: el cuantizador principal (alaw) usa nombres estándar para report.py
        if q_type == "alaw":
            before_png = os.path.join(figdir, "A_bits_hist_before.png")
            scr_png    = os.path.join(figdir, "A_bits_hist_scrambled.png")
        else:
            before_png = os.path.join(figdir, f"A_bits_hist_before_{prefix}.png")
            scr_png    = os.path.join(figdir, f"A_bits_hist_scrambled_{prefix}.png")

        # Histogramas individuales (probabilidad en eje Y)
        plot_hist_bits(bitsA, f"Fuente A ({q_name}): histograma de bits (antes)", before_png, as_probability=True)

        # 4) Scrambling (PRBS LFSR)
        bitsA_scr = scramble(bitsA, seed=lfsr_seed, taps=lfsr_taps, bitwidth=lfsr_bitwidth)
        plot_hist_bits(bitsA_scr, f"Fuente A ({q_name}): histograma de bits (scrambling)", scr_png, as_probability=True)

        # 5) Comparativa y evolución de entropía
        plot_hist_bits_compare(
            bitsA, "Antes", bitsA_scr, "Scrambling",
            "Fuente A: histogramas comparativos (probabilidad)",
            os.path.join(figdir, f"A_bits_hist_compare_{prefix}.png"),
            as_probability=True,
        )
        # Entropía binaria para original y scrambler
        plot_entropy_evolution(
            [bitsA, bitsA_scr],
            [f"{q_name} original", f"{q_name} + scrambler"],
            os.path.join(figdir, f"A_entropy_evolution_{prefix}.png"),
            step=entropy_step
        )
        save_signal_quantized_compare(
            x, xhat, fs,
            f"Fuente A ({q_name}): señal original vs reconstruida",
            os.path.join(figdir, f"A_signal_quantized_compare_{prefix}.png"),
        )
        save_quantizer_characteristic(
            x, xhat,
            f"Fuente A ({q_name}): característica entrada/salida",
            os.path.join(figdir, f"A_quantizer_characteristic_{prefix}.png"),
        )
        # 6) Métricas
        p0A, p1A, HA, varA = bits_entropy_stats(bitsA)
        p0A_s, p1A_s, HA_s, varA_s = bits_entropy_stats(bitsA_scr)

        rows_local.extend([
            (f"Fuente A [{q_name}] – Antes", p0A, p1A, HA, varA, float("nan")),
            (f"Fuente A [{q_name}] – Scrambling", p0A_s, p1A_s, HA_s, varA_s, float("nan")),
        ])

    if quantizer == "alaw":
        run_pipeline_for_quantizer("A-law", q_type="alaw")
    elif quantizer == "uniform":
        run_pipeline_for_quantizer("Uniforme", q_type="uniform")
    else:  # "both" — A-law como principal (nombres estándar), uniforme como comparación
        run_pipeline_for_quantizer("Uniforme", q_type="uniform")
        run_pipeline_for_quantizer("A-law", q_type="alaw")

    save_quantizer_low_level_compare(
        x, xhat_uniform, xhat_alaw, fs,
        "Fuente A: comparación en tramo de baja amplitud",
        os.path.join(figdir, "A_quantizer_low_level_compare.png"),
        data_csv=os.path.join(figdir, "A_quantizer_low_level_compare_data.csv"),
    )
    save_quantization_error_compare(
        x, xhat_uniform, xhat_alaw, fs,
        "Fuente A: error de cuantización",
        os.path.join(figdir, "A_quantization_error_compare.png"),
    )

    return tuple(rows_local)


def process_text(
    text_path: str,
    figdir: str,
    lfsr_seed: int = 0b1010110011,
    lfsr_taps=(9, 6),
    lfsr_bitwidth: int = 10,
    entropy_step: int = 500,
):
    # 1) Leer texto UTF-8 -> bytes -> bits
    txt = open(text_path, "r", encoding="utf-8").read()
    b = txt.encode("utf-8")
    bytes_list = list(b)

    bitsB = []
    for val in bytes_list:
        for bit in range(7, -1, -1):
            bitsB.append((val >> bit) & 1)

    # 2) Histograma de bits antes
    before_png = os.path.join(figdir, "B_bits_hist_before.png")
    plot_hist_bits(bitsB, "Fuente B: histograma de bits (antes)", before_png, as_probability=True)

    # 3) Scrambling
    bitsB_scr = scramble(bitsB, seed=lfsr_seed, taps=lfsr_taps, bitwidth=lfsr_bitwidth)
    scr_png = os.path.join(figdir, "B_bits_hist_scrambled.png")
    plot_hist_bits(bitsB_scr, "Fuente B: histograma de bits (scrambling)", scr_png, as_probability=True)

    # 4) Comparativa y evolución de entropía
    plot_hist_bits_compare(
        bitsB, "Antes", bitsB_scr, "Scrambling",
        "Fuente B: histogramas comparativos (probabilidad)",
        os.path.join(figdir, "B_bits_hist_compare.png"),
        as_probability=True
    )
    # Entropía binaria para original y scrambler
    plot_entropy_evolution(
        [bitsB, bitsB_scr],
        ["Texto original", "Texto + scrambler"],
        os.path.join(figdir, "B_entropy_evolution.png"),
        step=entropy_step
    )

    # 5) Métricas
    p0B, p1B, HB, varB = bits_entropy_stats(bitsB)
    p0B_s, p1B_s, HB_s, varB_s = bits_entropy_stats(bitsB_scr)

    return (
        ("Fuente B – Antes", p0B, p1B, HB, varB, float("nan")),
        ("Fuente B – Scrambling", p0B_s, p1B_s, HB_s, varB_s, float("nan")),
    )


def main():
    ap = argparse.ArgumentParser(description="Formateo – Cuantización y Scrambling")
    ap.add_argument("--audio", required=True, help="Ruta a WAV PCM (voz)")
    ap.add_argument("--text", required=True, help="Ruta a archivo de texto (UTF-8)")
    ap.add_argument("--out", default="outputs", help="Directorio de salida")
    ap.add_argument("--source", choices=["audio", "text"], default="audio",
                    help="Fuente a procesar en esta corrida. Default: audio")
    ap.add_argument("--fs", type=int, default=16000, help="Frecuencia de muestreo objetivo (Hz)")
    ap.add_argument("--n_bits", type=int, default=8, help="Bits de cuantificación (1-24)")
    ap.add_argument("--quantizer", choices=["uniform", "alaw", "both"], default="alaw",
                    help="Tipo de cuantizador para audio: uniforme, alaw (G.711 Argentina) o ambos. Default: alaw")
    ap.add_argument("--lfsr_seed", type=lambda x: int(x, 0), default=0b1010110011, help="Semilla del LFSR (base 0b o 0x aceptada)")
    ap.add_argument("--lfsr_taps", type=str, default="9,6", help="Taps del LFSR como lista separada por comas, ej: 9,6")
    ap.add_argument("--lfsr_bitwidth", type=int, default=10, help="Bitwidth del LFSR (1-32)")
    ap.add_argument("--hist_bins", type=int, default=50, help="Bins para histograma de amplitudes de audio")
    ap.add_argument("--entropy_step_a", type=int, default=10000, help="Paso para evolución de entropía (audio)")
    ap.add_argument("--entropy_step_b", type=int, default=500, help="Paso para evolución de entropía (texto)")
    args = ap.parse_args()

    figdir = ensure_dirs(args.out)

    rows = []
    # Parse taps
    try:
        taps = tuple(int(t.strip()) for t in str(args.lfsr_taps).split(",") if t.strip() != "")
    except Exception:
        taps = (9, 6)
    if args.source == "audio":
        rows.extend(process_audio(
            args.audio, figdir,
            fs_target=args.fs,
            n_bits=args.n_bits,
            quantizer=args.quantizer,
            lfsr_seed=args.lfsr_seed,
            lfsr_taps=taps,
            lfsr_bitwidth=args.lfsr_bitwidth,
            hist_bins=args.hist_bins,
            entropy_step=args.entropy_step_a,
        ))
    else:
        rows.extend(process_text(
            args.text, figdir,
            lfsr_seed=args.lfsr_seed,
            lfsr_taps=taps,
            lfsr_bitwidth=args.lfsr_bitwidth,
            entropy_step=args.entropy_step_b,
        ))

    # Evaluación SQNR/MSE automática para la Fuente A (audio)
    if args.source == "audio":
        try:
            x, _ = load_wav_mono(args.audio, target_fs=args.fs)
            xhat_uni = sqnr_eval.eval_uniform(x, bits=args.n_bits)
            mse_uni = sqnr_eval.mse(x, xhat_uni)
            sqnr_uni = sqnr_eval.sqnr_db(x, xhat_uni)

            # A-law (G.711 estándar Argentina) con A=87.6
            xhat_al = sqnr_eval.eval_alaw(x, A=87.6, bits=args.n_bits)
            mse_al = sqnr_eval.mse(x, xhat_al)
            sqnr_al = sqnr_eval.sqnr_db(x, xhat_al)

            import matplotlib.pyplot as plt
            labels = ["Uniforme", "A-law (87.6)"]
            sqnrs_plot = [sqnr_uni, sqnr_al]
            mses_plot  = [mse_uni, mse_al]
            # SQNR plot
            plt.figure()
            plt.bar(range(len(sqnrs_plot)), sqnrs_plot, color=["#64748b", "#2563eb"])
            plt.xticks(range(len(labels)), labels)
            plt.ylabel("SQNR (dB)")
            plt.title(f"Comparación de SQNR ({args.n_bits} bits)")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, "sqnr_comparacion.png"), dpi=140)
            plt.close()

            # MSE plot
            plt.figure()
            plt.bar(range(len(mses_plot)), mses_plot, color=["#64748b", "#2563eb"])
            plt.xticks(range(len(labels)), labels)
            plt.ylabel("MSE")
            plt.title(f"Comparación de MSE ({args.n_bits} bits)")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, "mse_comparacion.png"), dpi=140)
            plt.close()

            # CSV resumen
            import pandas as pd
            rows_sqnr = list(zip(labels, sqnrs_plot, mses_plot))
            df_sqnr = pd.DataFrame(rows_sqnr, columns=["Cuantizador", "SQNR (dB)", "MSE"])
            df_sqnr.to_csv(os.path.join(args.out, "sqnr_mse_resumen.csv"), index=False)
        except Exception as e:
            print(f"[WARN] No se pudo calcular SQNR/MSE: {e}")

    # Guardar CSV + Informe
    save_metrics_csv(args.out, rows)
    write_markdown(args.out)
    try:
        from .report import write_pdf as _write_pdf
        _write_pdf(args.out)
    except Exception as _e:
        print(f"[WARN] PDF no generado: {_e}")

    print(f"Listo. Salidas en: {args.out}")


if __name__ == "__main__":
    main()
