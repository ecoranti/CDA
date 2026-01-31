import argparse
import os
from .audio_utils import (
    load_wav_mono,
    uniform_quantize,
    mu_law_quantize,
    save_signal_plot,
    save_hist_amplitudes,
    plot_hist_bits,
    plot_hist_bits_compare,
    plot_entropy_evolution,
)
from .bits_utils import ints_to_bits, bits_to_bytes, bits_entropy_stats
from .scrambling import scramble
from .huffman import encode
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
    quantizer: str = "mulaw",
    mu_val: int = 255,
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
    save_hist_amplitudes(x, "Fuente A: histograma de amplitudes (antes)", os.path.join(figdir, "A_signal_hist.png"), bins=hist_bins)

    rows_local = []

    def run_pipeline_for_quantizer(q_name: str, use_mulaw: bool):
        # 3) Cuantificación -> símbolos (0..2^n_bits-1) -> bits
        if use_mulaw:
            qA, LA = mu_law_quantize(x, bits=n_bits, mu=mu_val, xmin=-1, xmax=1)
            prefix = "mulaw"
        else:
            qA, LA = uniform_quantize(x, bits=n_bits, xmin=-1, xmax=1)
            prefix = "uniform"

        bitsA = ints_to_bits(qA, n_bits)

        # nombres de archivo: si es mulaw, usamos los nombres estándar para que report.py funcione.
        if use_mulaw:
            before_png = os.path.join(figdir, "A_bits_hist_before.png")
            scr_png    = os.path.join(figdir, "A_bits_hist_scrambled.png")
            huf_png    = os.path.join(figdir, "A_bits_hist_huffman.png")
        else:
            before_png = os.path.join(figdir, f"A_bits_hist_before_{prefix}.png")
            scr_png    = os.path.join(figdir, f"A_bits_hist_scrambled_{prefix}.png")
            huf_png    = os.path.join(figdir, f"A_bits_hist_huffman_{prefix}.png")

        # Histogramas individuales (probabilidad en eje Y)
        plot_hist_bits(bitsA, f"Fuente A ({q_name}): histograma de bits (antes)", before_png, as_probability=True)

        # 4) Scrambling (PRBS LFSR)
        bitsA_scr = scramble(bitsA, seed=lfsr_seed, taps=lfsr_taps, bitwidth=lfsr_bitwidth)
        plot_hist_bits(bitsA_scr, f"Fuente A ({q_name}): histograma de bits (scrambling)", scr_png, as_probability=True)

        # 5) Huffman (sobre símbolos cuantizados, no sobre bits)
        bitsA_huff, codeA, LavgA = encode(qA.tolist())
        plot_hist_bits(bitsA_huff, f"Fuente A ({q_name}): histograma de bits (Huffman)", huf_png, as_probability=True)

        # 6) Comparativa y evolución de entropía
        plot_hist_bits_compare(
            bitsA, "Antes", bitsA_scr, "Scrambling",
            "Fuente A: histogramas comparativos (probabilidad)",
            os.path.join(figdir, f"A_bits_hist_compare_{prefix}.png"),
            as_probability=True,
        )
        plot_entropy_evolution(
            [bitsA, bitsA_scr],
            [f"{q_name} original", f"{q_name} + scrambler"],
            os.path.join(figdir, f"A_entropy_evolution_{prefix}.png"),
            step=entropy_step
        )

        # 7) Métricas
        p0A, p1A, HA, varA = bits_entropy_stats(bitsA)
        p0A_s, p1A_s, HA_s, varA_s = bits_entropy_stats(bitsA_scr)
        p0A_h, p1A_h, HA_h, varA_h = bits_entropy_stats(bitsA_huff)

        rows_local.extend([
            (f"Fuente A [{q_name}] – Antes", p0A, p1A, HA, varA, float("nan")),
            (f"Fuente A [{q_name}] – Scrambling", p0A_s, p1A_s, HA_s, varA_s, float("nan")),
            (f"Fuente A [{q_name}] – Huffman", p0A_h, p1A_h, HA_h, varA_h, LavgA),
        ])

    if quantizer == "mulaw":
        run_pipeline_for_quantizer("µ-law", use_mulaw=True)
    elif quantizer == "uniform":
        run_pipeline_for_quantizer("Uniforme", use_mulaw=False)
    else:  # "both"
        # Genera ambas variantes. La versión µ-law usa nombres estándar de imágenes para compatibilidad con report.py
        run_pipeline_for_quantizer("Uniforme", use_mulaw=False)
        run_pipeline_for_quantizer("µ-law", use_mulaw=True)

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

    # 4) Huffman (sobre bytes del texto)
    bitsB_huff, codeB, LavgB = encode(bytes_list)
    huf_png = os.path.join(figdir, "B_bits_hist_huffman.png")
    plot_hist_bits(bitsB_huff, "Fuente B: histograma de bits (Huffman)", huf_png, as_probability=True)

    # 5) Comparativa y evolución de entropía
    plot_hist_bits_compare(
        bitsB, "Antes", bitsB_scr, "Scrambling",
        "Fuente B: histogramas comparativos (probabilidad)",
        os.path.join(figdir, "B_bits_hist_compare.png"),
        as_probability=True
    )
    plot_entropy_evolution(
        [bitsB, bitsB_scr],
        ["Texto original", "Texto + scrambler"],
        os.path.join(figdir, "B_entropy_evolution.png"),
        step=entropy_step
    )

    # 6) Métricas
    p0B, p1B, HB, varB = bits_entropy_stats(bitsB)
    p0B_s, p1B_s, HB_s, varB_s = bits_entropy_stats(bitsB_scr)
    p0B_h, p1B_h, HB_h, varB_h = bits_entropy_stats(bitsB_huff)

    return (
        ("Fuente B – Antes", p0B, p1B, HB, varB, float("nan")),
        ("Fuente B – Scrambling", p0B_s, p1B_s, HB_s, varB_s, float("nan")),
        ("Fuente B – Huffman", p0B_h, p1B_h, HB_h, varB_h, LavgB),
    )


def main():
    ap = argparse.ArgumentParser(description="Lab 1 – Formateo y Ecualización del Histograma")
    ap.add_argument("--audio", required=True, help="Ruta a WAV PCM (voz)")
    ap.add_argument("--text", required=True, help="Ruta a archivo de texto (UTF-8)")
    ap.add_argument("--out", default="outputs", help="Directorio de salida")
    ap.add_argument("--fs", type=int, default=16000, help="Frecuencia de muestreo objetivo (Hz)")
    ap.add_argument("--n_bits", type=int, default=8, help="Bits de cuantificación (1-16)")
    ap.add_argument("--quantizer", choices=["uniform", "mulaw", "both"], default="mulaw",
                    help="Tipo de cuantizador para audio: uniforme, µ-law, o ambos (comparación). Default: mulaw")
    ap.add_argument("--mu", type=int, default=255, help="Parámetro µ de µ-law (solo si quantizer=mulaw)")
    ap.add_argument("--lfsr_seed", type=lambda x: int(x, 0), default=0b1010110011, help="Semilla del LFSR (base 0b o 0x aceptada)")
    ap.add_argument("--lfsr_taps", type=str, default="9,6", help="Taps del LFSR como lista separada por comas, ej: 9,6")
    ap.add_argument("--lfsr_bitwidth", type=int, default=10, help="Bitwidth del LFSR (1-32)")
    ap.add_argument("--hist_bins", type=int, default=50, help="Bins para histograma de amplitudes de audio")
    ap.add_argument("--entropy_step_a", type=int, default=10000, help="Paso para evolución de entropía (audio)")
    ap.add_argument("--entropy_step_b", type=int, default=500, help="Paso para evolución de entropía (texto)")
    ap.add_argument("--sqnr_mus", type=str, default="1,25,50,100,255", help="Lista de µ para gráfica SQNR/MSE (separados por coma)")
    args = ap.parse_args()

    figdir = ensure_dirs(args.out)

    rows = []
    # Parse taps
    try:
        taps = tuple(int(t.strip()) for t in str(args.lfsr_taps).split(",") if t.strip() != "")
    except Exception:
        taps = (9, 6)
    rows.extend(process_audio(
        args.audio, figdir,
        fs_target=args.fs,
        n_bits=args.n_bits,
        quantizer=args.quantizer,
        mu_val=args.mu,
        lfsr_seed=args.lfsr_seed,
        lfsr_taps=taps,
        lfsr_bitwidth=args.lfsr_bitwidth,
        hist_bins=args.hist_bins,
        entropy_step=args.entropy_step_a,
    ))
    rows.extend(process_text(
        args.text, figdir,
        lfsr_seed=args.lfsr_seed,
        lfsr_taps=taps,
        lfsr_bitwidth=args.lfsr_bitwidth,
        entropy_step=args.entropy_step_b,
    ))

    # Evaluación SQNR/MSE automática para la Fuente A (audio)
    try:
        x, _ = load_wav_mono(args.audio, target_fs=args.fs)
        xhat_uni = sqnr_eval.eval_uniform(x, bits=args.n_bits)
        mse_uni = sqnr_eval.mse(x, xhat_uni)
        sqnr_uni = sqnr_eval.sqnr_db(x, xhat_uni)

        try:
            mu_vals = [int(v.strip()) for v in str(args.sqnr_mus).split(",") if v.strip() != ""]
        except Exception:
            mu_vals = [1, 25, 50, 100, 255]
        mses, sqnrs = [], []
        for mu in mu_vals:
            xhat_mu = sqnr_eval.eval_mulaw(x, mu=mu, bits=args.n_bits)
            mses.append(sqnr_eval.mse(x, xhat_mu))
            sqnrs.append(sqnr_eval.sqnr_db(x, xhat_mu))

        import matplotlib.pyplot as plt
        labels = ["Uniforme"] + [f"µ={mu}" for mu in mu_vals]
        # SQNR plot
        plt.figure()
        plt.bar(range(len([sqnr_uni] + sqnrs)), [sqnr_uni] + sqnrs)
        plt.xticks(range(len(labels)), labels)
        plt.ylabel("SQNR (dB)")
        plt.title("Comparación de SQNR (8 bits)")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "sqnr_comparacion.png"), dpi=140)
        plt.close()

        # MSE plot
        plt.figure()
        plt.bar(range(len([mse_uni] + mses)), [mse_uni] + mses)
        plt.xticks(range(len(labels)), labels)
        plt.ylabel("MSE")
        plt.title("Comparación de MSE (8 bits)")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "mse_comparacion.png"), dpi=140)
        plt.close()

        # CSV resumen
        import pandas as pd
        rows_sqnr = [("Uniforme", sqnr_uni, mse_uni)] + [(f"µ={mu}", s, m) for mu, s, m in zip(mu_vals, sqnrs, mses)]
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
