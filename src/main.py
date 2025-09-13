import argparse
import os
from .audio_utils import (
    load_wav_mono,
    uniform_quantize,
    save_signal_plot,
    save_hist_amplitudes,
    plot_hist_bits,
)
from .bits_utils import ints_to_bits, bits_to_bytes, bits_entropy_stats
from .scrambling import scramble
from .huffman import encode
from .report import write_markdown, save_metrics_csv


def ensure_dirs(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    figdir = os.path.join(out_dir, "figures")
    os.makedirs(figdir, exist_ok=True)
    return figdir


def process_audio(audio_path: str, figdir: str, fs_target: int = 16000, n_bits: int = 8):
    # 1) Cargar audio, convertir a mono y normalizar
    x, fs = load_wav_mono(audio_path, target_fs=fs_target)

    # 2) Gráficos previos
    save_signal_plot(x, fs, "Fuente A: señal 'voz' (tiempo)", os.path.join(figdir, "A_signal_time.png"))
    save_hist_amplitudes(x, "Fuente A: histograma de amplitudes (antes)", os.path.join(figdir, "A_signal_hist.png"))

    # 3) Cuantificación uniforme -> símbolos (0..2^n_bits-1) -> bits
    qA, LA = uniform_quantize(x, bits=n_bits, xmin=-1, xmax=1)
    bitsA = ints_to_bits(qA, n_bits)
    plot_hist_bits(bitsA, "Fuente A: histograma de bits (antes)", os.path.join(figdir, "A_bits_hist_before.png"))

    # 4) Scrambling (PRBS LFSR)
    bitsA_scr = scramble(bitsA)
    plot_hist_bits(bitsA_scr, "Fuente A: histograma de bits (scrambling)", os.path.join(figdir, "A_bits_hist_scrambled.png"))

    # 5) Huffman (sobre símbolos cuantizados, no sobre bits)
    bitsA_huff, codeA, LavgA = encode(qA.tolist())
    plot_hist_bits(bitsA_huff, "Fuente A: histograma de bits (Huffman)", os.path.join(figdir, "A_bits_hist_huffman.png"))

    # 6) Métricas
    p0A, p1A, HA, varA = bits_entropy_stats(bitsA)
    p0A_s, p1A_s, HA_s, varA_s = bits_entropy_stats(bitsA_scr)
    p0A_h, p1A_h, HA_h, varA_h = bits_entropy_stats(bitsA_huff)

    return (
        ("Fuente A – Antes", p0A, p1A, HA, varA, float("nan")),
        ("Fuente A – Scrambling", p0A_s, p1A_s, HA_s, varA_s, float("nan")),
        ("Fuente A – Huffman", p0A_h, p1A_h, HA_h, varA_h, LavgA),
    )


def process_text(text_path: str, figdir: str):
    # 1) Leer texto UTF-8 -> bytes -> bits
    txt = open(text_path, "r", encoding="utf-8").read()
    b = txt.encode("utf-8")
    bytes_list = list(b)

    bitsB = []
    for val in bytes_list:
        for bit in range(7, -1, -1):
            bitsB.append((val >> bit) & 1)

    # 2) Histograma de bits antes
    plot_hist_bits(bitsB, "Fuente B: histograma de bits (antes)", os.path.join(figdir, "B_bits_hist_before.png"))

    # 3) Scrambling
    bitsB_scr = scramble(bitsB)
    plot_hist_bits(bitsB_scr, "Fuente B: histograma de bits (scrambling)", os.path.join(figdir, "B_bits_hist_scrambled.png"))

    # 4) Huffman (sobre bytes del texto)
    bitsB_huff, codeB, LavgB = encode(bytes_list)
    plot_hist_bits(bitsB_huff, "Fuente B: histograma de bits (Huffman)", os.path.join(figdir, "B_bits_hist_huffman.png"))

    # 5) Métricas
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
    ap.add_argument("--n_bits", type=int, default=8, help="Bits de cuantificación uniforme")
    args = ap.parse_args()

    figdir = ensure_dirs(args.out)

    rows = []
    rows.extend(process_audio(args.audio, figdir, fs_target=args.fs, n_bits=args.n_bits))
    rows.extend(process_text(args.text, figdir))

    # Guardar CSV + Informe
    save_metrics_csv(args.out, rows)
    write_markdown(args.out)

    print(f"Listo. Salidas en: {args.out}")


if __name__ == "__main__":
    main()
