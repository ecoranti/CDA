import os
import pandas as pd

def write_markdown(out_dir: str):
    md = """# Lab 1 – Formateo & Ecualización del Histograma

## 1) Señal original (audio)
![A_signal_time](figures/A_signal_time.png)
![A_signal_hist](figures/A_signal_hist.png)

## 2) Histogramas de bits antes de ecualizar
- Audio: ![A_bits_before](figures/A_bits_hist_before.png)
- Texto: ![B_bits_before](figures/B_bits_hist_before.png)

## 3) Método 1 – Scrambling (PRBS con LFSR)
- Audio: ![A_bits_scram](figures/A_bits_hist_scrambled.png)
- Texto: ![B_bits_scram](figures/B_bits_hist_scrambled.png)

## 4) Método 2 – Huffman
- Audio: ![A_bits_huff](figures/A_bits_hist_huffman.png)
- Texto: ![B_bits_huff](figures/B_bits_hist_huffman.png)

## 5) Métricas
Ver **resumen_metricas.csv**.

**Notas**
- Scrambling: busca equiprobabilidad P(0)≈P(1) sin cambiar la tasa.
- Huffman: reduce longitud media cuando hay redundancia.
"""
    with open(os.path.join(out_dir, "informe_lab1.md"), "w", encoding="utf-8") as f:
        f.write(md)

def save_metrics_csv(out_dir: str, rows):
    df = pd.DataFrame(
        rows,
        columns=[
            "Caso",
            "P(0)",
            "P(1)",
            "Entropía [bits/bit]",
            "Varianza bits",
            "Longitud media (Huffman)",
        ],
    )
    df.to_csv(os.path.join(out_dir, "resumen_metricas.csv"), index=False)
    return df
