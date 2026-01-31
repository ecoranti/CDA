import os
import pandas as pd


def _md_table_from_df(df: pd.DataFrame) -> str:
    # Construye una tabla Markdown simple a partir del DataFrame
    headers = " | ".join(df.columns)
    sep = " | ".join(["---"] * len(df.columns))
    rows = []
    for _, r in df.iterrows():
        rows.append(" | ".join(str(r[c]) for c in df.columns))
    return "\n".join([headers, sep] + rows)


def write_markdown(out_dir: str):
    fig = os.path.join(out_dir, "figures")

    # Detectar variantes de figuras de audio (mulaw/uniform)
    def pick(name_base: str):
        # Retorna el path relativo a figures/ si existe alguna variante conocida
        candidates = [
            f"{name_base}_mulaw.png",
            f"{name_base}_uniform.png",
            f"{name_base}.png",
        ]
        for c in candidates:
            if os.path.exists(os.path.join(fig, c)):
                return f"figures/{c}"
        return None

    a_before = pick("A_bits_hist_before") or "figures/A_bits_hist_before.png"
    a_scr = pick("A_bits_hist_scrambled") or "figures/A_bits_hist_scrambled.png"
    a_huf = pick("A_bits_hist_huffman") or "figures/A_bits_hist_huffman.png"
    a_cmp = pick("A_bits_hist_compare")
    a_ent = pick("A_entropy_evolution")

    b_before = "figures/B_bits_hist_before.png"
    b_scr = "figures/B_bits_hist_scrambled.png"
    b_huf = "figures/B_bits_hist_huffman.png"
    b_cmp = "figures/B_bits_hist_compare.png"
    b_ent = "figures/B_entropy_evolution.png"

    # Cargar métricas si existen para incrustar una tabla
    metrics_path = os.path.join(out_dir, "resumen_metricas.csv")
    metrics_md = None
    if os.path.exists(metrics_path):
        try:
            df = pd.read_csv(metrics_path)
            metrics_md = _md_table_from_df(df)
        except Exception:
            metrics_md = None

    parts = []
    parts.append("# Lab 1 – Formateo & Ecualización del Histograma")

    parts.append("\n## 1) Señal original (audio)")
    parts.append("![A_signal_time](figures/A_signal_time.png)")
    parts.append("![A_signal_hist](figures/A_signal_hist.png)")

    parts.append("\n## 2) Histogramas de bits (antes)")
    parts.append(f"- Audio: ![A_bits_before]({a_before})")
    parts.append(f"- Texto: ![B_bits_before]({b_before})")

    parts.append("\n## 3) Scrambling (PRBS con LFSR)")
    parts.append(f"- Audio: ![A_bits_scram]({a_scr})")
    parts.append(f"- Texto: ![B_bits_scram]({b_scr})")

    parts.append("\n## 4) Huffman")
    parts.append(f"- Audio: ![A_bits_huff]({a_huf})")
    parts.append(f"- Texto: ![B_bits_huff]({b_huf})")

    # Comparativas y evolución de entropía (si existen)
    comp_blocks = []
    if a_cmp:
        comp_blocks.append(f"- Audio: ![A_bits_compare]({a_cmp})")
    if os.path.exists(os.path.join(out_dir, b_cmp)):
        comp_blocks.append(f"- Texto: ![B_bits_compare]({b_cmp})")
    if comp_blocks:
        parts.append("\n## 5) Histogramas comparativos (Antes vs Scrambling)")
        parts.extend(comp_blocks)

    ent_blocks = []
    if a_ent:
        ent_blocks.append(f"- Audio: ![A_entropy]({a_ent})")
    if os.path.exists(os.path.join(out_dir, b_ent)):
        ent_blocks.append(f"- Texto: ![B_entropy]({b_ent})")
    if ent_blocks:
        parts.append("\n## 6) Evolución de la entropía")
        parts.extend(ent_blocks)

    # SQNR/MSE si existen
    sqnr_png = os.path.join(out_dir, "sqnr_comparacion.png")
    mse_png = os.path.join(out_dir, "mse_comparacion.png")
    if os.path.exists(sqnr_png) or os.path.exists(mse_png):
        parts.append("\n## 7) Métricas adicionales de cuantización")
        if os.path.exists(sqnr_png):
            parts.append("![SQNR](sqnr_comparacion.png)")
        if os.path.exists(mse_png):
            parts.append("![MSE](mse_comparacion.png)")

    parts.append("\n## 8) Métricas (tabla)")
    if metrics_md:
        parts.append(metrics_md)
    else:
        parts.append("Ver archivo resumen_metricas.csv.")

    parts.append("\n**Notas**")
    parts.append("- Scrambling: busca equiprobabilidad P(0)≈P(1) sin cambiar la tasa.")
    parts.append("- Huffman: reduce longitud media cuando hay redundancia.")

    # Sección para discusión crítica (para completar manualmente)
    parts.append("\n## 9) Discusión (para completar)")
    parts.append("- Ventajas/desventajas de las técnicas aplicadas.")
    parts.append("- Propuestas de mejora y aplicaciones.")

    md = "\n".join(parts) + "\n"
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


def write_pdf(out_dir: str):
    """Genera informe PDF con imágenes y tabla de métricas.

    Requiere reportlab instalado. Si falla, no rompe el pipeline; solo imprime aviso.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
        from reportlab.lib import colors
    except Exception as e:
        print(f"[WARN] No se pudo importar reportlab para generar PDF: {e}")
        return

    figdir = os.path.join(out_dir, "figures")

    def pick(name_base: str):
        for c in [f"{name_base}_mulaw.png", f"{name_base}_uniform.png", f"{name_base}.png"]:
            p = os.path.join(figdir, c)
            if os.path.exists(p):
                return p
        return None

    doc = SimpleDocTemplate(os.path.join(out_dir, "informe_lab1.pdf"), pagesize=A4)
    W, H = A4
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Lab 1 – Formateo & Ecualización del Histograma", styles['Title']))
    story.append(Spacer(1, 6*mm))

    def add_img(path: str, caption: str):
        if not path or not os.path.exists(path):
            return
        # Ajuste de ancho con margen
        iw = W - 30*mm
        im = Image(path, width=iw, height=None, kind='proportional')
        story.append(im)
        story.append(Paragraph(caption, styles['Normal']))
        story.append(Spacer(1, 4*mm))

    # 1) Señal original
    add_img(os.path.join(figdir, "A_signal_time.png"), "Señal en el tiempo (audio)")
    add_img(os.path.join(figdir, "A_signal_hist.png"), "Histograma de amplitudes (audio)")

    # 2) Histogramas de bits
    add_img(pick("A_bits_hist_before"), "Histogramas de bits – Audio (antes)")
    add_img(os.path.join(figdir, "B_bits_hist_before.png"), "Histogramas de bits – Texto (antes)")

    # 3) Scrambling
    add_img(pick("A_bits_hist_scrambled"), "Histogramas de bits – Audio (scrambling)")
    add_img(os.path.join(figdir, "B_bits_hist_scrambled.png"), "Histogramas de bits – Texto (scrambling)")

    # 4) Huffman
    add_img(pick("A_bits_hist_huffman"), "Histogramas de bits – Audio (Huffman)")
    add_img(os.path.join(figdir, "B_bits_hist_huffman.png"), "Histogramas de bits – Texto (Huffman)")

    # 5) Comparativas y entropía
    add_img(pick("A_bits_hist_compare"), "Comparativa Antes vs Scrambling – Audio")
    add_img(os.path.join(figdir, "B_bits_hist_compare.png"), "Comparativa Antes vs Scrambling – Texto")
    add_img(pick("A_entropy_evolution"), "Evolución de la entropía – Audio")
    add_img(os.path.join(figdir, "B_entropy_evolution.png"), "Evolución de la entropía – Texto")

    # 6) Métricas de cuantización
    add_img(os.path.join(out_dir, "sqnr_comparacion.png"), "Comparación de SQNR (dB)")
    add_img(os.path.join(out_dir, "mse_comparacion.png"), "Comparación de MSE")

    # 7) Tabla de métricas
    metrics_path = os.path.join(out_dir, "resumen_metricas.csv")
    if os.path.exists(metrics_path):
        try:
            df = pd.read_csv(metrics_path)
            data = [list(df.columns)] + df.values.tolist()
            tbl = Table(data, repeatRows=1)
            tbl.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                ('GRID', (0,0), (-1,-1), 0.3, colors.grey),
                ('ALIGN', (1,1), (-1,-1), 'CENTER'),
            ]))
            story.append(Spacer(1, 4*mm))
            story.append(tbl)
        except Exception as e:
            story.append(Paragraph(f"No se pudo cargar resumen_metricas.csv: {e}", styles['Italic']))

    doc.build(story)
    return os.path.join(out_dir, "informe_lab1.pdf")
