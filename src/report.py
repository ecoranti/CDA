import os
import json
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
    params_path = os.path.join(out_dir, "params.json")
    params = {}
    if os.path.exists(params_path):
        try:
            with open(params_path, "r", encoding="utf-8") as f:
                params = json.load(f)
        except Exception:
            params = {}

    source = (params.get("source") or "audio").lower()
    quantizer = (params.get("quantizer") or "alaw").lower()
    quantizer_label = "A-law" if quantizer == "alaw" else "Uniforme"

    # Detectar variantes de figuras de audio (A-law principal o uniforme como comparación)
    def pick(name_base: str):
        # Retorna el path relativo a figures/ si existe alguna variante conocida
        preferred = f"{name_base}_{quantizer}.png"
        fallback = "uniform" if quantizer == "alaw" else "alaw"
        candidates = [
            preferred,
            f"{name_base}_{fallback}.png",
            f"{name_base}.png",
        ]
        for c in candidates:
            if os.path.exists(os.path.join(fig, c)):
                return f"figures/{c}"
        return None

    a_before = pick("A_bits_hist_before") or "figures/A_bits_hist_before.png"
    a_scr = pick("A_bits_hist_scrambled") or "figures/A_bits_hist_scrambled.png"
    a_cmp = pick("A_bits_hist_compare")
    a_ent = pick("A_entropy_evolution")
    a_sig_cmp = pick("A_signal_quantized_compare")
    a_qchar = pick("A_quantizer_characteristic")
    a_low_cmp = "figures/A_quantizer_low_level_compare.png" if os.path.exists(os.path.join(fig, "A_quantizer_low_level_compare.png")) else None
    a_err_cmp = "figures/A_quantization_error_compare.png" if os.path.exists(os.path.join(fig, "A_quantization_error_compare.png")) else None

    b_before = "figures/B_bits_hist_before.png"
    b_scr = "figures/B_bits_hist_scrambled.png"
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
    parts.append("# Formateo – Cuantización y Scrambling")

    if source == "audio":
        parts.append("\n## 1) Señal original (audio)")
        parts.append("![A_signal_time](figures/A_signal_time.png)")
        parts.append("![A_signal_hist](figures/A_signal_hist.png)")
        if os.path.exists(os.path.join(fig, "A_wav_int16_hist.png")):
            parts.append("![A_wav_int16_hist](figures/A_wav_int16_hist.png)")

        parts.append(f"\n## 2) Histogramas de bits (antes) [{quantizer_label}]")
        parts.append(f"![A_bits_before]({a_before})")

        parts.append("\n## 3) Scrambling (PRBS con LFSR)")
        parts.append(f"![A_bits_scram]({a_scr})")
    else:
        parts.append("\n## 1) Fuente original (texto)")
        parts.append(f"![B_bits_before]({b_before})")

        parts.append("\n## 2) Scrambling (PRBS con LFSR)")
        parts.append(f"![B_bits_scram]({b_scr})")

    # Comparativas y evolución de entropía (si existen)
    comp_blocks = []
    if source == "audio" and a_cmp:
        comp_blocks.append(f"![A_bits_compare]({a_cmp})")
    if source == "text" and os.path.exists(os.path.join(out_dir, b_cmp)):
        comp_blocks.append(f"![B_bits_compare]({b_cmp})")
    if comp_blocks:
        parts.append("\n## 4) Histogramas comparativos (Antes vs Scrambling)")
        parts.extend(comp_blocks)

    ent_blocks = []
    if source == "audio" and a_ent:
        ent_blocks.append(f"![A_entropy]({a_ent})")
    if source == "text" and os.path.exists(os.path.join(out_dir, b_ent)):
        ent_blocks.append(f"![B_entropy]({b_ent})")
    if ent_blocks:
        parts.append("\n## 5) Evolución de la entropía (original + scrambler)")
        parts.extend(ent_blocks)

    if source == "audio":
        qviz_blocks = []
        if a_sig_cmp:
            qviz_blocks.append(f"![A_sig_q]({a_sig_cmp})")
        if a_qchar:
            qviz_blocks.append(f"![A_q_char]({a_qchar})")
        if a_low_cmp:
            qviz_blocks.append(f"![A_low_cmp]({a_low_cmp})")
        if a_err_cmp:
            qviz_blocks.append(f"![A_err_cmp]({a_err_cmp})")
        if qviz_blocks:
            parts.append(f"\n## 6) Visualización de la cuantización [{quantizer_label}]")
            parts.extend(qviz_blocks)

    # SQNR/MSE si existen
    sqnr_png = os.path.join(out_dir, "sqnr_comparacion.png")
    mse_png = os.path.join(out_dir, "mse_comparacion.png")
    ecm_png = os.path.join(out_dir, "ecm_evolucion.png")
    if os.path.exists(sqnr_png) or os.path.exists(mse_png) or os.path.exists(ecm_png):
        parts.append(f"\n## 7) Métricas adicionales de cuantización [{quantizer_label}]")
        if source == "audio" and os.path.exists(sqnr_png):
            parts.append("![SQNR](sqnr_comparacion.png)")
        if source == "audio" and os.path.exists(mse_png):
            parts.append("![MSE](mse_comparacion.png)")
        if source == "audio" and os.path.exists(ecm_png):
            parts.append("![ECM](ecm_evolucion.png)")

    parts.append("\n## 8) Métricas (tabla)")
    if metrics_md:
        parts.append(metrics_md)
    else:
        parts.append("Ver archivo resumen_metricas.csv.")

    parts.append("\n**Notas**")
    parts.append("- Scrambling: busca equiprobabilidad P(0)≈P(1) sin cambiar la tasa.")

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
            "Longitud media",
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
        from reportlab.lib.utils import ImageReader
        from reportlab.lib import colors
    except Exception as e:
        print(f"[WARN] No se pudo importar reportlab para generar PDF: {e}")
        return

    figdir = os.path.join(out_dir, "figures")
    params_path = os.path.join(out_dir, "params.json")
    params = {}
    if os.path.exists(params_path):
        try:
            with open(params_path, "r", encoding="utf-8") as f:
                params = json.load(f)
        except Exception:
            params = {}

    source = (params.get("source") or "audio").lower()
    quantizer = (params.get("quantizer") or "alaw").lower()
    quantizer_label = "A-law" if quantizer == "alaw" else "Uniforme"

    def pick(name_base: str):
        preferred = f"{name_base}_{quantizer}.png"
        fallback = "uniform" if quantizer == "alaw" else "alaw"
        for c in [preferred, f"{name_base}_{fallback}.png", f"{name_base}.png"]:
            p = os.path.join(figdir, c)
            if os.path.exists(p):
                return p
        return None

    doc = SimpleDocTemplate(os.path.join(out_dir, "informe_lab1.pdf"), pagesize=A4)
    W, H = A4
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Formateo – Cuantización y Scrambling", styles['Title']))
    story.append(Spacer(1, 6*mm))

    def add_img(path: str, caption: str):
        if not path or not os.path.exists(path):
            return
        # Ajuste proporcional con margen
        iw, ih = ImageReader(path).getSize()
        max_w = W - 30*mm
        scale = max_w / float(iw) if iw else 1.0
        im = Image(path, width=iw*scale, height=ih*scale)
        story.append(im)
        story.append(Paragraph(caption, styles['Normal']))
        story.append(Spacer(1, 4*mm))

    if source == "audio":
        add_img(os.path.join(figdir, "A_signal_time.png"), "Señal en el tiempo (audio)")
        add_img(os.path.join(figdir, "A_signal_hist.png"), "Histograma de amplitudes (audio)")
        add_img(
            os.path.join(figdir, "A_wav_int16_hist.png"),
            "Histograma de muestras WAV en escala PCM16 (rango int16: -32768 a 32767)",
        )
        add_img(pick("A_bits_hist_before"), f"Histogramas de bits – Audio (antes, {quantizer_label})")
        add_img(pick("A_bits_hist_scrambled"), f"Histogramas de bits – Audio (scrambling, {quantizer_label})")
        add_img(pick("A_bits_hist_compare"), "Comparativa Antes vs Scrambling – Audio")
        add_img(pick("A_entropy_evolution"), "Evolución de la entropía – Audio (original + scrambler)")
        add_img(pick("A_signal_quantized_compare"), f"Señal original vs reconstruida – {quantizer_label}")
        add_img(pick("A_quantizer_characteristic"), f"Característica entrada/salida – {quantizer_label}")
        add_img(os.path.join(figdir, "A_quantizer_low_level_compare.png"), "Comparación uniforme vs A-law en tramo de baja amplitud")
        add_img(os.path.join(figdir, "A_quantization_error_compare.png"), "Comparación del error de cuantización")
        add_img(os.path.join(out_dir, "sqnr_comparacion.png"), "Comparación de SQNR (dB)")
        add_img(os.path.join(out_dir, "mse_comparacion.png"), "Comparación de MSE")
        add_img(os.path.join(out_dir, "ecm_evolucion.png"), "Evolución del ECM acumulado")
    else:
        add_img(os.path.join(figdir, "B_bits_hist_before.png"), "Histogramas de bits – Texto (antes)")
        add_img(os.path.join(figdir, "B_bits_hist_scrambled.png"), "Histogramas de bits – Texto (scrambling)")
        add_img(os.path.join(figdir, "B_bits_hist_compare.png"), "Comparativa Antes vs Scrambling – Texto")
        add_img(os.path.join(figdir, "B_entropy_evolution.png"), "Evolución de la entropía – Texto (original + scrambler)")

    # 6) Tabla de métricas
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
