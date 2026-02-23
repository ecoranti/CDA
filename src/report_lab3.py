from __future__ import annotations

import os
from typing import Iterable

import pandas as pd


def _exists(path: str) -> bool:
    return os.path.exists(path) and os.path.isfile(path)


def _first_existing(out_dir: str, names: Iterable[str]) -> str | None:
    for n in names:
        p = os.path.join(out_dir, n)
        if _exists(p):
            return p
    return None


def _md_rel(path: str, out_dir: str) -> str:
    return os.path.relpath(path, out_dir).replace("\\", "/")


def write_markdown(out_dir: str) -> str:
    """Genera informe_lab3.md con la estructura pedida por la guía."""
    parts: list[str] = []
    parts.append("# Lab 3 – Demodulación Digital (Canal AWGN + Filtro Acoplado + Estimación Bayesiana)")
    parts.append("")
    parts.append("## 1) Descripción teórica y diagrama de bloques")
    parts.append("Cadena completa utilizada: **Lab 2 (Tx IQ) -> Canal AWGN -> Filtro acoplado (RRC) -> Muestreo óptimo -> Decisión ML/MAP -> BER**.")
    parts.append("")
    parts.append("## 2) Curva Pb(Eb/N0) experimental vs teórica")
    p = _first_existing(out_dir, ["ber_curve.png"])
    if p:
        parts.append(f"![ber_curve]({_md_rel(p, out_dir)})")
    else:
        parts.append("No disponible.")
    parts.append("")

    parts.append("## 3) Constellaciones antes y después del canal")
    p = _first_existing(out_dir, ["tx_rx_constellations.png"])
    if p:
        parts.append(f"![tx_rx_const]({_md_rel(p, out_dir)})")
    else:
        parts.append("No disponible.")
    parts.append("")

    parts.append("## 4) Salida del receptor (tiempo, constelación y ojo)")
    for name in ["rx_time.png", "rx_constellation.png", "rx_eye.png"]:
        p = _first_existing(out_dir, [name])
        if p:
            parts.append(f"![{name}]({_md_rel(p, out_dir)})")
    parts.append("")

    parts.append("## 5) Filtro acoplado")
    p1 = _first_existing(out_dir, ["mf_impulse.png"])
    p2 = _first_existing(out_dir, ["mf_freq.png"])
    if p1:
        parts.append(f"![mf_impulse]({_md_rel(p1, out_dir)})")
    if p2:
        parts.append(f"![mf_freq]({_md_rel(p2, out_dir)})")
    if not p1 and not p2:
        parts.append("No disponible.")
    parts.append("")

    parts.append("## 6) Decisión ML/MAP")
    p = _first_existing(out_dir, ["rx_decision.png", "ber_point.png"])
    if p:
        parts.append(f"![rx_decision]({_md_rel(p, out_dir)})")
    else:
        parts.append("No disponible.")
    parts.append("")

    parts.append("## 7) Tabla de resultados")
    csv_path = os.path.join(out_dir, "ber_results.csv")
    if _exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            parts.append(df.to_markdown(index=False))
        except Exception:
            parts.append("Ver `ber_results.csv`.")
    else:
        parts.append("No disponible.")
    parts.append("")

    parts.append("## 8) Discusión técnica (completar)")
    parts.append("- Comparación teoría vs simulación (desvíos y causas).")
    parts.append("- Impacto de roll-off, longitud de filtro y sps.")
    parts.append("- Trade-offs entre ancho de banda, ISI y complejidad.")
    parts.append("")

    out_md = os.path.join(out_dir, "informe_lab3.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(parts).rstrip() + "\n")
    return out_md


def write_pdf(out_dir: str) -> str | None:
    """Genera informe_lab3.pdf a partir de salidas del Lab 3."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
        from reportlab.lib import colors
        from reportlab.lib.utils import ImageReader
    except Exception as e:
        print(f"[WARN] No se pudo importar reportlab para PDF Lab3: {e}")
        return None

    pdf_path = os.path.join(out_dir, "informe_lab3.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    W, _ = A4
    story = []

    story.append(Paragraph("Lab 3 – Demodulación Digital (AWGN + MF + BER)", styles["Title"]))
    story.append(Spacer(1, 5 * mm))

    def add_img(name: str, caption: str):
        p = os.path.join(out_dir, name)
        if not _exists(p):
            return
        iw, ih = ImageReader(p).getSize()
        max_w = W - 30 * mm
        max_h = 90 * mm
        scale = min(max_w / float(iw), max_h / float(ih))
        story.append(Image(p, width=iw * scale, height=ih * scale))
        story.append(Paragraph(caption, styles["Normal"]))
        story.append(Spacer(1, 4 * mm))

    add_img("ber_curve.png", "Curva BER experimental vs teórica (con banda de confianza).")
    add_img("tx_rx_constellations.png", "Constelaciones Tx vs Rx.")
    add_img("rx_time.png", "Señal Rx filtrada en el tiempo (I/Q).")
    add_img("rx_constellation.png", "Constelación Rx.")
    add_img("rx_eye.png", "Diagrama de ojo Rx.")
    add_img("mf_impulse.png", "Filtro acoplado: respuesta impulsiva.")
    add_img("mf_freq.png", "Filtro acoplado: respuesta en frecuencia.")
    add_img("rx_decision.png", "Salida del filtro y umbral de decisión ML/MAP.")
    add_img("ber_point.png", "Punto de operación BER para Eb/N0 puntual.")

    csv_path = os.path.join(out_dir, "ber_results.csv")
    if _exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            data = [list(df.columns)] + df.values.tolist()
            table = Table(data, repeatRows=1)
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                    ]
                )
            )
            story.append(Spacer(1, 4 * mm))
            story.append(table)
        except Exception as e:
            story.append(Paragraph(f"No se pudo renderizar tabla CSV: {e}", styles["Italic"]))

    doc.build(story)
    return pdf_path
