
# Lab 1 – Formateo y Ecualización del Histograma (Pipeline reproducible)

Este repositorio implementa el pipeline para el **Laboratorio N°1**: tomar dos fuentes (audio y texto), **formatear** a binario y aplicar **ecualización del histograma** por dos métodos:
1) **Scrambling PRBS (LFSR)** – blanqueo/equiprobabilidad sin cambiar la tasa.
2) **Codificación de fuente (Huffman)** – compresión por redundancia de símbolos.

Se generan **gráficos** (señal, histogramas) y **métricas** (P(0), P(1), entropía, varianza, longitud media de Huffman), además de un **informe Markdown** y un **CSV** resumen.


## Requisitos
- Python 3.10+
- Instalar dependencias:
```bash
pip install -r requirements.txt


# 1) Activar venv (opcional)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Instalar deps
pip install -r requirements.txt

# 3) Ejecutar el pipeline
python -m src.main --audio data/voice.wav --text data/sample_text.txt --out outputs --fs 16000 --n_bits 8

## Uso rápido

# 1) Activar venv (opcional)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Instalar deps
pip install -r requirements.txt

# 3) Ejecutar el pipeline
python -m src.main --audio data/voice.wav --text data/sample_text.txt --out outputs --fs 16000 --n_bits 8

Parámetros relevantes:
    •    --audio: WAV (PCM) a procesar (mono o estéreo). Si es estéreo, se convierte a mono.
    •    --text: archivo de texto (UTF-8).
    •    --fs: frecuencia de muestreo objetivo del audio (por defecto 16000 Hz).
    •    --n_bits: bits de cuantificación uniforme (por defecto 8).

Salidas
    •    outputs/figures/*.png: señal en el tiempo, histograma de amplitudes, histogramas de bits antes y después (scrambling y Huffman).
    •    outputs/resumen_metricas.csv
    •    outputs/informe_lab1.md

Notas didácticas
    •    Scrambling busca P(0)≈P(1) para mejorar transmisión (relojeo, DC balance). No comprime.
    •    Huffman explota redundancia de símbolos (texto, niveles cuantizados). Reduce la longitud media, acercándose al límite dado por la entropía.
