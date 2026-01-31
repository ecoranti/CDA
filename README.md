# Proyecto CDA - Laboratorios 1, 2 y 3

Este repositorio integra los pipelines de procesamiento para la materia Comunicación de Datos y Audio (Sistemas de Comunicaciones). Incluye una **Interfaz Web (Flask)** unificada para facilitar la ejecución y visualización de los tres laboratorios:

*   **Lab 1 - Formateo y Ecualización**: Cuantización (Uniforme/$\mu$-law), Compresión (Huffman) y Scrambling.
*   **Lab 2 - Modulación Digital + RRC**: Mapeo de símbolos (BPSK/QPSK) y filtro conformador de pulso (Raíz de Coseno Alzado).
*   **Lab 3 - Demodulación y BER**: Canal AWGN, Filtro Acoplado, recuperación de reloj y estimación de Tasa de Error de Bit (BER).

---

## 🚀 Guía de Inicio Rápido (De cero)

Sigue estos pasos para instalar y correr el proyecto en tu sistema local.

### 1. Prerrequisitos
*   **Python 3.10** o superior.
*   **Git** (opcional, para clonar).

### 2. Instalación

Se recomienda encarecidamente usar un **entorno virtual** para evitar conflictos de dependencias.

#### Paso 1: Clonar o descargar el código
Si tienes git:
```bash
git clone <url-del-repo>
cd cda
```
O simplemente descomprime el archivo ZIP en una carpeta.

#### Paso 2: Crear y activar entorno virtual
En la terminal (dentro de la carpeta del proyecto):

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Paso 3: Instalar dependencias
```bash
pip install -r requirements.txt
```

---

## 🖥️ Ejecución de la Aplicación Web

La forma más fácil de interactuar con el proyecto es mediante la aplicación web incluida.

1.  Asegúrate de tener el entorno virtual activado.
2.  Ejecuta el servidor:
    ```bash
    # Opción A (Puerto por defecto 5000)
    python app/app.py

    # Opción B (Especificar puerto, útil si el 5000 está ocupado)
    PORT=5001 python app/app.py
    ```
3.  Abre tu navegador (Chrome, Firefox, Safari) e ingresa a:
    *   **http://127.0.0.1:5000/** (o el puerto que hayas configurado).

Verás el menú principal con acceso a los tres laboratorios.

---

## 🧪 Descripción de los Laboratorios

### Lab 1: Formateo y Fuente
Convierte señales analógicas (audio) o texto a un flujo de bits digital.
*   **Features**: Cuantización ajustable (bits, $\mu$), codificación entrópica (Huffman) y aleatorización (Scrambling).
*   **Salida**: Gráficos de histogramas de bits, evolución de entropía, comparativas de SQNR/MSE.

### Lab 2: Transmisor Digital
Toma una secuencia de bits (o genera una aleatoria) y simula la etapa de transmisión.
*   **Features**: Modulaciones BPSK/QPSK, Filtro RRC con *roll-off* ($\alpha$) variable, sobremuestreo (SPS).
*   **Salida**: Diagrama de Constelación (Tx), Ojo, Espectro, y archivos `.bin` (IQ flotante) para SDR.

### Lab 3: Receptor y Canal
Simula el canal de comunicaciones y la etapa de recepción.
*   **Features**: Canal AWGN (ruido gaussiano), Filtro Acoplado (Matched Filter), estimación de BER vs Eb/N0.
*   **Modos**:
    *   **Simulación de Curva**: Barre valores de Eb/N0 para generar la curva de BER.
    *   **Integración**: Puede tomar la salida del Lab 2 y demodularla para verificar la transmisión completa.

---

## ⌨️ Ejecución vía Consola (CLI)

Si prefieres usar la línea de comandos para scripts automatizados:

**Lab 1:**
```bash
python -m src.main --audio data/voice.wav --n_bits 8 --quantizer mulaw --out outputs/cli_lab1
```

**Ayuda:**
```bash
python -m src.main -h
```

---

## 📂 Estructura de Archivos

*   `app/`: Código de la aplicación web (Flask) y templates HTML.
*   `src/`: Librerías core de procesamiento DSP.
    *   `main.py`: Lógica Lab 1.
    *   `lab2_rrc.py`: Lógica Lab 2.
    *   `lab3_demod.py`: Lógica Lab 3.
*   `data/`: Archivos de entrada de ejemplo (audio, texto).
*   `outputs_ui/`: Carpeta donde se guardan los resultados de las corridas web (organizados por fecha).
