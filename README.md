# CDA - Sistema Digital por Etapas

Repositorio del trabajo de **Comunicación Digital Avanzada** con una aplicación web en **Flask** para ejecutar, visualizar y encadenar las tres etapas de la cadena digital:

1. **Lab 1 - Formateo**
2. **Lab 2 - Modulación + pulso RRC**
3. **Lab 3 - Canal y recepción**

La app permite correr el sistema de punta a punta con **audio o texto**, guardar cada corrida en `outputs_ui/`, recuperar la fuente en recepción y generar informes/figuras para el trabajo.

## Entrega

- Informe final en PDF: [Final_CDA_EliasCoranti.pdf](Final_CDA_EliasCoranti.pdf)

---

## Qué hace el proyecto

### Lab 1 - Formateo
- Toma una fuente de **audio** o **texto**
- Muestrea o remuestrea
- Cuantiza en uniforme, **A-law** o ambos
- Convierte a bits
- Aplica **scrambling**
- Genera histogramas, entropía, SQNR/MSE, histograma PCM16 del WAV e informe

### Lab 2 - Modulación
- Toma los bits reales de Formateo
- Mapea a **BPSK**, **QPSK** o **M-PSK**
- Aplica **sobremuestreo**
- Conforma con **RRC**
- Guarda la señal transmitida como vector IQ en `iq.bin` / `iq_tx.bin`
- Genera constelación, espectro, pulso RRC, ojo y figuras didácticas

### Lab 3 - Canal y Rx
- Usa la salida real de Lab 2
- Permite canal **ideal** o **AWGN**
- Aplica filtro acoplado (**RRC Rx**)
- Muestrea, decide y recupera bits
- Reconstruye **audio** o **texto** a la salida del receptor
- Genera curva BER, constelaciones Rx, ojo Rx, decisión e informe

---

## Requisitos

- **Python 3.10 o superior**
- `pip`
- Navegador web moderno
- En macOS, los selectores nativos de archivo/carpeta usan `osascript`

No hace falta base de datos ni servicios externos.

---

## Instalación

### 1. Clonar el repo

```bash
git clone https://github.com/ecoranti/CDA.git
cd CDA
```

### 2. Crear entorno virtual

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## Cómo levantar la aplicación

Con el entorno virtual activo:

```bash
python app/app.py
```

Por defecto levanta en:

- [http://127.0.0.1:5001](http://127.0.0.1:5001)

Si querés usar otro puerto:

```bash
PORT=5000 python app/app.py
```

---

## Flujo recomendado de uso

La cadena correcta es:

```text
Fuente -> Muestreo -> Cuantización -> Bits -> Scrambling
-> Mapeo simbólico -> Sobremuestreo -> RRC Tx -> Canal
-> RRC Rx / Filtro acoplado -> Muestreo -> Decisión
-> Bits recuperados -> Descrambling -> Fuente recuperada
```

### Orden recomendado en la app

1. **Lab 1 - Formateo**
   - elegir fuente `audio` o `text`
   - fijar `fs`, `n_bits`, cuantizador
   - ejecutar

2. **Lab 2 - Modulación**
   - usar la salida real de Formateo
   - elegir `BPSK`, `QPSK` o `M-PSK`
   - definir `sps`, `roll-off`, `span`
   - ejecutar

3. **Lab 3 - Canal y Rx**
   - elegir la carpeta de salida de Lab 2
   - seleccionar canal `ideal` o `AWGN`
   - definir barrido de `Eb/N0` o punto único
   - ejecutar

---

## Estructura de carpetas

### Código principal

- [`app/app.py`](app/app.py): servidor Flask y rutas
- [`src/main.py`](src/main.py): pipeline de Formateo
- [`src/lab2_rrc.py`](src/lab2_rrc.py): Modulación y figuras de RRC
- [`src/lab3_demod.py`](src/lab3_demod.py): Canal, Rx, BER y recuperación

### Templates

- [`app/templates/lab1.html`](app/templates/lab1.html)
- [`app/templates/lab2.html`](app/templates/lab2.html)
- [`app/templates/lab3.html`](app/templates/lab3.html)

### Datos de ejemplo

- [`data/`](data)

### Resultados

- [`outputs_ui/lab1/`](outputs_ui/lab1)
- [`outputs_ui/lab2/`](outputs_ui/lab2)
- [`outputs_ui/lab3/`](outputs_ui/lab3)

Cada corrida crea una subcarpeta con timestamp:

```text
outputs_ui/lab2/20260410_123456/
```

Ahí se guardan:
- `params.json`
- figuras
- binarios
- informes
- audio/texto recuperado si corresponde

---

## Archivos importantes que genera cada etapa

### Lab 1
- `bits.bin`
- `bits_tx.bin`
- figuras de histograma, entropía, SQNR, MSE, PCM16
- informe `.md` y `.pdf`

### Lab 2
- `iq.bin` / `iq_tx.bin`
- `bits.bin`, `bits_tx.bin`, `bits_from_lab1.bin`
- `params.json`
- `constellation_symbols.png`
- `constellation_shaped.png`
- `spectrum.png`
- `rrc_impulse.png`
- `eye_diagram.png`

### Lab 3
- `ber_curve.png`
- `ber_results.csv`
- `rx_constellation.png`
- `rx_eye.png`
- `rx_time.png`
- `rx_decision.png`
- `tx_rx_constellations.png`
- `ber_point.png`
- `audio_rx.wav` o `text_rx.txt`
- `audio_tx_ref.wav` o `text_tx_ref.txt`

---

## Formato de la señal modulada

La salida de Modulación se guarda como un **vector IQ complejo** en banda base equivalente.

En disco queda como archivo binario `float32` intercalado:

```text
I0, Q0, I1, Q1, I2, Q2, ...
```

Los archivos usados por Lab 3 son:

- `iq.bin`
- o `iq_tx.bin`

Lab 3 reconstruye el vector complejo a partir de ese archivo y lo usa como entrada del canal.

---

## Canal ideal y AWGN

En Lab 3 hay dos modos:

### Canal ideal

```text
r[n] = s[n]
```

No se agrega ruido.

### Canal AWGN

```text
r[n] = s[n] + w[n]
```

Se agrega ruido blanco gaussiano según el `Eb/N0` elegido.

---

## Curva BER

La curva BER se construye barriendo varios valores de `Eb/N0`.

Para cada punto:

1. se toma la señal IQ transmitida
2. se la pasa por el canal
3. se aplica el filtro acoplado
4. se muestrea
5. se decide
6. se comparan bits transmitidos y recuperados

La BER simulada se calcula como:

```text
BER = N_errores / N_bits
```

La app también grafica referencias teóricas en AWGN para:

- `BPSK`
- `QPSK`
- `M-PSK` (por ejemplo `8-PSK` y `16-PSK`)

---

## Cómo recuperar audio o texto en recepción

Si la corrida original viene de:

- `source = audio`, Lab 3 genera:
  - `audio_rx.wav`
  - `audio_tx_ref.wav`

- `source = text`, Lab 3 genera:
  - `text_rx.txt`
  - `text_tx_ref.txt`

La reconstrucción se hace **después** del canal, del filtro acoplado, del muestreo y de la decisión.

---

## Consejos de uso

- Para una corrida simple de Rx, elegí directamente la carpeta timestamp final de Lab 2.
- Si querés usar una carpeta base con muchas corridas, seleccioná la **subcarpeta** desde Lab 3.
- Si cambiás templates o backend y el navegador sigue mostrando algo viejo, recargá fuerte la página o reiniciá Flask.
- Si querés una reconstrucción casi perfecta para validar la cadena, probá primero con **canal ideal**.

---

## Problemas comunes

### 1. No aparece la última corrida
- Revisá que la corrida exista dentro de `outputs_ui/labX/`
- Usá `Actualizar lista`
- O elegí la carpeta manualmente

### 2. No aparece audio o texto recuperado
- Asegurate de que la corrida de Lab 2 venga realmente de Lab 1
- Revisá que exista `params.json` con metadata de `lab1`
- Volvé a correr Lab 2 y luego Lab 3 si la corrida era vieja

### 3. La página sigue mostrando valores viejos
- Reiniciá el servidor Flask
- Recargá fuerte el navegador

### 4. Cancelar una simulación no responde
- Esperá unos segundos: la cancelación es cooperativa
- La corrida se corta en los puntos de control del barrido/trials

---

## Desarrollo

### Ejecutar compilación rápida de verificación

```bash
python3 -m compileall app/app.py
python3 -m compileall src/lab2_rrc.py
python3 -m compileall src/lab3_demod.py
```

### Estado del repo

El proyecto se está usando para una entrega real, así que conviene:

- correr siempre en un entorno virtual
- conservar `outputs_ui/` para trazabilidad
- evitar borrar corridas útiles antes de generar informes finales

---

## Resumen rápido para levantarlo desde cero

```bash
git clone https://github.com/ecoranti/CDA.git
cd CDA
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app/app.py
```

Abrir:

- [http://127.0.0.1:5001](http://127.0.0.1:5001)

Y ejecutar:

1. Formateo
2. Modulación
3. Canal y Rx
