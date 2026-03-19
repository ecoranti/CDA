# Plantilla de Artículo Técnico (Canal y Rx)

Formato sugerido: IEEE o APA, 4 a 6 páginas.

## Título
Demodulación Digital en Canal AWGN con Filtro Acoplado y Detección Bayesiana: Evaluación Experimental de BER Encadenada con un Transmisor RRC

## Autores
[Nombre 1], [Nombre 2], [Nombre 3]

## Resumen
Este artículo presenta la implementación y evaluación de una cadena de comunicación digital completa, encadenando la salida del transmisor baseband (Modulación) con el receptor digital (Canal y Rx). Se modela un canal AWGN sobre señales IQ reales de transmisión, se aplica filtro acoplado RRC, muestreo óptimo y detección ML/MAP para reconstrucción de bits. Se estima la BER experimental en función de Eb/N0 y se compara con la referencia teórica de BPSK/QPSK coherente. Se incluyen estimaciones de Eb/N0 efectivo y bandas de confianza por Monte Carlo para validar la consistencia estadística de los resultados.

**Palabras clave:** comunicación digital, AWGN, filtro acoplado, BER, Eb/N0, ML/MAP.

## 1. Introducción
- Motivación del problema.
- Importancia de cerrar la cadena Tx-Rx.
- Objetivo del trabajo.

## 2. Marco Teórico
### 2.1 Sistema transmisor-receptor
- Pulso RRC en transmisión.
- Filtro acoplado en recepción.

### 2.2 Canal AWGN
- Modelo: `r[n] = s[n] + w[n]`.

### 2.3 Detección Bayesiana (ML/MAP)
- Regla de decisión por umbral para BPSK/QPSK.

### 2.4 BER Teórica
- `Pb = 0.5 * erfc(sqrt(Eb/N0))` para BPSK/QPSK coherente.

## 3. Metodología
### 3.1 Flujo experimental encadenado
1. Generación de bits y señal IQ en Modulación.
2. Exportación (`iq.bin`, bits de referencia, `params.json`).
3. Ingreso de esos archivos al Canal y Rx.
4. Barrido de Eb/N0 con Monte Carlo por punto.

### 3.2 Parámetros
- Modulación: [...]
- `sps`: [...]
- `alpha`: [...]
- `span`: [...]
- Rango Eb/N0: [...]
- Trials por Eb/N0: [...]

### 3.3 Métricas
- BER experimental.
- BER teórica.
- `Eb/N0` estimado (media y desvío).
- Intervalo de confianza 95% de BER.

## 4. Resultados
### 4.1 Caracterización del receptor
- Figura: `rx_time.png`
- Figura: `rx_eye.png`
- Figura: `rx_constellation.png`

### 4.2 Filtro acoplado
- Figura: `mf_impulse.png`
- Figura: `mf_freq.png`

### 4.3 Decisión y constelaciones comparativas
- Figura: `rx_decision.png`
- Figura: `tx_rx_constellations.png`

### 4.4 Curva BER
- Figura: `ber_curve.png`
- Tabla desde `ber_results.csv`.

## 5. Discusión
- Diferencias entre teoría y simulación.
- Efecto de `alpha`, `sps`, `span`.
- Compromiso ancho de banda vs robustez al ruido.
- Reproducibilidad y estabilidad con Monte Carlo.

## 6. Conclusiones
- Síntesis de hallazgos.
- Alcance de la validación experimental.
- Propuestas de mejora.

## 7. Referencias
- Proakis, J. G., *Digital Communications*.
- Sklar, B., *Digital Communications*.
- Guía de Laboratorio N°3, cátedra.

---

## Notas para completar rápido
- Usar el PDF automático del proyecto (`informe_lab3.pdf`) como fuente de figuras.
- Convertir la tabla de `ber_results.csv` a formato del journal.
- Mantener consistencia de nomenclatura (`Eb/N0`, `BER`, `RRC`, `MF`).

