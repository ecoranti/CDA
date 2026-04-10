# Notas Examen Oral — CDA
> Fuente: grabación de audio de compañero. Examen aprobado con éxito (~40 diapositivas, explicación muy detallada).

---

## 1. Error de cuantización

- El error de cuantización `Δ` (paso de cuantización) se define como `Δ = (xmax - xmin) / L`, donde `L = 2^n` es el número de niveles.
- La **única forma de reducir el error** (achicar `Δ`) manteniendo el mismo rango dinámico `[xmin, xmax]` es **aumentar L** (agregar más niveles, es decir, más bits por muestra).
- **Consecuencia directa:** aumentar `L` implica:
  - Más bits por muestra (`n = log2(L)`)
  - Mayor tasa de bits `Rb = fs · n`
  - Más señales a transmitir → mayor ancho de banda requerido o menor velocidad de símbolo

**Tradeoff clave:** precisión de cuantización vs. carga en el canal.

---

## 2. Fuentes de información — Labs 1, 2 y 3

Presentar los tres laboratorios como **etapas de un sistema de comunicaciones completo**:

- **Lab 1 – Formateo:** fuente de texto y audio → muestreo → cuantización → bits → scrambling
- **Lab 2 – Modulación:** bits → mapeo a símbolos → conformación de pulso (RRC) → señal IQ
- **Lab 3 – Canal y Receptor:** canal AWGN → filtro acoplado RRC → muestreo → decisión → BER

Para cada elección de diseño (QPSK, µ-law, span, rolloff, sps, etc.) **desarrollar los tradeoffs explícitamente** en el informe. No alcanza con decir qué se eligió; hay que justificar *por qué*.

---

## 3. Scrambling y entropía

### ¿Por qué buscamos alta entropía?

- Una fuente con entropía máxima (`H = 1 bit/bit`) tiene bits equiprobables: `P(0) = P(1) = 0.5`.
- Esto es deseable porque los algoritmos de modulación, codificación y sincronización asumen esta condición.

### ¿Qué pasaría sin scrambling, con entropía baja?

- Si `P(0) >> P(1)` (o viceversa), la señal modulada tiene **energía desequilibrada** → el receptor puede tener dificultades para recuperar la portadora o el reloj (problemas de sincronización).
- En BPSK/QPSK, una secuencia de muchos ceros o unos consecutivos genera largas rachas sin transiciones → el PLL del receptor pierde la referencia de fase.
- La curva BER real se aleja de la teórica porque la señal ya no cumple las hipótesis del modelo gaussiano.

### ¿Cómo lo manejamos sin scrambling?

- Se puede usar **codificación diferencial** (DBPSK, DQPSK) para no depender de la fase absoluta.
- O aplicar un código de línea que garantice balance DC (ej. 8B/10B).
- En nuestro caso: usamos un **LFSR** (registro de desplazamiento con retroalimentación lineal) con semilla `0b1010110011` y taps `(9,6)`, que genera una secuencia pseudoaleatoria de período `2^10 - 1 = 1023` con la que se hace XOR bit a bit → la distribución de bits converge a `P(0) ≈ P(1) ≈ 0.5`.

### Detallar en el informe:
- El polinomio generador del LFSR y por qué garantiza maximal-length sequence (m-sequence)
- Por qué la m-sequence tiene `P(1) = (2^n/2) / (2^n - 1) ≈ 0.5`
- Cómo se descrambliza en el receptor (mismo LFSR, misma semilla, sincronizado)

---

## 4. Filtros RRC — Por qué usamos DOS

### El coseno alzado completo (RC)

El pulso de Nyquist que elimina ISI es el **coseno alzado (Raised Cosine)**. Su transformada de Fourier es:

```
H_RC(f) = H_RRC(f) · H_RRC(f)  →  h_RC[n] = h_RRC[n] * h_RRC[n]
```

Se divide la raíz entre TX y RX: cada uno aplica un **RRC (Root Raised Cosine)**, y la cascada produce el RC completo. Si se usara un solo filtro RC en el TX, el receptor no tendría filtrado acoplado y la SNR no sería máxima.

### Por qué usar 2 RRC (no 1 RC en TX solamente):

1. **Filtro acoplado óptimo:** el filtro acoplado al pulso `h_TX` es `h_TX*(-t)`. Como el RRC es simétrico, el filtro acoplado al RRC es el propio RRC → `h_RC = h_RRC * h_RRC`.
2. **Máxima SNR en el instante de muestreo:** el teorema del filtro acoplado garantiza que la SNR a la salida del receptor es máxima cuando `h_RX = h_TX*`.
3. **ISI nula:** los cruces por cero del RC ocurren en múltiplos de `Ts` → en los instantes de muestreo óptimos no hay interferencia entre símbolos.

### Cruces por cero del RC:

El pulso RC tiene la propiedad de Nyquist: `h_RC(nTs) = δ[n]`, es decir vale 1 en `t=0` y **exactamente 0 en todos los demás instantes de muestreo** `t = ±Ts, ±2Ts, ...` → ISI = 0.

### Efecto de cada parámetro — justificar en el informe:

| Parámetro | Efecto | Elección típica |
|-----------|--------|-----------------|
| **rolloff α** | α=0 → mínimo BW pero pulso infinito; α=1 → doble BW pero pulso compacto. α intermedio balancea BW vs. sensibilidad al timing. | α = 0.25 → buen balance |
| **span** | Cuántos símbolos dura el filtro. Más span = mejor aproximación teórica del RRC, menos ISI residual, pero más latencia y más cómputo. | span = 8 → suficiente para α=0.25 |
| **sps** | Muestras por símbolo. Mayor sps = mejor representación del pulso analógico, pero mayor carga computacional. | sps = 8 |

**Hacer un barrido de span con ISI:** comparar BER o ISI residual para span = 4, 6, 8, 10, 12 y justificar la elección. Con α=0.25, span=8 es generalmente suficiente para que la energía fuera del span sea despreciable.

---

## 5. Justificación del µ (mu-law)

- El parámetro µ controla la **no-linealidad del compandor**: a mayor µ, más compresión de amplitudes altas y más expansión de amplitudes bajas.
- La justificación óptima de µ se puede hacer **maximizando la SQNR** (Signal-to-Quantization-Noise Ratio):
  - Para señales con distribución Laplaciana o gaussiana (típico en voz), la SQNR uniforme varía mucho con la amplitud de la señal.
  - El compandor µ-law aplana la curva de SQNR vs. nivel de entrada → **SQNR aproximadamente constante** en un rango amplio de amplitudes.
- **Procedimiento para justificar µ = 255:** calcular la SQNR en función de µ para el archivo de voz real y mostrar que µ = 255 maximiza (o está en el máximo) la SQNR promedio o la SQNR mínima sobre el rango dinámico de la señal. El valor µ = 255 es el estándar ITU-T G.711 (telefonía PCM).

---

## 6. Por qué usamos seed

- La **semilla (seed)** del generador de números pseudoaleatorios fija la secuencia de bits de prueba y el ruido AWGN generado.
- Con la misma seed, **cualquier persona que reproduzca la simulación obtiene exactamente los mismos resultados** → reproducibilidad científica.
- Permite comparar resultados entre diferentes modulaciones, parámetros o implementaciones bajo exactamente las mismas condiciones.
- También permite separar "mala suerte estadística" de diferencias reales de rendimiento.

---

## 7. Si usáramos un solo filtro RRC → regla MAP en el receptor

Si en el TX se aplica un filtro RC completo (no dividido) y en el RX no hay filtro:
- El receptor no es el filtro acoplado óptimo → SNR subóptima.
- Si además los bits no son equiprobables (sin scrambling), el umbral de decisión óptimo ya no es cero.

**Regla MAP (Maximum A Posteriori):**

```
Decidir s1 si:  P(s1|z) > P(s2|z)
```

Aplicando Bayes:

```
Umbral MAP:  z > (N0/2) · ln[P(s2)/P(s1)]  +  (E_s1 + E_s2)/2
```

- Si `P(s1) = P(s2)` → el log es 0 → MAP coincide con ML → umbral en 0 (nuestro caso actual).
- Si `P(0) ≠ P(1)` (baja entropía, sin scrambling) → el umbral se desplaza → hay que calcularlo y ajustarlo en el receptor para no degradar la BER.

**Conclusión:** el scrambling + doble RRC + hard decision con umbral en 0 es el diseño completo y correcto. Si se elimina cualquiera de las tres piezas, hay que compensar en otra parte.

---

## 8. Basar el trabajo en una normativa

Elegir una norma de referencia para justificar los parámetros del sistema, por ejemplo:

- **ITU-T G.711** (PCM de voz, 8 kHz, 8 bits/muestra, µ=255) → justifica `fs = 8 kHz`, `n = 8 bits`, `µ = 255`
- **DVB, LTE, WiFi (802.11)** → si se quiere justificar el ancho de banda y la modulación

**Procedimiento:**

1. Partir del ancho de banda disponible `B` (dato de la norma o supuesto)
2. Con α elegido: `B = Rs · (1 + α) / 2` → despejar `Rs = 2B / (1 + α)`
3. Con la modulación elegida (QPSK: 2 bits/símbolo): `Rb = Rs · log2(M) = Rs · 2`
4. Verificar que `Rb ≥ fs · n` (la tasa de bits del canal soporta la fuente cuantizada)
5. Ajustar `fs`, `n`, o `M` si no cierra

---

## 9. Código Gray — qué es y cómo lo aplicamos

### Qué es

El Código Gray es un sistema de numeración binaria donde **dos valores consecutivos difieren en exactamente 1 bit**. Ejemplo para 2 bits:

```
Decimal  Binario natural  Gray
  0          00            00
  1          01            01
  2          10            11
  3          11            10
```

### Por qué se usa en comunicaciones

En QPSK o QAM, cuando el receptor comete un error de símbolo, lo más probable es que confunda el símbolo recibido con uno **vecino** en la constelación. Con Gray coding, cada par de vecinos difiere en 1 bit → **1 error de símbolo = 1 error de bit** → se minimiza la BER.

Sin Gray coding, una confusión entre símbolos no-adyacentes podría causar 2 errores de bit (en QPSK) → BER real peor que la teórica.

### Cómo lo aplicamos en el proyecto

En `lab2_rrc.py`, función `map_bits_to_symbols` para QPSK:

```python
I = np.where(b0 == 0, 1.0, -1.0)   # b0=0 → +1,  b0=1 → -1
Q = np.where(b1 == 0, 1.0, -1.0)   # b1=0 → +1,  b1=1 → -1
```

El mapeo resultante es Gray: símbolos adyacentes (misma I, Q diferente o viceversa) difieren en exactamente 1 bit. Esto permite que la curva BER de QPSK con Gray coincida con la de BPSK: `Pb = ½·erfc(√(Eb/N0))`.

---

## 10. Hard Decision (Decisión Dura)

### Qué es

La **decisión dura** es cuando el receptor toma una decisión **binaria definitiva e inmediata** sobre cada muestra recibida:

```
z > umbral  →  decide símbolo A  (bit 1 o bit 0)
z ≤ umbral  →  decide símbolo B
```

No se retiene información sobre qué tan cerca estuvo la muestra del umbral. Se "tira" toda la información de confiabilidad.

La alternativa es la **decisión blanda (soft decision)**: en lugar de decidir, se pasa el valor de la métrica (o el LLR, Log-Likelihood Ratio) a un decodificador de canal (ej. Turbo, LDPC, Viterbi). La ganancia de soft decision sobre hard decision es típicamente ~2 dB en BER.

### Cómo lo aplicamos en el proyecto

En `lab3_demod.py`:

```python
def demap_bpsk(samples):
    return np.where(samples.real > 0, 1, 0)  # umbral = 0

def demap_qpsk(samples):
    bits_0 = np.where(samples.real > 0, 0, 1)  # componente I
    bits_1 = np.where(samples.imag > 0, 0, 1)  # componente Q
```

El umbral en `0` es el **umbral ML óptimo** bajo las condiciones del sistema (AWGN, símbolos equiprobables, energías iguales). Es el punto donde `P(s1|z) = P(s2|z)` y corresponde exactamente al punto de cruce de las dos gaussianas condicionales `f(z|s1)` y `f(z|s2)`.

### Justificación del umbral en 0

Para BPSK: `s1 = +√Es`, `s2 = -√Es`. La verosimilitud de cada símbolo es:

```
f(z | s_i) = (1/√(πN0)) · exp(-(z - s_i)² / N0)
```

La regla ML elige el `si` que maximiza esta expresión. El umbral es el punto donde `f(z|s1) = f(z|s2)`, que se resuelve en `z = 0`.

---

## Resumen — Preguntas frecuentes del oral

| Pregunta | Respuesta clave |
|----------|-----------------|
| ¿Por qué µ = 255? | Estándar G.711; maximiza SQNR en rango dinámico de voz |
| ¿Por qué 2 RRC y no 1 RC? | Filtro acoplado óptimo → máxima SNR en el muestreo |
| ¿Por qué scrambling? | Equiprobabilidad de bits → evitar problemas de sync y mantener umbral en 0 |
| ¿Por qué Gray code? | 1 error de símbolo = 1 error de bit → BER mínima |
| ¿Por qué hard decision con umbral en 0? | Criterio ML óptimo con bits equiprobables bajo AWGN |
| ¿Qué pasa si baja entropía sin scrambling? | Umbral óptimo ya no es 0 → hay que usar MAP con umbral desplazado |
| ¿Por qué seed fija? | Reproducibilidad de la simulación |
| ¿Qué es el span? | Duración del filtro en símbolos; barrido para encontrar span mínimo sin ISI significativa |
