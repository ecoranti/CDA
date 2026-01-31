from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import argparse
from pathlib import Path
from dataclasses import dataclass

from . import lab2_rrc

@dataclass
class Lab3Params:
    out_dir: str
    n_bits: int = 10000
    modulation: str = "QPSK"
    sps: int = 8
    rolloff: float = 0.25
    span: int = 8
    ebn0_start: float = 0.0
    ebn0_end: float = 12.0
    ebn0_step: float = 2.0
    seed: int = 0

def add_noise(signal: np.ndarray, ebn0_db: float, sps: int, bits_per_symbol: int) -> np.ndarray:
    """
    Agrega ruido AWGN complejo a la señal.
    
    Eb/N0 = (Es/N0) / bits_per_symbol
    Es/N0 = Eb/N0 * bits_per_symbol
    
    SNR_linear = Es/N0 / sps (pues la energía se reparte en sps muestras si consideramos potencia constante)
    Sin embargo, en simulación banda base con RRC unitario:
       Es = sum(|h|^2) = 1 (aprox) si h está normalizado. 
       Pero 'signal' aquí son muestras. 
    
    Criterio estándar para simulaciones de este tipo:
    - Señal tiene potencia promedio P_s.
    - Ruido tiene potencia P_n.
    - SNR = P_s / P_n.
    
    Relación con Eb/N0:
      SNR = (Eb * Rb) / (N0 * W)
      Para pulso Nyquist, W ~ Rs (ancho de banda de símbolo).
      Entonces SNR ~ (Eb/N0) * (Rb/Rs) = (Eb/N0) * bits_per_symbol.
      
      Pero estamos operando a 'sps' muestras por símbolo. El ancho de banda de la simulación es fs = sps * Rs.
      La potencia de ruido debe definirse en fs.
      
      Es/N0 = (Eb/N0)_dB + 10*log10(k)   [k = bits/simbolo]
      SNR_sim = Es/N0 - 10*log10(sps)
      
    Args:
        signal: señal compleja transmitida.
        ebn0_db: Eb/N0 deseado en dB.
        sps: muestras por símbolo.
        bits_per_symbol: k (1 para BPSK, 2 para QPSK).
        
    Returns:
        Señal con ruido.
    """
    # 1. Calcular potencia de señal
    p_signal = np.mean(np.abs(signal)**2)
    
    # 2. Calcular SNR linear requerido
    # Es/N0_dB = Eb/N0_dB + 10*log10(k)
    esn0_db = ebn0_db + 10 * np.log10(bits_per_symbol)
    
    # SNR_dB = Es/N0_dB - 10*log10(sps)
    snr_db = esn0_db - 10 * np.log10(sps)
    snr_linear = 10**(snr_db / 10.0)
    
    # 3. Calcular potencia de ruido
    p_noise = p_signal / snr_linear
    
    # 4. Generar ruido complejo
    # Potencia total p_noise. Parte real p_noise/2, imag p_noise/2.
    noise = np.sqrt(p_noise/2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    
    return signal + noise

def matched_filter(rx_signal: np.ndarray, sps: int, alpha: float, span: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Aplica filtro acoplado (RRC) a la señal recibida.
    """
    # Obtener taps del transmisor (mismo RRC)
    h = lab2_rrc.rrc_taps(alpha, sps, span)
    
    # Convolución. 'same' preserva longitud, pero hay que cuidar el delay.
    # Usaremos 'full' y recortaremos como en lab2 para consistencia, aunque
    # en Rx lo importante es encontrar el instante de muestreo óptimo.
    y = np.convolve(rx_signal, h, mode='full')
    
    return y, h

def downsample(filtered_signal: np.ndarray, sps: int, delay_samples: int, n_symbols: int) -> np.ndarray:
    """
    Muestrea la señal filtrada en los instantes óptimos.
    
    Args:
        filtered_signal: salida del MF.
        sps: steps per symbol.
        delay_samples: retardo total acumulado (Tx + Rx filter).
                       Tx e Rx son RRC iguales. 
                       Si Tx delay = D, Rx delay = D. Total 2D.
        n_symbols: número de símbolos esperados.
        
    Returns:
        Muestras a tasa de símbolo.
    """
    # El pico central del pulso completo (RC) ocurre en delay_samples.
    # Muestreamos ahí y luego cada sps.
    start = delay_samples
    # Generar índices
    idxs = np.arange(start, len(filtered_signal), sps)
    
    # Recortar al número de símbolos enviados
    if len(idxs) > n_symbols:
        idxs = idxs[:n_symbols]
        
    samples = filtered_signal[idxs]
    return samples

def demap_qpsk(samples: np.ndarray) -> np.ndarray:
    """
    Demodulación QPSK (Hard Decision).
    Mapeo Gray esperado: 
      Re > 0, Im > 0 -> 11 (+1+1j) ??
      Depende del mapeo en Tx.
      
      En lab2_rrc.py, QPSK map:
        b0=0 -> I=1, b0=1 -> I=-1
        b1=0 -> Q=1, b1=1 -> Q=-1
        (I + 1j*Q)/sqrt(2)
        
      Rx Decision:
        I > 0 -> b0=0
        I < 0 -> b0=1
        Q > 0 -> b1=0
        Q < 0 -> b1=1
    """
    # Escalar o usar signo directo.
    # b0: 0 si Real > 0, 1 si Real <= 0
    bits_0 = np.where(samples.real > 0, 0, 1)
    bits_1 = np.where(samples.imag > 0, 0, 1)
    
    # Intercalar
    out = np.empty(2 * len(samples), dtype=np.uint8)
    out[0::2] = bits_0
    out[1::2] = bits_1
    return out

def demap_bpsk(samples: np.ndarray) -> np.ndarray:
    """
    BPSK Map en lab2_rrc:
       0 -> -1
       1 -> +1 (ojo: lab2 dice s = 2*bits - 1. Si bit=1 -> s=1. Si bit=0 -> s=-1)
       
    Rx Decision:
       Real > 0 -> 1
       Real < 0 -> 0
    """
    return np.where(samples.real > 0, 1, 0).astype(np.uint8)

def calculate_ber(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
    # Alinear longitudes si difieren por algun simbolo
    L = min(len(tx_bits), len(rx_bits))
    tx = tx_bits[:L]
    rx = rx_bits[:L]
    errors = np.sum(tx != rx)
    return float(errors) / L

def theoretical_ber_qpsk(ebn0_db_arr):
    # Pb_QPSK = Q(sqrt(2*Eb/N0))
    # Igual a BPSK coherente
    ebn0_lin = 10**(np.array(ebn0_db_arr)/10.0)
    return 0.5 * erfc(np.sqrt(ebn0_lin))

def run_simulation(params: Lab3Params):
    print(f"Iniciando simulación Lab3. Mod={params.modulation}, n_bits={params.n_bits}...")
    
    # 1. Generar bits y Tx (usando lab2_rrc)
    # Necesitamos reproducir la cadena Tx internamente
    rng = np.random.default_rng(params.seed)
    bits_tx = rng.integers(0, 2, size=params.n_bits, dtype=np.uint8)
    
    # params.modulation -> symbol mapping
    syms_tx = lab2_rrc.map_bits_to_symbols(bits_tx, params.modulation)
    
    # Tx Filter
    # Ojo: upsample_and_filter de lab2 devuelve señal filtrada y normalizada
    # y = convolve(up, h). 
    taps_tx = lab2_rrc.rrc_taps(params.rolloff, params.sps, params.span)
    tx_signal = lab2_rrc.upsample_and_filter(syms_tx, params.sps, taps_tx)
    
    # Calcular retardo de filtro Tx (samples)
    # rrc_taps genera 2*span*sps + 1 taps (aprox). Center at index len//2.
    delay_tx = (len(taps_tx) - 1) // 2
    
    results_ebn0 = []
    results_ber = []
    
    ebn0_vals = np.arange(params.ebn0_start, params.ebn0_end + 0.1, params.ebn0_step)
    
    bps = 2 if params.modulation == "QPSK" else 1
    
    for ebn0 in ebn0_vals:
        # 2. Canal AWGN
        # Necesitamos pasar sps y bps para escalar ruido correctamente
        rx_noisy = add_noise(tx_signal, ebn0, params.sps, bps)
        
        # 3. Rx Matched Filter
        # Filtro identico al Tx
        rx_filtered, taps_rx = matched_filter(rx_noisy, params.sps, params.rolloff, params.span)
        
        # Delay total = delay_rx (ya que lab2_rrc elimina el delay de Tx)
        delay_rx = (len(taps_rx) - 1) // 2
        total_delay = delay_rx
        
        # 4. Downsampling
        rx_syms = downsample(rx_filtered, params.sps, total_delay, len(syms_tx))
        
        # 5. Demap / Decision
        if params.modulation == "QPSK":
            bits_rx = demap_qpsk(rx_syms)
        else:
            bits_rx = demap_bpsk(rx_syms)
            
        # 6. BER
        ber = calculate_ber(bits_tx, bits_rx)
        results_ebn0.append(ebn0)
        results_ber.append(ber)
        
        print(f"Eb/N0 = {ebn0:5.1f} dB | BER = {ber:.2e} | (TxSyms={len(syms_tx)}, RxSyms={len(rx_syms)})")
        
    # 7. Graficar y guardar
    Path(params.out_dir).mkdir(parents=True, exist_ok=True)
    
    # Curva teórica
    ebn0_fine = np.linspace(params.ebn0_start, params.ebn0_end, 100)
    ber_theory = theoretical_ber_qpsk(ebn0_fine) # Valido para BPSK también
    
    plt.figure()
    plt.semilogy(ebn0_fine, ber_theory, 'k-', label='Teórica (BPSK/QPSK)')
    plt.semilogy(results_ebn0, results_ber, 'bo--', label=f'Simulada ({params.modulation})')
    plt.grid(True, which='both', alpha=0.3)
    plt.xlabel('Eb/N0 [dB]')
    plt.ylabel('BER (Bit Error Rate)')
    plt.title(f'Curva BER vs Eb/N0 - {params.modulation}')
    plt.legend()
    plt.ylim(bottom=1e-6)
    
    plot_path = Path(params.out_dir) / "ber_curve.png"
    plt.savefig(plot_path, dpi=140)
    plt.close()
    
    # Guardar CSV
    import csv
    with open(Path(params.out_dir) / "ber_results.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["EbN0_dB", "BER_Sim", "BER_Theory"])
        for e, b in zip(results_ebn0, results_ber):
            # Calcular teórico puntual para el CSV
            th = theoretical_ber_qpsk([e])[0]
            writer.writerow([e, b, th])
            
    print(f"Resultados guardados en {params.out_dir}")
    
    return {
        "out_dir": str(params.out_dir),
        "ber_plot": str(plot_path),
        "ber_csv": str(Path(params.out_dir) / "ber_results.csv"),
        "ber_data": list(zip(results_ebn0, results_ber)),
        "modulation": params.modulation
    }

def run_single(params: Lab3Params, bits: np.ndarray | None = None) -> dict:
    """
    Ejecuta una corrida única con parámetros específicos y genera gráficos detallados.
    Args:
        params: Parámetros (usa ebn0_start como el valor Eb/N0 deseado).
        bits: Bits a transmitir (si None, se generan aleatorios según params.n_bits).
    """
    outp = Path(params.out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    
    # 1. Bits
    if bits is None:
        rng = np.random.default_rng(params.seed)
        bits_tx = rng.integers(0, 2, size=params.n_bits, dtype=np.uint8)
    else:
        bits_tx = bits.astype(np.uint8)
    
    # 2. Tx
    syms_tx = lab2_rrc.map_bits_to_symbols(bits_tx, params.modulation)
    taps_tx = lab2_rrc.rrc_taps(params.rolloff, params.sps, params.span)
    tx_signal = lab2_rrc.upsample_and_filter(syms_tx, params.sps, taps_tx)
    
    # 3. Channel
    # Usamos params.ebn0_start como EL valor de Eb/N0
    ebn0_val = params.ebn0_start
    bps = 2 if params.modulation == "QPSK" else 1
    rx_noisy = add_noise(tx_signal, ebn0_val, params.sps, bps)
    
    # 4. Rx Matched Filter
    rx_filtered, taps_rx = matched_filter(rx_noisy, params.sps, params.rolloff, params.span)
    delay_rx = (len(taps_rx) - 1) // 2
    
    # 5. Downsample
    rx_syms = downsample(rx_filtered, params.sps, delay_rx, len(syms_tx))
    
    # 6. Demod
    if params.modulation == "QPSK":
        bits_rx = demap_qpsk(rx_syms)
    else:
        bits_rx = demap_bpsk(rx_syms)
        
    ber = calculate_ber(bits_tx, bits_rx)
    
    # 7. Graficos
    paths = {}
    
    # Rx Time (un segmento)
    lab2_rrc.plot_time_iq(rx_filtered, params.sps, str(outp / "rx_time.png"), nsamples=200)
    paths["rx_time_png"] = str(outp / "rx_time.png")
    
    # Rx Constellation (muestras en instantes óptimos)
    # Reutilizamos plot_constellation de lab2 pero le pasamos un "falso" sps=1 porque rx_syms ya está muestreado
    # O mejor, graficamos nosotros.
    lab2_rrc.plot_constellation(upsample(rx_syms,1), 1, str(outp / "rx_constellation.png")) # Hacky reuse, or just use matplotlib
    # Mejor usar una implementacion directa para constelacion de simbolos recibidos
    plt.figure()
    plt.scatter(rx_syms.real, rx_syms.imag, s=8, alpha=0.5)
    plt.axhline(0, color='gray', lw=0.6)
    plt.axvline(0, color='gray', lw=0.6)
    lim = np.max(np.abs(rx_syms)) * 1.1 if len(rx_syms)>0 else 1.5
    plt.xlim(-lim, lim); plt.ylim(-lim, lim)
    plt.title(f"Constelación Rx (Eb/N0={ebn0_val} dB)")
    plt.xlabel("I"); plt.ylabel("Q")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outp / "rx_constellation.png", dpi=140)
    plt.close()
    paths["rx_constellation_png"] = str(outp / "rx_constellation.png")
    
    lab2_rrc.plot_eye(rx_filtered, params.sps, str(outp / "rx_eye.png"))
    paths["rx_eye_png"] = str(outp / "rx_eye.png")
    
    # BER Point on Curve
    eb_min, eb_max = 0, max(12, ebn0_val + 2)
    eb_arr = np.linspace(eb_min, eb_max, 100)
    ber_theo = theoretical_ber_qpsk(eb_arr)
    
    plt.figure()
    plt.semilogy(eb_arr, ber_theo, 'k-', label='Teórica')
    plt.semilogy([ebn0_val], [ber], 'ro', markersize=8, label='Simulada')
    plt.grid(True, which='both', alpha=0.3)
    plt.xlabel('Eb/N0 [dB]')
    plt.ylabel('BER')
    plt.title(f'Punto de Operación (Eb/N0={ebn0_val:.1f})')
    plt.legend()
    plt.ylim(bottom=1e-6)
    plt.tight_layout()
    plt.savefig(outp / "ber_point.png", dpi=140)
    plt.close()
    fig_paths["ber_point_png"] = str(outp / "ber_point.png")
    
    return {
        "ebn0": float(ebn0_val),
        "ber": float(ber),
        "n_errors": int(n_errors),
        "n_bits": int(len(bits_tx)),
        "paths": fig_paths,
        "ok": True
    }


def run_from_file(lab2_dir: str, ebn0: float, out_dir: str = "outputs/lab3_integrated") -> dict:
    """
    Ejecuta el pipeline de recepción utilizando como entrada los archivos generados por Lab 2 (IQ y bits).
    
    Args:
        lab2_dir: Ruta al directorio de salida de Lab 2 (debe contener params.json, iq.bin, bits_from_lab1.bin).
        ebn0: Eb/N0 deseado para la simulación del canal.
        out_dir: Directorio donde guardar resultados de Lab 3.
    """
    lab2_path = Path(lab2_dir)
    params_file = lab2_path / "params.json"
    iq_file = lab2_path / "iq.bin"
    
    # Intentar varios nombres posibles para bits binarios
    bits_file = lab2_path / "bits_from_lab1.bin"
    if not bits_file.exists():
        bits_file = lab2_path / "bits.bin"
        
    if not params_file.exists(): raise FileNotFoundError(f"No se encontró params.json en {lab2_dir}")
    if not iq_file.exists(): raise FileNotFoundError(f"No se encontró iq.bin en {lab2_dir}")
    if not bits_file.exists(): raise FileNotFoundError(f"No se encontró archivo de bits binarios en {lab2_dir}")
    
    # 1. Cargar Parámetros
    import json
    with open(params_file, 'r') as f:
        meta = json.load(f)
        
    # Extraer config de lab2 desde el JSON
    # La estructura puede ser directa o anidada en "lab2"
    l2_conf = meta.get("lab2", meta) 
    
    mod = l2_conf.get("modulation", "QPSK")
    sps = int(l2_conf.get("sps", 8))
    alpha = float(l2_conf.get("rolloff", 0.25))
    span = int(l2_conf.get("span", 8))
    
    # 2. Cargar Datos
    # IQ: float32 interleaved (I, Q)
    raw_iq = np.fromfile(iq_file, dtype=np.float32)
    tx_signal = raw_iq[0::2] + 1j * raw_iq[1::2]
    
    # Bits: uint8 compactados o 1 byte por bit?
    # Lab 1/2 suelen guardar 1 byte por bit si es raw, o packed.
    # Revisando app.py: _write_bits_bin hace bits.tofile(), y bits es np.array(bits, dtype=np.uint8)
    # Entonces es 1 byte por bit (0/1).
    bits_tx = np.fromfile(bits_file, dtype=np.uint8)
    
    # 3. Pipeline Rx
    # Reconstruir params para reutilizar funciones
    # Ojo: run_single crea Tx signal. Aquí ya la tenemos.
    # Adaptaremos el flujo: Add Noise -> Match -> Demod -> BER
    
    # Add Noise
    params_mock = Lab3Params(
        out_dir=out_dir,
        modulation=mod,
        ebn0_start=ebn0,
        ebn0_end=ebn0,
        ebn0_step=1.0,
        n_bits=len(bits_tx),
        sps=sps,
        rolloff=alpha,
        span=span,
        seed=0 # Irrelevante para AWGN (usa random interno)
    )
    
    # Calculamos rate (bits_per_symbol) correctamente
    rate = 1 if mod == "BPSK" else 2
    rx_noisy = add_noise(tx_signal, ebn0, sps, rate)
    
    # Matched Filter
    taps_rx = lab2_rrc.rrc_taps(alpha, sps, span)
    avg_power_tx = np.mean(np.abs(tx_signal)**2) 
    # matched_filter ya hace convolución.
    rx_filtered, _ = matched_filter(rx_noisy, sps, alpha, span)
    
    # Sampling
    delay = (len(taps_rx) - 1) // 2
    n_syms_expected = len(bits_tx) if mod == "BPSK" else len(bits_tx) // 2
    rx_syms = downsample(rx_filtered, sps, delay, n_syms_expected)
    if mod == "QPSK" and len(bits_tx) % 2 != 0: n_syms_expected += 1
        
    # Recortar excedente
    rx_syms = rx_syms[:n_syms_expected]
    
    # Demod
    if mod == "QPSK":
        bits_rx = demap_qpsk(rx_syms)
    elif mod == "BPSK":
        bits_rx = demap_bpsk(rx_syms)
    else:
        raise ValueError(f"Modulación {mod} no soportada")
        
    # Ajustar longitud si hubo padding en Tx (QPSK impar)
    bits_rx = bits_rx[:len(bits_tx)]
    
    # BER
    n_errors = np.sum(bits_tx != bits_rx)
    ber = n_errors / len(bits_tx)
    
    # Gráficos
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig_paths = {}
    
    # BER Point Plot (reutilizando lógica)
    import matplotlib.pyplot as plt
    theoretical_ber = theoretical_ber_qpsk if mod == "QPSK" else None # TODO: BPSK theory
    
    if theoretical_ber:
        ebn0_range = np.linspace(-2, 12, 100)
        ber_theory = theoretical_ber(ebn0_range)
        plt.figure(figsize=(6, 4))
        plt.semilogy(ebn0_range, ber_theory, 'k-', label='Teórica')
        plt.semilogy(ebn0, ber, 'ro', markersize=8, label='Simulada (Lab 2->3)')
        plt.grid(True, which="both", alpha=0.3)
        plt.xlabel('Eb/N0 [dB]')
        plt.ylabel('BER')
        plt.title(f'Punto de Operación (Eb/N0={ebn0})')
        plt.legend()
        plt.ylim(bottom=1e-6)
        out_path = Path(out_dir) / "ber_point.png"
        plt.savefig(out_path, dpi=100, bbox_inches='tight')
        plt.close()
        fig_paths["ber_point_png"] = str(out_path)
    
    # Eye Diagram & Constellation
    # Eye Diagram & Constellation
    # plot_eye espera señal compleja
    lab2_rrc.plot_eye(rx_filtered, sps, str(Path(out_dir)/"rx_eye.png"))
    fig_paths["rx_eye_png"] = str(Path(out_dir)/"rx_eye.png")
    
    # plot_constellation espera (iq, sps, out_path). 
    # rx_syms ya está muestreado (1 muestra/símbolo), así que sps=1.
    lab2_rrc.plot_constellation(rx_syms, 1, str(Path(out_dir)/"rx_constellation.png"))
    fig_paths["rx_constellation_png"] = str(Path(out_dir)/"rx_constellation.png")

    # Time Domain (First 200 samples)
    plt.figure(figsize=(8,3))
    plt.plot(rx_filtered[:200].real, label='I (Rx Filtered)')
    plt.plot(rx_filtered[:200].imag, label='Q (Rx Filtered)', alpha=0.7)
    plt.legend()
    plt.title("Señal Recibida (Filtrada, primeros 200 samples)")
    plt.grid(alpha=0.3)
    out_path = Path(out_dir) / "rx_time.png"
    plt.savefig(out_path, dpi=100)
    plt.close()
    fig_paths["rx_time_png"] = str(out_path)

    return {
        "ebn0": float(ebn0),
        "ber": float(ber),
        "n_errors": int(n_errors),
        "n_bits": int(len(bits_tx)),
        "paths": fig_paths,
        "ok": True,
        "source": "file_integration",
        "debug": {
            "tx_10": bits_tx[:10].tolist(),
            "rx_10": bits_rx[:10].tolist(),
            "len_tx": len(bits_tx),
            "len_rx": len(bits_rx),
            "n_errors": int(n_errors)
        }
    }

def upsample(symbols: np.ndarray, sps: int) -> np.ndarray:
    x = np.zeros(len(symbols) * sps, dtype=np.complex128)
    x[::sps] = symbols
    return x


def main():
    ap = argparse.ArgumentParser("Lab3 - Demodulación Digital")
    ap.add_argument("--out", default="outputs/lab3")
    ap.add_argument("--n_bits", type=int, default=100000)
    ap.add_argument("--mod", default="QPSK", choices=["BPSK", "QPSK"])
    ap.add_argument("--eb_start", type=float, default=0.0)
    ap.add_argument("--eb_end", type=float, default=10.0)
    ap.add_argument("--eb_step", type=float, default=1.0)
    ap.add_argument("--sps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    
    args = ap.parse_args()
    
    p = Lab3Params(
        out_dir=args.out,
        n_bits=args.n_bits,
        modulation=args.mod,
        sps=args.sps,
        ebn0_start=args.eb_start,
        ebn0_end=args.eb_end,
        ebn0_step=args.eb_step,
        seed=args.seed
    )
    
    run_simulation(p)

if __name__ == "__main__":
    main()
