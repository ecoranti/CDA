from __future__ import annotations

import os
import shutil
import threading
import uuid
import subprocess
import tempfile
import json
from pathlib import Path
import sys
from typing import Optional
import re

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, jsonify, send_file

ROOT = Path(__file__).resolve().parent.parent
# Asegurar que la raíz del repo esté en sys.path al ejecutar `python app/app.py`
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Forzar backend no interactivo para Matplotlib (evita GUI en threads)
os.environ.setdefault("MPLBACKEND", "Agg")

# Importar pipelines
from src import main as lab1
from src import report as lab1_report
from src import report_lab3 as lab3_report
from src import sqnr_eval as lab1_sqnr
from src import lab2_rrc
from src.audio_utils import load_wav_mono, uniform_quantize, a_law_quantize, plot_hist_bits
from src.bits_utils import ints_to_bits
from src.scrambling import scramble
from src import lab3_demod


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = "cda-key"

    ROOT = Path(__file__).resolve().parent.parent


    DEFAULTS = {
            "audio": str(ROOT / "data/voice.wav"),
            "text": str(ROOT / "data/sample_text.txt"),
            "bw_hz": 100000,
            "out_lab1": str(ROOT / "outputs_ui/lab1"),
            "out_lab2": str(ROOT / "outputs_ui/lab2"),
            "out_lab3": str(ROOT / "outputs_ui/lab3"),
        }

    OUTPUTS_ROOT = ROOT / "outputs_ui"

    def _normalize_output_base(base: Path) -> Path:
        p = base.expanduser()
        ts_re = re.compile(r"^\d{8}_\d{6}$")
        try:
            resolved_outputs = OUTPUTS_ROOT.resolve()
        except Exception:
            resolved_outputs = OUTPUTS_ROOT
        while ts_re.match(p.name or ""):
            try:
                parent_resolved = p.parent.resolve()
            except Exception:
                parent_resolved = p.parent
            if str(parent_resolved).startswith(str(resolved_outputs)):
                p = p.parent
            else:
                break
        return p

    def _ts_dir(base: Path) -> Path:
        import datetime
        base = _normalize_output_base(base)
        d = base / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        d.mkdir(parents=True, exist_ok=True)
        (d / "figures").mkdir(parents=True, exist_ok=True)
        return d

    def _write_json(path: Path, data: dict):
        import json
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _append_log(path: Path, line: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line.rstrip() + "\n")

    def _safe_alias(src: Path, dst: Path):
        try:
            if not src.exists() or dst.exists():
                return
            shutil.copyfile(src, dst)
        except Exception:
            pass

    def _generate_quantization_quality_plots(out_dir: Path, x, n_bits: int):
        """Genera gráficos comparativos de SQNR/MSE y evolución de ECM (audio)."""
        import numpy as _np
        import matplotlib.pyplot as _plt
        import pandas as _pd

        xhat_uni = lab1_sqnr.eval_uniform(x, bits=n_bits)
        mse_uni = lab1_sqnr.mse(x, xhat_uni)
        sqnr_uni = lab1_sqnr.sqnr_db(x, xhat_uni)

        xhat_al = lab1_sqnr.eval_alaw(x, A=87.6, bits=n_bits)
        mse_al = lab1_sqnr.mse(x, xhat_al)
        sqnr_al = lab1_sqnr.sqnr_db(x, xhat_al)

        labels = ["Uniforme", "A-law (87.6)"]
        sqnrs_plot = [sqnr_uni, sqnr_al]
        mses_plot = [mse_uni, mse_al]

        _plt.figure()
        _plt.bar(range(len(sqnrs_plot)), sqnrs_plot, color=["#64748b", "#2563eb"])
        _plt.xticks(range(len(labels)), labels)
        _plt.ylabel("SQNR (dB)")
        _plt.title("Comparación de SQNR ({} bits)".format(n_bits))
        _plt.tight_layout()
        _plt.savefig(out_dir / "sqnr_comparacion.png", dpi=140)
        _plt.close()

        _plt.figure()
        _plt.bar(range(len(mses_plot)), mses_plot, color=["#64748b", "#2563eb"])
        _plt.xticks(range(len(labels)), labels)
        _plt.ylabel("ECM")
        _plt.title("Comparación de ECM ({} bits)".format(n_bits))
        _plt.tight_layout()
        _plt.savefig(out_dir / "mse_comparacion.png", dpi=140)
        _plt.close()

        # Evolución del ECM acumulado: ECM[n] = (1/n) * sum_{k=1..n} e[k]^2
        e_uni2 = _np.square(_np.asarray(x, dtype=_np.float64) - _np.asarray(xhat_uni, dtype=_np.float64))
        e_al2 = _np.square(_np.asarray(x, dtype=_np.float64) - _np.asarray(xhat_al, dtype=_np.float64))
        n = _np.arange(1, len(e_uni2) + 1, dtype=_np.float64)
        ecm_uni = _np.cumsum(e_uni2) / n
        ecm_al = _np.cumsum(e_al2) / n

        max_points = 2500
        step = max(1, len(n) // max_points)
        idx = _np.arange(0, len(n), step, dtype=int)
        if idx[-1] != len(n) - 1:
            idx = _np.append(idx, len(n) - 1)

        _plt.figure()
        _plt.plot(n[idx], ecm_uni[idx], label="Uniforme", color="#64748b")
        _plt.plot(n[idx], ecm_al[idx], label="A-law (87.6)", color="#2563eb")
        _plt.xlabel("Cantidad de muestras procesadas")
        _plt.ylabel("ECM acumulado")
        _plt.title("Evolución del ECM acumulado")
        _plt.grid(True, alpha=0.3)
        _plt.legend()
        _plt.tight_layout()
        _plt.savefig(out_dir / "ecm_evolucion.png", dpi=140)
        _plt.close()

        rows_sqnr = list(zip(labels, sqnrs_plot, mses_plot))
        df_sqnr = _pd.DataFrame(rows_sqnr, columns=["Cuantizador", "SQNR (dB)", "MSE"])
        df_sqnr.to_csv(out_dir / "sqnr_mse_resumen.csv", index=False)

    def _generate_shannon_theory_plots(out_dir: Path, bw_hz: float = 100_000.0, rb_bps: Optional[float] = None):
        """Genera curvas teóricas de capacidad de Shannon-Hartley y límite de Shannon."""
        import numpy as _np
        import matplotlib.pyplot as _plt

        # 1) Capacidad C = W log2(1+SNR)
        snr_db = _np.linspace(-10.0, 30.0, 400)
        snr_lin = 10.0 ** (snr_db / 10.0)
        c_bps = bw_hz * _np.log2(1.0 + snr_lin)

        _plt.figure()
        _plt.plot(snr_db, c_bps / 1e3, color="#2563eb", lw=2.0, label="Capacidad teórica")
        if rb_bps is not None and rb_bps > 0:
            _plt.axhline(rb_bps / 1e3, color="#dc2626", ls="--", lw=1.4, label=f"R_b objetivo = {rb_bps/1e3:.1f} kb/s")
        _plt.xlabel("SNR [dB]")
        _plt.ylabel("Capacidad C [kb/s]")
        _plt.title(f"Shannon-Hartley (W={bw_hz/1e3:.0f} kHz)")
        _plt.grid(True, alpha=0.3)
        _plt.legend()
        _plt.tight_layout()
        _plt.savefig(out_dir / "capacidad_shannon_hartley.png", dpi=140)
        _plt.close()

        # 2) Límite de Shannon: (Eb/N0)_min = (2^eta - 1)/eta
        eta = _np.linspace(0.05, 4.0, 400)  # [bit/s/Hz]
        ebn0_min_lin = (_np.power(2.0, eta) - 1.0) / eta
        ebn0_min_db = 10.0 * _np.log10(ebn0_min_lin)

        _plt.figure()
        _plt.plot(eta, ebn0_min_db, color="#111827", lw=2.0, label="Límite teórico")
        _plt.axhline(-1.59, color="#6b7280", ls=":", lw=1.3, label="-1.59 dB (eta→0)")
        if rb_bps is not None and rb_bps > 0 and bw_hz > 0:
            eta0 = rb_bps / bw_hz
            if eta0 > 0:
                eb0 = (2.0 ** eta0 - 1.0) / eta0
                eb0_db = 10.0 * _np.log10(eb0)
                _plt.plot([eta0], [eb0_db], "o", color="#dc2626", ms=5, label=f"Trabajo (η={eta0:.2f})")
        _plt.xlabel("Eficiencia espectral η = R_b/W [bit/s/Hz]")
        _plt.ylabel("E_b/N_0 mínimo [dB]")
        _plt.title("Límite de Shannon")
        _plt.grid(True, alpha=0.3)
        _plt.legend()
        _plt.tight_layout()
        _plt.savefig(out_dir / "limite_shannon.png", dpi=140)
        _plt.close()

    def _latest_lab2_output_dir(base_lab2: str) -> Optional[str]:
        try:
            root = Path(base_lab2)
            if not root.exists():
                return None
            candidates = []
            for d in root.iterdir():
                if d.is_dir() and (d / "iq.bin").exists() and (d / "params.json").exists():
                    candidates.append(d)
            if not candidates:
                return None
            latest = sorted(candidates, key=lambda p: p.name, reverse=True)[0]
            return str(latest)
        except Exception:
            return None

    def _collect_lab2_runs(base_dir: Path):
        def _has_iq(p: Path) -> bool:
            return (p / "iq.bin").exists() or (p / "iq_tx.bin").exists()
        try:
            if not base_dir.exists():
                return []
            runs = []
            for params in base_dir.rglob("params.json"):
                run_dir = params.parent
                if _has_iq(run_dir):
                    rel = str(run_dir.relative_to(base_dir))
                    runs.append((rel, run_dir))
            runs = sorted(runs, key=lambda x: x[0])
            out = []
            seen = set()
            for rel, rd in runs:
                if rel in seen:
                    continue
                seen.add(rel)
                out.append((rel, rd))
            nested = [(rel, rd) for rel, rd in out if rel != "."]
            if nested:
                return nested
            if (base_dir / "params.json").exists() and _has_iq(base_dir):
                return [(".", base_dir)]
            return out
        except Exception:
            return []

    @app.template_filter("relative_path")
    def relative_path_filter(s: str) -> str:
        try:
            p = Path(s)
            return str(p.relative_to(ROOT))
        except Exception:
            return s

    @app.template_filter("basename")
    def basename_filter(s: str) -> str:
        return Path(s).name

    def _parse_int_auto(val, default: int) -> int:
        """Parsea enteros desde string con prefijos 0b/0x o decimal. Si falla, devuelve default.
        Acepta ints ya numéricos.
        """
        if val is None or val == "":
            return int(default)
        try:
            if isinstance(val, (int, float)):
                return int(val)
            s = str(val).strip()
            return int(s, 0)  # autodetect base
        except Exception:
            return int(default)

    # Construir bits desde parámetros del Lab 1
    def _build_bits_from_lab1(audio: str, text: str, fs: int, n_bits: int, quantizer: str,
                               source: str, method: str,
                               lfsr_seed: int = 0b1010110011,
                               lfsr_taps: tuple[int, ...] = (9, 6),
                               lfsr_bitwidth: int = 10) -> list[int]:
        # source: 'audio'|'text'
        # method: 'scrambling'
        bits_audio: list[int] = []
        bits_text: list[int] = []

        if source == "audio":
            x, _ = load_wav_mono(audio, target_fs=fs)
            if quantizer == "alaw":
                qA, _ = a_law_quantize(x, bits=n_bits, A=87.6, xmin=-1, xmax=1)
            else:
                qA, _ = uniform_quantize(x, bits=n_bits, xmin=-1, xmax=1)
            base_bits = ints_to_bits(qA, n_bits)
            bits_audio = [int(b) for b in scramble(base_bits, seed=lfsr_seed, taps=lfsr_taps, bitwidth=lfsr_bitwidth)]

        if source == "text":
            txt = open(text, "r", encoding="utf-8").read()
            b = txt.encode("utf-8")
            bytes_list = list(b)
            base_bits_b = []
            for val in bytes_list:
                for bit in range(7, -1, -1):
                    base_bits_b.append((val >> bit) & 1)
            bits_text = [int(b) for b in scramble(base_bits_b, seed=lfsr_seed, taps=lfsr_taps, bitwidth=lfsr_bitwidth)]

        if source == "audio":
            return bits_audio
        if source == "text":
            return bits_text
        raise ValueError("source (Formateo) inválido. Use 'audio' o 'text'.")

    def _generate_formateo_outputs(out_dir: Path, audio: str, text_path: str, fs: int, n_bits: int, quantizer: str,
                                  source: str, lfsr_seed: int, lfsr_taps: tuple[int, ...], lfsr_bitwidth: int,
                                  hist_bins: int = 50, entropy_step_a: int = 10000, entropy_step_b: int = 500,
                                  bw_hz: float = 100000.0) -> dict:
        """Genera las figuras de Formateo dentro de una carpeta específica y retorna rutas relativas."""
        out_dir.mkdir(parents=True, exist_ok=True)
        figdir = lab1.ensure_dirs(str(out_dir))

        rows = []
        if source == "audio":
            rows.extend(lab1.process_audio(
                audio, figdir, fs_target=fs, n_bits=n_bits, quantizer=quantizer,
                lfsr_seed=lfsr_seed, lfsr_taps=lfsr_taps, lfsr_bitwidth=lfsr_bitwidth,
                hist_bins=hist_bins, entropy_step=entropy_step_a,
            ))
        elif source == "text":
            rows.extend(lab1.process_text(
                text_path, figdir, lfsr_seed=lfsr_seed, lfsr_taps=lfsr_taps,
                lfsr_bitwidth=lfsr_bitwidth, entropy_step=entropy_step_b,
            ))
        else:
            raise ValueError("source (Formateo) inválido. Use 'audio' o 'text'.")

        # SQNR/MSE comparativos: Uniforme vs A-law (G.711 Argentina)
        if source == "audio":
            try:
                x, _ = lab1.load_wav_mono(audio, target_fs=fs)
                _generate_quantization_quality_plots(out_dir, x, n_bits)
                _generate_shannon_theory_plots(out_dir, bw_hz=float(bw_hz), rb_bps=float(fs * n_bits))
            except Exception:
                pass
        else:
            try:
                _generate_shannon_theory_plots(out_dir, bw_hz=float(bw_hz), rb_bps=None)
            except Exception:
                pass

        # Guardar métricas CSV para trazabilidad
        try:
            lab1_report.save_metrics_csv(str(out_dir), rows)
        except Exception:
            pass

        figs = []
        for p in sorted((Path(figdir)).glob('*.png')):
            try:
                figs.append(str(p.relative_to(ROOT)))
            except Exception:
                figs.append(str(p))
        # incluir sqnr/mse si existen
        for name in [
            "sqnr_comparacion.png",
            "mse_comparacion.png",
            "ecm_evolucion.png",
            "capacidad_shannon_hartley.png",
            "limite_shannon.png",
        ]:
            p = out_dir / name
            if p.exists():
                try:
                    figs.append(str(p.relative_to(ROOT)))
                except Exception:
                    figs.append(str(p))
        return {"out": str(out_dir), "figs": figs}


    def _validate_lab1_inputs(audio: str, text: str, out_dir: str, fs: int, n_bits: int, quantizer: str,
                              source: str,
                              lfsr_seed: int, lfsr_bitwidth: int, hist_bins: int,
                              entropy_step_a: int, entropy_step_b: int, lfsr_taps: tuple[int, ...],
                              bw_hz: float):
        if source not in {"audio", "text"}:
            raise ValueError("source (Formateo) inválido. Use 'audio' o 'text'.")
        if source == "audio" and not os.path.isfile(audio):
            raise ValueError(f"Audio no existe: {audio}")
        if source == "text" and not os.path.isfile(text):
            raise ValueError(f"Texto no existe: {text}")
        if fs <= 0:
            raise ValueError("fs debe ser > 0")
        if bw_hz <= 0:
            raise ValueError("ancho de banda (bw_hz) debe ser > 0")
        if not (1 <= n_bits <= 24):
            raise ValueError("n_bits debe estar entre 1 y 24")
        if quantizer not in {"uniform", "alaw", "both"}:
            raise ValueError("quantizer inválido")
        if not (1 <= lfsr_bitwidth <= 32):
            raise ValueError("lfsr_bitwidth debe estar entre 1 y 32")
        if hist_bins <= 0:
            raise ValueError("hist_bins debe ser > 0")
        if entropy_step_a <= 0 or entropy_step_b <= 0:
            raise ValueError("entropy steps deben ser > 0")
        for t in lfsr_taps:
            if t < 0 or t >= lfsr_bitwidth:
                raise ValueError("lfsr_taps fuera de rango para bitwidth")
        # Intentar crear carpeta de salida
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        test_path = Path(out_dir) / ".write_test"
        with open(test_path, "w") as f:
            f.write("ok")
        test_path.unlink(missing_ok=True)

    # Home con las cajitas de etapas
    @app.get("/")
    def index():
        stages = [
            {
                "id": "lab1",
                "title": "Formateo",
                "desc": "Muestreo, cuantización y scrambling",
                "link": url_for("lab1_page"),
            },
            {
                "id": "lab2",
                "title": "Modulación + RRC",
                "desc": "Transmisor digital, mapeo IQ y pulso de Nyquist",
                "link": url_for("lab2_page"),
            },
            {
                "id": "lab3",
                "title": "Canal y Rx",
                "desc": "Canal AWGN, Filtro Acoplado y Estimación de BER",
                "link": url_for("lab3_page"),
            },

        ]
        return render_template("index.html", stages=stages)

    # ---------- Lab 1 ----------
    @app.get("/lab1")
    def lab1_page():
        return render_template(
            "lab1.html",
            defaults=DEFAULTS,
        )

    @app.post("/lab1/run")
    def lab1_run():
        source = (request.form.get("source") or "audio").lower()
        audio = request.form.get("audio") or DEFAULTS["audio"]
        text = request.form.get("text") or DEFAULTS["text"]
        base_out = request.form.get("out") or DEFAULTS["out_lab1"]
        fs = int(request.form.get("fs") or 16000)
        bw_hz = float(request.form.get("bw_hz") or DEFAULTS["bw_hz"])
        n_bits = int(request.form.get("n_bits") or 8)
        quantizer = request.form.get("quantizer") or "alaw"
        lfsr_seed = _parse_int_auto(request.form.get("lfsr_seed"), int("0b1010110011", 2))
        lfsr_taps_str = (request.form.get("lfsr_taps") or "9,6").strip()
        try:
            lfsr_taps = tuple(int(t.strip()) for t in lfsr_taps_str.split(",") if t.strip() != "")
        except Exception:
            lfsr_taps = (9, 6)
        lfsr_bitwidth = _parse_int_auto(request.form.get("lfsr_bitwidth"), 10)
        hist_bins = _parse_int_auto(request.form.get("hist_bins"), 50)
        entropy_step_a = _parse_int_auto(request.form.get("entropy_step_a"), 10000)
        entropy_step_b = _parse_int_auto(request.form.get("entropy_step_b"), 500)

        # Crear subcarpeta con timestamp para reproducibilidad
        out_dir = str(_ts_dir(Path(base_out)))

        try:
            _validate_lab1_inputs(audio, text, out_dir, fs, n_bits, quantizer, source,
                                  lfsr_seed, lfsr_bitwidth, hist_bins, entropy_step_a, entropy_step_b, lfsr_taps, bw_hz)
            log_path = Path(out_dir) / "run_log.txt"
            _append_log(log_path, f"Formateo start: source={source}, audio={audio}, text={text}, fs={fs}, n_bits={n_bits}, quantizer={quantizer}")

            figdir = lab1.ensure_dirs(out_dir)

            rows = []
            if source == "audio":
                rows.extend(lab1.process_audio(audio, figdir, fs_target=fs, n_bits=n_bits, quantizer=quantizer,
                                               lfsr_seed=lfsr_seed, lfsr_taps=lfsr_taps, lfsr_bitwidth=lfsr_bitwidth,
                                               hist_bins=hist_bins, entropy_step=entropy_step_a))

                try:
                    x, _ = lab1.load_wav_mono(audio, target_fs=fs)
                    _generate_quantization_quality_plots(Path(out_dir), x, n_bits)
                    _generate_shannon_theory_plots(Path(out_dir), bw_hz=bw_hz, rb_bps=float(fs * n_bits))
                except Exception as ex:
                    _append_log(log_path, f"[WARN] SQNR/MSE: {ex}")
                    flash(f"[WARN] No se pudo calcular SQNR/MSE: {ex}", "warning")
            else:
                rows.extend(lab1.process_text(text, figdir, lfsr_seed=lfsr_seed, lfsr_taps=lfsr_taps,
                                              lfsr_bitwidth=lfsr_bitwidth, entropy_step=entropy_step_b))
                try:
                    _generate_shannon_theory_plots(Path(out_dir), bw_hz=bw_hz, rb_bps=None)
                except Exception:
                    pass

            lab1_report.save_metrics_csv(out_dir, rows)
            lab1_report.write_markdown(out_dir)
            try:
                from src.report import write_pdf as _write_pdf
                _write_pdf(out_dir)
            except Exception as _e:
                _append_log(Path(out_dir) / "run_log.txt", f"[WARN] PDF no generado: {_e}")
            # Guardar params
            _write_json(Path(out_dir) / "params.json", {
                "source": source,
                "audio": audio,
                "text": text,
                "fs": fs,
                "bw_hz": bw_hz,
                "n_bits": n_bits,
                "quantizer": quantizer,
                "lfsr_seed": lfsr_seed,
                "lfsr_taps": list(lfsr_taps),
                "lfsr_bitwidth": lfsr_bitwidth,
                "hist_bins": hist_bins,
                "entropy_step_a": entropy_step_a,
                "entropy_step_b": entropy_step_b,
                "out": out_dir,
            })

        except Exception as e:
            flash(f"Error al ejecutar Formateo: {e}", "error")
            return redirect(url_for("lab1_page"))

        return redirect(url_for("lab1_results", out=out_dir))

    @app.get("/lab1/results")
    def lab1_results():
        out_dir = request.args.get("out") or DEFAULTS["out_lab1"]
        figures_dir = Path(out_dir) / "figures"
        summary = {}
        pp = Path(out_dir) / "params.json"
        if pp.exists():
            try:
                import json
                summary = json.loads(pp.read_text(encoding="utf-8"))
            except Exception:
                summary = {}
        figs = []
        if figures_dir.exists():
            for p in sorted(figures_dir.glob("*.png")):
                figs.append(p.name)
        root_figs = []
        for name in [
            "sqnr_comparacion.png",
            "mse_comparacion.png",
            "ecm_evolucion.png",
            "capacidad_shannon_hartley.png",
            "limite_shannon.png",
        ]:
            if (Path(out_dir) / name).exists():
                root_figs.append(name)
        files = []
        for p in sorted(Path(out_dir).glob("*")):
            if p.is_file() and p.suffix in {".png", ".csv", ".md"}:
                files.append(p.name)
        # Cargar logs si existen
        log_text = None
        lp = Path(out_dir) / "run_log.txt"
        if lp.exists():
            try:
                log_text = lp.read_text(encoding="utf-8")
            except Exception:
                log_text = None
        low_level_csv = None
        llp = Path(out_dir) / "figures" / "A_quantizer_low_level_compare_data.csv"
        if llp.exists():
            low_level_csv = str(llp)
        return render_template(
            "lab1_results.html",
            out=out_dir,
            figs=figs,
            root_figs=root_figs,
            files=files,
            log_text=log_text,
            summary=summary,
            low_level_csv=low_level_csv,
        )

    # ---------- Lab 2 ----------
    @app.get("/lab2")
    def lab2_page():
        return render_template(
            "lab2.html",
            defaults=DEFAULTS,
        )

    @app.post("/lab2/run")
    def lab2_run():
        base_out = request.form.get("out") or DEFAULTS["out_lab2"]
        out_dir = str(_ts_dir(Path(base_out)))
        mod = (request.form.get("modulation") or "QPSK").upper()
        m_order = int(request.form.get("m_order") or 4)
        bits_len = int(request.form.get("bits_len") or 2000)
        sps = int(request.form.get("sps") or 8)
        alpha = float(request.form.get("alpha") or 0.25)
        span = int(request.form.get("span_symbols") or 8)
        seed = int(request.form.get("seed") or 0)
        eye_span = int(request.form.get("eye_span") or 2)
        eye_traces = int(request.form.get("eye_traces") or 300)

        try:
            if sps <= 0:
                raise ValueError("sps debe ser > 0")
            if bits_len <= 0:
                raise ValueError("bits_len debe ser > 0")
            if mod not in {"BPSK", "QPSK", "MPSK", "M-PSK"}:
                raise ValueError("Modulación no soportada (use BPSK, QPSK o M-PSK)")
            if mod in {"MPSK", "M-PSK"}:
                if m_order < 2 or (m_order & (m_order - 1)) != 0:
                    raise ValueError("Para M-PSK, M debe ser potencia de 2 (>=2)")
            if eye_span < 1:
                raise ValueError("eye_span debe ser >= 1")
            if eye_traces <= 0:
                raise ValueError("eye_traces debe ser > 0")
            params = lab2_rrc.Lab2Params(
                out_dir=out_dir,
                n_bits=bits_len,
                modulation=mod,
                m_order=m_order,
                sps=sps,
                rolloff=alpha,
                span=span,
                seed=seed,
                eye_span=eye_span,
                eye_traces=eye_traces,
            )
            paths = lab2_rrc.run_lab2(params)
            # Log simple
            _write_json(Path(out_dir) / "params.json", {
                "n_bits": bits_len,
                "modulation": mod,
                "m_order": m_order,
                "sps": sps,
                "rolloff": alpha,
                "span": span,
                "seed": seed,
                "out": out_dir,
                "eye_span": eye_span,
                "eye_traces": eye_traces,
            })
            _append_log(Path(out_dir) / "run_log.txt", f"Modulación run OK: {params}")
            try:
                _generate_shannon_theory_plots(Path(out_dir), bw_hz=100_000.0, rb_bps=None)
            except Exception:
                pass
        except Exception as e:
            flash(f"Error al ejecutar Modulación: {e}", "error")
            return redirect(url_for("lab2_page"))

        return redirect(url_for("lab2_results", out=out_dir))

    @app.get("/lab2/results")
    def lab2_results():
        out_dir = request.args.get("out") or DEFAULTS["out_lab2"]
        p = Path(out_dir)
        figs = []
        # Preferencia por nombres estándar sugeridos
        for name in ["iq_time.png", "constellation_symbols.png", "constellation_shaped.png", "spectrum.png", "rrc_impulse.png", "eye_diagram.png", "l1_bits_hist.png", "bits_iq_transition.png", "rrc_discrete_upsampling.png", "rrc_discrete_shaping.png", "rrc_two_symbols.png", "isi_vs_sps.png", "capacidad_shannon_hartley.png", "limite_shannon.png"]:
            fn = p / name
            if fn.exists():
                figs.append(name)
        # si no existen, incluir cualquier PNG
        if not figs:
            figs = [f.name for f in sorted(p.glob("*.png"))]
        bin_files = [f.name for f in sorted(p.glob("*.bin"))]
        other = []
        for name in ["params.json", "run_log.txt"]:
            if (p / name).exists():
                other.append(name)
        return render_template("lab2_results.html", out=out_dir, figs=figs, bin_files=bin_files, other=other)

    @app.post("/api/lab2/run_from_lab1")
    def api_lab2_run_from_lab1():
        data = request.get_json(force=True) or {}
        base_out = data.get("out_dir") or data.get("out") or DEFAULTS["out_lab2"]
        out_dir = str(_ts_dir(Path(base_out)))
        base_abs = (ROOT / out_dir).resolve() if not Path(out_dir).is_absolute() else Path(out_dir).resolve()
        # Lab2 params
        mod = (data.get("modulation") or "QPSK").upper()
        m_order = int(data.get("m_order") or 4)
        sps = int(data.get("sps") or 8)
        alpha = float(data.get("alpha") or data.get("rolloff") or 0.25)
        span = int(data.get("span") or 8)
        raw_deltas = data.get("isi_sps_deltas") or [0, 0, 0, 0]
        try:
            deltas_list = list(raw_deltas)[:4]
            deltas_list += [0] * (4 - len(deltas_list))
            isi_sps_deltas = tuple(max(0, int(v)) for v in deltas_list)
        except Exception:
            isi_sps_deltas = (0, 0, 0, 0)
        seed = 0
        # Lab1 params
        l1 = data.get("lab1") or {}
        audio = l1.get("audio") or DEFAULTS["audio"]
        text = l1.get("text") or DEFAULTS["text"]
        fs = int(l1.get("fs") or 16000)
        bw_hz = float(l1.get("bw_hz") or DEFAULTS["bw_hz"])
        n_bits = int(l1.get("n_bits") or 8)
        quantizer = (l1.get("quantizer") or "alaw").lower()
        source = (l1.get("source") or "audio").lower()
        method = (l1.get("method") or "scrambling").lower()
        l1_seed = _parse_int_auto(l1.get("lfsr_seed"), int("0b1010110011", 2))
        taps_str = (str(l1.get("lfsr_taps")) if l1.get("lfsr_taps") is not None else "9,6").strip()
        try:
            l1_taps = tuple(int(t.strip()) for t in taps_str.split(',') if t.strip() != '')
        except Exception:
            l1_taps = (9, 6)
        l1_bitwidth = _parse_int_auto(l1.get("lfsr_bitwidth"), 10)
        # Generar figuras de Formateo para encadenado (una vez por corrida)
        formateo_dir = base_abs / "formateo"
        formateo_payload = _generate_formateo_outputs(
            formateo_dir, audio, text, fs, n_bits, quantizer, source,
            l1_seed, l1_taps, l1_bitwidth, bw_hz=bw_hz
        )
        try:
            if sps <= 0:
                raise ValueError("sps debe ser > 0")
            if mod not in {"BPSK", "QPSK", "MPSK", "M-PSK"}:
                raise ValueError("Modulación no soportada (use BPSK, QPSK o M-PSK)")
            if mod in {"MPSK", "M-PSK"}:
                if m_order < 2 or (m_order & (m_order - 1)) != 0:
                    raise ValueError("Para M-PSK, M debe ser potencia de 2 (>=2)")
            if not (1 <= n_bits <= 24):
                raise ValueError("n_bits (Formateo) debe estar entre 1 y 24")
            if quantizer not in {"alaw", "uniform"}:
                raise ValueError("quantizer (Formateo) inválido")
            if source not in {"audio", "text"}:
                raise ValueError("source (Formateo) inválido. Use 'audio' o 'text'")
            if method != "scrambling":
                raise ValueError("method (Formateo) inválido")
            # Construir bits desde Formateo con parámetros avanzados
            import numpy as _np

            def _run_one(label: str, src: str, method_use: str, out_dir_use: Path):
                bits_local = _build_bits_from_lab1(
                    audio, text, fs, n_bits, quantizer, src, method_use,
                    lfsr_seed=l1_seed, lfsr_taps=l1_taps, lfsr_bitwidth=l1_bitwidth
                )
                if len(bits_local) == 0:
                    raise ValueError(f"La secuencia de bits ({label}) está vacía")

                out_dir_use.mkdir(parents=True, exist_ok=True)
                bf = out_dir_use / "bits_from_lab1.bin"
                _np.asarray(bits_local, dtype=_np.uint8).ravel().tofile(bf)
                bf_alias = out_dir_use / "bits_formateo.bin"
                _safe_alias(bf, bf_alias)

                try:
                    plot_hist_bits(bits_local, f"Bits (Formateo→Modulación) [{label}]", str(out_dir_use / "l1_bits_hist.png"), as_probability=True)
                except Exception:
                    pass

                from src.bits_utils import bits_entropy_stats as _bes
                p0, p1, H, var = _bes(bits_local)
                audit = {
                    "bits": {"count": len(bits_local), "p0": float(p0), "p1": float(p1), "H": float(H), "var": float(var),
                             "source": src, "method": method_use}
                }

                import hashlib
                if src == "audio":
                    try:
                        h = hashlib.md5()
                        with open(audio, 'rb') as _f:
                            h.update(_f.read())
                        x, _ = load_wav_mono(audio, target_fs=fs)
                        audit["audio"] = {
                            "path": audio,
                            "md5": h.hexdigest(),
                            "fs_target": fs,
                            "samples": int(len(x)),
                            "duration_s": float(len(x) / float(fs)) if fs else None,
                        }
                    except Exception as _e:
                        audit["audio_error"] = str(_e)
                if src == "text":
                    try:
                        h = hashlib.md5()
                        b = open(text, 'rb').read()
                        h.update(b)
                        preview = open(text, 'r', encoding='utf-8', errors='ignore').read(200)
                        audit["text"] = {"path": text, "md5": h.hexdigest(), "bytes": len(b), "preview": preview}
                    except Exception as _e:
                        audit["text_error"] = str(_e)

                _write_json(out_dir_use / "params.json", {
                    "lab2": {"modulation": mod, "m_order": m_order, "sps": sps, "rolloff": alpha, "span": span, "seed": seed,
                             "isi_sps_deltas": list(isi_sps_deltas)},
                    "lab1": {"audio": audio, "text": text, "fs": fs, "n_bits": n_bits, "quantizer": quantizer,
                             "source": src, "method": method_use, "lfsr_seed": l1_seed,
                             "lfsr_taps": list(l1_taps), "lfsr_bitwidth": l1_bitwidth},
                    "n_bits_effective": len(bits_local),
                    "out": str(out_dir_use),
                    "l1_metrics": audit["bits"],
                })

                params = lab2_rrc.Lab2Params(
                    out_dir=str(out_dir_use), n_bits=len(bits_local), modulation=mod, m_order=m_order, sps=sps, rolloff=alpha, span=span,
                    seed=seed, isi_sps_deltas=isi_sps_deltas
                )
                paths = lab2_rrc.run_lab2(params, bits=_np.array(bits_local, dtype=_np.uint8))
                _write_json(out_dir_use / "params.json", {
                    "lab2": {"modulation": mod, "m_order": m_order, "sps": sps, "rolloff": alpha, "span": span, "seed": seed,
                             "isi_sps_deltas": list(isi_sps_deltas)},
                    "lab1": {"audio": audio, "text": text, "fs": fs, "n_bits": n_bits, "quantizer": quantizer,
                             "source": src, "method": method_use, "lfsr_seed": l1_seed,
                             "lfsr_taps": list(l1_taps), "lfsr_bitwidth": l1_bitwidth},
                    "n_bits_effective": len(bits_local),
                    "out": str(out_dir_use),
                    "l1_metrics": audit["bits"],
                })
                gen = {}
                for k, v in paths.items():
                    p = Path(v)
                    try:
                        gen[k] = str(p.relative_to(ROOT))
                    except Exception:
                        gen[k] = str(p)
                try:
                    if bf_alias.exists():
                        gen["bits_formateo_bin"] = str(bf_alias.relative_to(ROOT))
                    elif bf.exists():
                        gen["bits_formateo_bin"] = str(bf.relative_to(ROOT))
                except Exception:
                    gen["bits_formateo_bin"] = str(bf_alias if bf_alias.exists() else bf)
                l1_hist = out_dir_use / "l1_bits_hist.png"
                if l1_hist.exists():
                    try:
                        gen["l1_bits_hist"] = str(l1_hist.relative_to(ROOT))
                    except Exception:
                        gen["l1_bits_hist"] = str(l1_hist)
                audit["params"] = {"lfsr_seed": l1_seed, "lfsr_taps": list(l1_taps), "lfsr_bitwidth": l1_bitwidth,
                                    "fs": fs, "n_bits": n_bits, "quantizer": quantizer, "source": src, "method": method_use}
                return {"paths": gen, "audit": audit, "n_bits": len(bits_local), "out": str(out_dir_use), "label": label}

            source_label = "Audio" if source == "audio" else "Texto"
            result = _run_one(source_label, source, method, base_abs)
            return jsonify({"ok": True, "out": str(base_abs), "paths": result["paths"], "n_bits": result["n_bits"], "audit": result["audit"], "formateo": formateo_payload})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    # ---------- API Endpoints ----------
    @app.post("/api/lab1/run")
    def api_lab1_run():
        data = request.get_json(force=True) or {}
        source = (data.get("source") or "audio").lower()
        audio = data.get("audio") or DEFAULTS["audio"]
        text = data.get("text") or DEFAULTS["text"]
        base_out = data.get("out") or DEFAULTS["out_lab1"]
        fs = int(data.get("fs") or 16000)
        bw_hz = float(data.get("bw_hz") or DEFAULTS["bw_hz"])
        n_bits = int(data.get("n_bits") or 8)
        quantizer = data.get("quantizer") or "alaw"
        lfsr_seed = _parse_int_auto(data.get("lfsr_seed"), int("0b1010110011", 2))
        lfsr_taps_str = (data.get("lfsr_taps") or "9,6").strip()
        try:
            lfsr_taps = tuple(int(t.strip()) for t in lfsr_taps_str.split(",") if t.strip() != "")
        except Exception:
            lfsr_taps = (9, 6)
        lfsr_bitwidth = _parse_int_auto(data.get("lfsr_bitwidth"), 10)
        hist_bins = _parse_int_auto(data.get("hist_bins"), 50)
        entropy_step_a = _parse_int_auto(data.get("entropy_step_a"), 10000)
        entropy_step_b = _parse_int_auto(data.get("entropy_step_b"), 500)
        out_dir = str(_ts_dir(Path(base_out)))
        try:
            _validate_lab1_inputs(audio, text, out_dir, fs, n_bits, quantizer, source,
                                  lfsr_seed, lfsr_bitwidth, hist_bins, entropy_step_a, entropy_step_b, lfsr_taps, bw_hz)
            figdir = lab1.ensure_dirs(out_dir)
            rows = []
            if source == "audio":
                rows.extend(lab1.process_audio(audio, figdir, fs_target=fs, n_bits=n_bits, quantizer=quantizer,
                                               lfsr_seed=lfsr_seed, lfsr_taps=lfsr_taps, lfsr_bitwidth=lfsr_bitwidth,
                                               hist_bins=hist_bins, entropy_step=entropy_step_a))
                try:
                    x, _ = lab1.load_wav_mono(audio, target_fs=fs)
                    _generate_quantization_quality_plots(Path(out_dir), x, n_bits)
                    _generate_shannon_theory_plots(Path(out_dir), bw_hz=bw_hz, rb_bps=float(fs * n_bits))
                except Exception:
                    pass
            else:
                rows.extend(lab1.process_text(text, figdir, lfsr_seed=lfsr_seed, lfsr_taps=lfsr_taps,
                                              lfsr_bitwidth=lfsr_bitwidth, entropy_step=entropy_step_b))
                try:
                    _generate_shannon_theory_plots(Path(out_dir), bw_hz=bw_hz, rb_bps=None)
                except Exception:
                    pass
            lab1_report.save_metrics_csv(out_dir, rows)
            lab1_report.write_markdown(out_dir)
            try:
                from src.report import write_pdf as _write_pdf
                _write_pdf(out_dir)
            except Exception as _e:
                pass
            _write_json(Path(out_dir) / "params.json", {
                "source": source,
                "audio": audio,
                "text": text,
                "fs": fs,
                "bw_hz": bw_hz,
                "n_bits": n_bits,
                "quantizer": quantizer,
                "lfsr_seed": lfsr_seed,
                "lfsr_taps": list(lfsr_taps),
                "lfsr_bitwidth": lfsr_bitwidth,
                "hist_bins": hist_bins,
                "entropy_step_a": entropy_step_a,
                "entropy_step_b": entropy_step_b,
                "out": out_dir,
            })
            # Listar archivos
            base = Path(out_dir)
            if not base.is_absolute():
                base_abs = (ROOT / base).resolve()
            else:
                base_abs = base.resolve()
            gen = []
            for p in base_abs.rglob("*"):
                if p.is_file():
                    try:
                        gen.append(str(p.relative_to(ROOT)))
                    except Exception:
                        gen.append(str(p))
            return jsonify({"ok": True, "out": str(base_abs), "files": gen})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    def _write_lab3_status(out_dir: str, job_id: str, state: str, progress: float, message: str) -> None:
        try:
            p = Path(out_dir) / "status.json"
            existing = {}
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                except Exception:
                    existing = {}
            _write_json(p, {
                "job_id": job_id,
                "state": state,
                "progress": progress,
                "message": message,
                "out_dir": out_dir,
                "cancel_requested": bool(existing.get("cancel_requested", False)),
            })
        except Exception:
            pass

    def _lab3_cancel_requested(out_dir: str) -> bool:
        try:
            p = Path(out_dir) / "status.json"
            if not p.exists():
                return False
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return bool(data.get("cancel_requested", False))
        except Exception:
            return False


    def _run_lab3_pipeline(
        base_out: str,
        lab2_path: str,
        subrun: str,
        mod: str,
        n_bits: int,
        eb_start: float,
        eb_end: float,
        eb_step: float,
        trials: int,
        theory_points: int,
        use_rx_rrc: bool,
        timing_offset_ts: float,
        sps: int,
        seed: int,
        channel_mode: str,
        progress_cb=None,
        cancel_cb=None,
        prepared_out_dir: Optional[str] = None,
    ) -> str:
        out_dir = prepared_out_dir or str(_ts_dir(Path(base_out)))
        use_l2_chain = True
        if trials <= 0:
            raise ValueError("trials_per_ebn0 debe ser > 0")
        l2 = lab2_path or _latest_lab2_output_dir(DEFAULTS["out_lab2"])
        if subrun:
            l2 = str(Path(l2).joinpath(subrun))
        if not l2:
            raise ValueError("No hay salida de Modulación disponible. Ejecutá Modulación primero.")
        l2_base = Path(l2)
        is_direct_run = ((((l2_base / "iq.bin").exists()) or ((l2_base / "iq_tx.bin").exists())) and (l2_base / "params.json").exists())
        if is_direct_run:
            runs = [(".", l2_base)]
        else:
            runs = _collect_lab2_runs(l2_base)
        if not runs:
            raise ValueError("No se encontraron corridas válidas en Modulación.")
        multi = len(runs) > 1 or (runs and runs[0][0] != ".")

        if multi:
            runs_meta = []
            total = len(runs)
            for i, (rel, rdir) in enumerate(runs, 1):
                if cancel_cb and cancel_cb():
                    raise InterruptedError("Cancelado por el usuario")
                if progress_cb:
                    progress_cb(i - 1, total, rel)
                out_sub = Path(out_dir) / rel
                out_sub.mkdir(parents=True, exist_ok=True)
                res_run = lab3_demod.run_simulation_from_file(
                    lab2_dir=str(rdir),
                    out_dir=str(out_sub),
                    ebn0_start=eb_start,
                    ebn0_end=eb_end,
                    ebn0_step=eb_step,
                    trials_per_ebn0=trials,
                    theory_points=theory_points,
                    seed=seed,
                    channel_mode=channel_mode,
                    use_rx_rrc=use_rx_rrc,
                    timing_offset_ts=timing_offset_ts,
                    cancel_cb=cancel_cb,
                )
                try:
                    lab3_report.write_markdown(str(out_sub))
                    lab3_report.write_pdf(str(out_sub))
                except Exception as _e:
                    _append_log(out_sub / "run_log.txt", f"[WARN] Informe Canal y Rx no generado: {_e}")
                ber_plot = out_sub / "ber_curve.png"
                ber_csv = out_sub / "ber_results.csv"
                runs_meta.append({
                    "label": rel,
                    "lab2_path": str(rdir),
                    "out_dir": str(out_sub),
                    "ber_plot": str(ber_plot) if ber_plot.exists() else None,
                    "ber_csv": str(ber_csv) if ber_csv.exists() else None,
                    "modulation": res_run.get("modulation"),
                    "m_order": res_run.get("m_order"),
                    "sps": res_run.get("sps"),
                    "rolloff": res_run.get("rolloff"),
                    "span": res_run.get("span"),
                })

            if progress_cb:
                progress_cb(total, total, "final")

            _write_json(Path(out_dir) / "runs.json", {
                "lab2_base": str(l2_base),
                "runs": runs_meta,
            })

            _write_json(Path(out_dir) / "params.json", {
                "mode": "lab2_chain_multi",
                "lab2_path": str(l2_base),
                "n_bits": n_bits,
                "modulation": "multi",
                "m_order": None,
                "eb_start": eb_start,
                "eb_end": eb_end,
                "eb_step": eb_step,
                "trials_per_ebn0": trials,
                "theory_points": theory_points,
                "use_rx_rrc": bool(use_rx_rrc),
                "timing_offset_ts": float(timing_offset_ts),
                "sps": sps,
                "seed": seed,
                "channel_mode": channel_mode,
                "out": out_dir,
                "runs": [r[0] for r in runs],
                "run_configs": runs_meta,
            })

            _append_log(Path(out_dir) / "run_log.txt", f"Canal y Rx run OK: mode=lab2_chain_multi runs={len(runs_meta)}")
            return out_dir

        # Single run
        rdir = runs[0][1]
        if cancel_cb and cancel_cb():
            raise InterruptedError("Cancelado por el usuario")
        res_single = lab3_demod.run_simulation_from_file(
            lab2_dir=str(rdir),
            out_dir=out_dir,
            ebn0_start=eb_start,
            ebn0_end=eb_end,
            ebn0_step=eb_step,
            trials_per_ebn0=trials,
            theory_points=theory_points,
            seed=seed,
            channel_mode=channel_mode,
            use_rx_rrc=use_rx_rrc,
            timing_offset_ts=timing_offset_ts,
            cancel_cb=cancel_cb,
        )

        _write_json(Path(out_dir) / "params.json", {
            "mode": "lab2_chain" if use_l2_chain else "standalone",
            "lab2_path": str(rdir) if use_l2_chain else None,
            "n_bits": n_bits,
            "modulation": res_single.get("modulation", mod),
            "m_order": res_single.get("m_order"),
            "eb_start": eb_start,
            "eb_end": eb_end,
            "eb_step": eb_step,
            "trials_per_ebn0": trials,
            "theory_points": theory_points,
            "use_rx_rrc": bool(use_rx_rrc),
            "timing_offset_ts": float(timing_offset_ts),
            "sps": res_single.get("sps", sps),
            "rolloff": res_single.get("rolloff"),
            "span": res_single.get("span"),
            "seed": seed,
            "channel_mode": channel_mode,
            "out": out_dir
        })
        try:
            lab3_report.write_markdown(out_dir)
            lab3_report.write_pdf(out_dir)
        except Exception as _e:
            _append_log(Path(out_dir) / "run_log.txt", f"[WARN] Informe Canal y Rx no generado: {_e}")
        _append_log(Path(out_dir) / "run_log.txt", f"Canal y Rx run OK: mode={'lab2_chain' if use_l2_chain else 'standalone'}")
        return out_dir


    @app.get("/lab3")
    def lab3_page():
        return render_template(
            "lab3.html",
            defaults=DEFAULTS,
        )

    @app.post("/lab3/run")
    def lab3_run():
        base_out = request.values.get("out") or DEFAULTS["out_lab3"]
        lab2_path = (request.values.get("lab2_path") or "").strip()
        subrun = (request.values.get("subrun") or "").strip()
        mod = "QPSK"
        n_bits = 10000
        eb_start = float(request.values.get("eb_start") or 0.0)
        eb_end = float(request.values.get("eb_end") or 12.0)
        eb_step = float(request.values.get("eb_step") or 2.0)
        trials = int(request.values.get("trials_per_ebn0") or 20)
        theory_points = int(request.values.get("theory_points") or 300)
        use_rx_rrc = str(request.values.get("use_rx_rrc") or "").lower() in {"1", "true", "on", "yes"}
        timing_offset_ts = float(request.values.get("timing_offset_ts") or 0.0)
        sps = 8
        seed = 42
        channel_mode = (request.values.get("channel_mode") or "awgn").strip().lower()

        try:
            out_dir = _run_lab3_pipeline(
                base_out=base_out,
                lab2_path=lab2_path,
                subrun=subrun,
                mod=mod,
                n_bits=n_bits,
                eb_start=eb_start,
                eb_end=eb_end,
                eb_step=eb_step,
                trials=trials,
                theory_points=theory_points,
                use_rx_rrc=use_rx_rrc,
                timing_offset_ts=timing_offset_ts,
                sps=sps,
                seed=seed,
                channel_mode=channel_mode,
            )
        except Exception as e:
            flash(f"Error al ejecutar Canal y Rx: {e}", "error")
            return redirect(url_for("lab3_page"))

        return redirect(url_for("lab3_results", out=out_dir))

    @app.post("/api/lab3/run_async")
    def api_lab3_run_async():
        data = request.get_json(force=True) or {}
        base_out = data.get("out") or DEFAULTS["out_lab3"]
        lab2_path = (data.get("lab2_path") or "").strip()
        subrun = (data.get("subrun") or "").strip()
        mod = "QPSK"
        n_bits = 10000
        eb_start = float(data.get("eb_start") or 0.0)
        eb_end = float(data.get("eb_end") or 12.0)
        eb_step = float(data.get("eb_step") or 2.0)
        trials = int(data.get("trials_per_ebn0") or 20)
        theory_points = int(data.get("theory_points") or 300)
        use_rx_rrc = str(data.get("use_rx_rrc") or "").lower() in {"1", "true", "on", "yes"}
        timing_offset_ts = float(data.get("timing_offset_ts") or 0.0)
        sps = 8
        seed = 42
        channel_mode = (data.get("channel_mode") or "awgn").strip().lower()

        job_id = uuid.uuid4().hex
        out_dir = str(_ts_dir(Path(base_out)))
        _write_lab3_status(out_dir, job_id, "running", 0.0, "Iniciando simulación")

        def _worker():
            try:
                def _progress(done, total, label):
                    if _lab3_cancel_requested(out_dir):
                        raise InterruptedError("Cancelado por el usuario")
                    if total <= 0:
                        return
                    prog = max(0.0, min(1.0, float(done) / float(total)))
                    if done >= total:
                        msg = "Finalizando"
                    else:
                        msg = f"Procesando {label} ({done+1}/{total})"
                    _write_lab3_status(out_dir, job_id, "running", prog, msg)

                _run_lab3_pipeline(
                    base_out=base_out,
                    lab2_path=lab2_path,
                    subrun=subrun,
                    mod=mod,
                    n_bits=n_bits,
                    eb_start=eb_start,
                    eb_end=eb_end,
                    eb_step=eb_step,
                    trials=trials,
                    theory_points=theory_points,
                    use_rx_rrc=use_rx_rrc,
                    timing_offset_ts=timing_offset_ts,
                    sps=sps,
                    seed=seed,
                    channel_mode=channel_mode,
                    progress_cb=_progress,
                    cancel_cb=lambda: _lab3_cancel_requested(out_dir),
                    prepared_out_dir=out_dir,
                )
                _write_lab3_status(out_dir, job_id, "done", 1.0, "Completado")
            except InterruptedError:
                _write_lab3_status(out_dir, job_id, "canceled", 0.0, "Cancelado por el usuario")
            except Exception as e:
                _write_lab3_status(out_dir, job_id, "error", 0.0, str(e))

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        return jsonify({"ok": True, "job_id": job_id, "out_dir": out_dir})

    @app.get("/api/lab3/status")
    def api_lab3_status():
        import json
        out_dir = (request.args.get("out") or "").strip()
        if not out_dir:
            return jsonify({"ok": False, "error": "out requerido"}), 400
        p = Path(out_dir) / "status.json"
        if not p.exists():
            return jsonify({"ok": False, "error": "status no encontrado"}), 404
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

        log_tail = None
        log_path = Path(out_dir) / "run_log.txt"
        if log_path.exists():
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                log_tail = "".join(lines[-15:])
            except Exception:
                log_tail = None

        data["log_tail"] = log_tail
        return jsonify({"ok": True, "status": data})

    @app.post("/api/lab3/cancel")
    def api_lab3_cancel():
        out_dir = (request.get_json(force=True) or {}).get("out", "")
        out_dir = str(out_dir).strip()
        if not out_dir:
            return jsonify({"ok": False, "error": "out requerido"}), 400
        p = Path(out_dir) / "status.json"
        if not p.exists():
            return jsonify({"ok": False, "error": "status no encontrado"}), 404
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        data["cancel_requested"] = True
        if data.get("state") == "running":
            data["message"] = "Cancelación solicitada…"
        _write_json(p, data)
        return jsonify({"ok": True})

    @app.get("/lab3/results")
    def lab3_results():
        out_dir = request.args.get("out") or DEFAULTS["out_lab3"]
        p = Path(out_dir)
        summary = {}
        rx_figs = []
        text_rx_preview = None
        text_tx_preview = None
        recovery_ebn0_db = None
        ber_rows = []
        ber_context = {}

        params_file = p / "params.json"
        if params_file.exists():
            try:
                with open(params_file, "r", encoding="utf-8") as f:
                    summary = json.load(f)
            except Exception:
                summary = {}

        if summary.get("diag_ebn0_db") is not None:
            recovery_ebn0_db = summary.get("diag_ebn0_db")
        elif summary.get("eb_start") is not None:
            recovery_ebn0_db = summary.get("eb_start")
        summary.setdefault("use_rx_rrc", True)
        summary.setdefault("timing_offset_ts", 0.0)

        for name in [
            "tx_rx_constellations.png",
            "rx_time.png",
            "rx_downsampling.png",
            "rx_constellation.png",
            "rx_eye.png",
            "mf_impulse.png",
            "mf_freq.png",
            "rx_decision.png",
            "ber_point.png",
            "audio_compare_rx.png",
        ]:
            fp = p / name
            if fp.exists():
                rx_figs.append(name)

        runs = None
        runs_file = p / "runs.json"
        if runs_file.exists():
            try:
                with open(runs_file, "r", encoding="utf-8") as f:
                    runs = json.load(f)
            except Exception:
                runs = None
        
        # Check specific files (single-run)
        ber_plot_path = p / "ber_curve.png"
        ber_plot = str(ber_plot_path) if ber_plot_path.exists() else None
        ber_csv = "ber_results.csv" if (p / "ber_results.csv").exists() else None

        ber_csv_path = p / "ber_results.csv"
        if ber_csv_path.exists():
            try:
                with open(ber_csv_path, "r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        ber_rows.append({
                            "ebn0_target_db": float(row.get("EbN0_Target_dB", 0.0)),
                            "ber_sim": float(row.get("BER_Sim", 0.0)),
                            "ber_theory": float(row.get("BER_Theory", 0.0)),
                            "ber_std": float(row.get("BER_Std_MonteCarlo", 0.0)),
                            "ber_ci95": float(row.get("BER_CI95_MonteCarlo", 0.0)),
                            "ebn0_est_mean_db": float(row.get("EbN0_Est_Mean_dB", 0.0)),
                            "ebn0_est_std_db": float(row.get("EbN0_Est_Std_dB", 0.0)),
                            "trials": int(float(row.get("Trials", 0) or 0)),
                        })
            except Exception:
                ber_rows = []

        if ber_rows:
            ber_context = {
                "eb_start": ber_rows[0]["ebn0_target_db"],
                "eb_end": ber_rows[-1]["ebn0_target_db"],
                "eb_step": summary.get("eb_step"),
                "sim_points": len(ber_rows),
                "theory_points": int(summary.get("theory_points") or 300),
                "trials_per_ebn0": int(summary.get("trials_per_ebn0") or ber_rows[0]["trials"]),
                "modulation": summary.get("modulation"),
                "m_order": summary.get("m_order"),
                "use_rx_rrc": bool(summary.get("use_rx_rrc", True)),
                "timing_offset_ts": float(summary.get("timing_offset_ts") or 0.0),
            }
        
        other = []
        for name in [
            "params.json",
            "run_log.txt",
            "informe_lab3.md",
            "informe_lab3.pdf",
            "runs.json",
            "audio_rx.wav",
            "audio_tx_ref.wav",
            "text_rx.txt",
            "text_tx_ref.txt",
        ]:
            if (p / name).exists():
                other.append(name)

        rx_txt = p / "text_rx.txt"
        if rx_txt.exists():
            try:
                text_rx_preview = rx_txt.read_text(encoding="utf-8")[:4000]
            except Exception:
                text_rx_preview = None

        tx_txt = p / "text_tx_ref.txt"
        if tx_txt.exists():
            try:
                text_tx_preview = tx_txt.read_text(encoding="utf-8")[:4000]
            except Exception:
                text_tx_preview = None
                
        return render_template(
            "lab3_results.html",
            out=out_dir,
            ber_plot=ber_plot,
            ber_csv=ber_csv,
            other=other,
            runs=runs,
            summary=summary,
            rx_figs=rx_figs,
            text_rx_preview=text_rx_preview,
            text_tx_preview=text_tx_preview,
            recovery_ebn0_db=recovery_ebn0_db,
            ber_rows=ber_rows,
            ber_context=ber_context,
        )




    @app.post("/api/lab3/run_single")
    def api_lab3_run_single():
        data = request.get_json(force=True) or {}
        
        # --- Mode A: File-Based Integration (Lab 2 Output) ---
        if "lab2_path" in data and data["lab2_path"]:
            try:
                l2_path = data["lab2_path"]
                ebn0 = float(data.get("ebn0", 10.0))
                channel_mode = (data.get("channel_mode") or "awgn").strip().lower()
                use_rx_rrc = str(data.get("use_rx_rrc") or "").lower() in {"1", "true", "on", "yes"}
                timing_offset_ts = float(data.get("timing_offset_ts") or 0.0)
                # Output dir
                out_base = data.get("out") or DEFAULTS["out_lab3"]
                out_ts = _ts_dir(Path(out_base))
                
                res = lab3_demod.run_from_file(
                    l2_path,
                    ebn0,
                    str(out_ts),
                    channel_mode=channel_mode,
                    use_rx_rrc=use_rx_rrc,
                    timing_offset_ts=timing_offset_ts,
                )
                # Add relative paths for UI
                gen = {}
                for k, v in res["paths"].items():
                     p = Path(v)
                     try:
                         gen[k] = str(p.relative_to(ROOT))
                     except:
                         gen[k] = str(p)
                res["paths"] = gen
                return jsonify(res)
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 400

        # --- Mode B: Interactive / Parameter-Based ---
        base_out = data.get("out_dir") or data.get("out") or DEFAULTS["out_lab3"]
        out_dir = str(_ts_dir(Path(base_out)))
        base_abs = (ROOT / out_dir).resolve() if not Path(out_dir).is_absolute() else Path(out_dir).resolve()
        
        # Params
        mod = (data.get("modulation") or "QPSK").upper()
        m_order = int(data.get("m_order") or 4)
        sps = int(data.get("sps") or 8)
        alpha = float(data.get("rolloff") if data.get("rolloff") is not None else (data.get("alpha") or 0.25))
        span = int(data.get("span") or 8)
        seed = int(data.get("seed") or 42)
        ebn0 = float(data.get("ebn0") or 10.0)
        channel_mode = (data.get("channel_mode") or "awgn").strip().lower()
        use_rx_rrc = str(data.get("use_rx_rrc") or "").lower() in {"1", "true", "on", "yes"}
        timing_offset_ts = float(data.get("timing_offset_ts") or 0.0)
        
        # Lab1 bits integration
        l1 = data.get("lab1") or {}
        bits = []
        l1_audit = {}
        
        if l1:
            try:
                audio = l1.get("audio") or DEFAULTS["audio"]
                text = l1.get("text") or DEFAULTS["text"]
                fs = int(l1.get("fs") or 16000)
                n_bits_l1 = int(l1.get("n_bits") or 8) # n_bits of quantization
                quantizer = (l1.get("quantizer") or "alaw").lower()
                source = (l1.get("source") or "audio").lower()
                method = (l1.get("method") or "scrambling").lower()
                l1_seed = _parse_int_auto(l1.get("lfsr_seed"), int("0b1010110011", 2))
                taps_str = (str(l1.get("lfsr_taps")) if l1.get("lfsr_taps") is not None else "9,6").strip()
                try:
                    l1_taps = tuple(int(t.strip()) for t in taps_str.split(',') if t.strip() != '')
                except:
                    l1_taps = (9, 6)
                l1_bitwidth = _parse_int_auto(l1.get("lfsr_bitwidth"), 10)
        # Generar figuras de Formateo para encadenado (una vez por corrida)
                
                # Build bits
                bits = _build_bits_from_lab1(audio, text, fs, n_bits_l1, quantizer, source, method,
                                             lfsr_seed=l1_seed, lfsr_taps=l1_taps, lfsr_bitwidth=l1_bitwidth)
                
                if bits:
                    l1_audit = {
                        "source": source, "method": method, "count": len(bits),
                        "audio": audio if source == "audio" else None,
                        "text": text if source == "text" else None
                    }
            except Exception as e:
                 return jsonify({"ok": False, "error": f"Formateo integration error: {e}"}), 400

        # If no bits from Lab 1, use n_bits random
        n_bits_sim = len(bits) if bits else int(data.get("n_bits") or 10000)
        
        try:
            if mod not in {"BPSK", "QPSK", "MPSK", "M-PSK"}:
                raise ValueError("Modulación no soportada (use BPSK, QPSK o M-PSK)")
            if mod in {"MPSK", "M-PSK"}:
                if m_order < 2 or (m_order & (m_order - 1)) != 0:
                    raise ValueError("Para M-PSK, M debe ser potencia de 2 y >= 2")
            params = lab3_demod.Lab3Params(
                out_dir=str(base_abs),
                n_bits=n_bits_sim,
                modulation=mod,
                m_order=m_order,
                sps=sps,
                rolloff=alpha,
                span=span,
                ebn0_start=ebn0, # Use start as THE value for single run
                seed=seed,
                channel_mode=channel_mode,
                use_rx_rrc=use_rx_rrc,
                timing_offset_ts=timing_offset_ts,
            )
            
            import numpy as _np
            bits_arr = _np.array(bits, dtype=_np.uint8) if bits else None
            
            res = lab3_demod.run_single(params, bits=bits_arr, lab1_meta=l1 if l1 else None)
            
            # Paths relative
            gen = {}
            for k, v in res["paths"].items():
                p = Path(v)
                try:
                    gen[k] = str(p.relative_to(ROOT))
                except:
                    gen[k] = str(p)
                    
            res["paths"] = gen
            res["ok"] = True
            res["audit"] = l1_audit
            
            return jsonify(res)
            
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.post("/api/lab2/run")
    def api_lab2_run():
        data = request.get_json(force=True) or {}
        base_out = data.get("out_dir") or DEFAULTS["out_lab2"]
        out_dir = str(_ts_dir(Path(base_out)))
        mod = (data.get("modulation") or "QPSK").upper()
        m_order = int(data.get("m_order") or 4)
        n_bits = int(data.get("n_bits") or 2000)
        sps = int(data.get("sps") or 8)
        alpha = float(data.get("alpha") or data.get("rolloff") or 0.25)
        span = int(data.get("span") or 8)
        seed = int(data.get("seed") or 0)
        try:
            if mod not in {"BPSK", "QPSK", "MPSK", "M-PSK"}:
                raise ValueError("Modulación no soportada (use BPSK, QPSK o M-PSK)")
            if mod in {"MPSK", "M-PSK"}:
                if m_order < 2 or (m_order & (m_order - 1)) != 0:
                    raise ValueError("Para M-PSK, M debe ser potencia de 2 (>=2)")
            params = lab2_rrc.Lab2Params(out_dir=out_dir, n_bits=n_bits, modulation=mod, m_order=m_order, sps=sps, rolloff=alpha, span=span, seed=seed)
            paths = lab2_rrc.run_lab2(params)
            try:
                _generate_shannon_theory_plots(Path(out_dir), bw_hz=100_000.0, rb_bps=None)
            except Exception:
                pass
            gen = {}
            for k, v in paths.items():
                p = Path(v)
                try:
                    gen[k] = str(p.relative_to(ROOT))
                except Exception:
                    gen[k] = str(p)
            base_abs = (ROOT / out_dir).resolve() if not Path(out_dir).is_absolute() else Path(out_dir).resolve()
            return jsonify({"ok": True, "out": str(base_abs), "paths": gen})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.post("/api/open_folder")
    def api_open_folder():
        data = request.get_json(silent=True) or {}
        rel = data.get("path") or request.args.get("path")
        if not rel:
            return jsonify({"ok": False, "error": "path requerido"}), 400
        p = Path(rel)
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        if p.is_file():
            p = p.parent
        # Restringir a outputs_ui
        if not str(p).startswith(str(OUTPUTS_ROOT.resolve())):
            return jsonify({"ok": False, "error": "Acceso denegado"}), 403
        if not p.exists() or not p.is_dir():
            return jsonify({"ok": False, "error": "Carpeta no encontrada"}), 404
        try:
            if sys.platform.startswith("darwin"):
                subprocess.Popen(["open", str(p)])
            elif sys.platform.startswith("win"):
                os.startfile(str(p))
            else:
                subprocess.Popen(["xdg-open", str(p)])
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.post("/api/pick_file")
    def api_pick_file():
        data = request.get_json(silent=True) or {}
        current_path = (data.get("path") or "").strip()
        kind = (data.get("kind") or "any").strip().lower()

        start_dir = Path(current_path).expanduser() if current_path else ROOT
        if start_dir.is_file():
            start_dir = start_dir.parent
        if not start_dir.exists():
            start_dir = ROOT

        try:
            if sys.platform.startswith("darwin"):
                type_clause = ""
                if kind == "audio":
                    type_clause = ' of type {"wav","wave"}'
                elif kind == "text":
                    type_clause = ' of type {"txt","md","text"}'
                start_posix = str(start_dir.resolve()).replace("\\", "\\\\").replace('"', '\\"')
                script = (
                    f'set chosenFile to choose file with prompt "Seleccionar archivo"{type_clause} '
                    f'default location POSIX file "{start_posix}/"\n'
                    'POSIX path of chosenFile'
                )
                res = subprocess.run(
                    ["osascript", "-e", script],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if res.returncode != 0:
                    err = (res.stderr or res.stdout or "").strip()
                    if "User canceled" in err or "cancel" in err.lower():
                        return jsonify({"ok": False, "canceled": True})
                    return jsonify({"ok": False, "error": err or "No se pudo seleccionar archivo"}), 500
                picked = (res.stdout or "").strip()
                return jsonify({"ok": True, "path": picked})

            return jsonify({"ok": False, "error": "Selector nativo no soportado en esta plataforma"}), 501
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.post("/api/pick_folder")
    def api_pick_folder():
        data = request.get_json(silent=True) or {}
        current_path = (data.get("path") or "").strip()

        start_dir = Path(current_path).expanduser() if current_path else ROOT
        if start_dir.is_file():
            start_dir = start_dir.parent
        if not start_dir.exists():
            start_dir = ROOT

        try:
            if sys.platform.startswith("darwin"):
                start_posix = str(start_dir.resolve()).replace("\\", "\\\\").replace('"', '\\"')
                script = (
                    f'set chosenFolder to choose folder with prompt "Seleccionar carpeta" '
                    f'default location POSIX file "{start_posix}/"\n'
                    'POSIX path of chosenFolder'
                )
                res = subprocess.run(
                    ["osascript", "-e", script],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if res.returncode != 0:
                    err = (res.stderr or res.stdout or "").strip()
                    if "User canceled" in err or "cancel" in err.lower():
                        return jsonify({"ok": False, "canceled": True})
                    return jsonify({"ok": False, "error": err or "No se pudo seleccionar carpeta"}), 500
                picked = (res.stdout or "").strip()
                return jsonify({"ok": True, "path": picked})

            return jsonify({"ok": False, "error": "Selector nativo no soportado en esta plataforma"}), 501
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.get("/api/zip")
    def api_zip():
        rel = request.args.get("path")
        if not rel:
            return jsonify({"ok": False, "error": "path requerido"}), 400
        p = Path(rel)
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        # Restringir a outputs_ui
        if not str(p).startswith(str(OUTPUTS_ROOT.resolve())):
            return jsonify({"ok": False, "error": "Acceso denegado"}), 403
        if not p.exists() or not p.is_dir():
            return jsonify({"ok": False, "error": "Carpeta no encontrada"}), 404
        try:
            tmp_dir = Path(tempfile.mkdtemp(prefix="cda_zip_"))
            base_name = tmp_dir / p.name
            zip_path = shutil.make_archive(str(base_name), "zip", root_dir=str(p))
            return send_file(zip_path, as_attachment=True, download_name=f"{p.name}.zip")
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.get("/api/lab2/subruns")
    def api_lab2_subruns():
        rel = request.args.get("path")
        if not rel:
            return jsonify({"ok": False, "error": "path requerido"}), 400
        p = Path(rel)
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        # Restringir a outputs_ui
        if not str(p).startswith(str(OUTPUTS_ROOT.resolve())):
            return jsonify({"ok": False, "error": "Acceso denegado"}), 403
        if not p.exists() or not p.is_dir():
            return jsonify({"ok": False, "error": "Carpeta no encontrada"}), 404
        runs = _collect_lab2_runs(p)
        rels = []
        for rel, _dir in runs:
            rels.append(rel)
        return jsonify({"ok": True, "runs": rels})

    @app.get("/api/files")
    def api_files():
        rel = request.args.get("path")
        if not rel:
            return jsonify({"ok": False, "error": "path requerido"}), 400
        p = (ROOT / rel).resolve()
        # Restringir a outputs_ui
        if not str(p).startswith(str(OUTPUTS_ROOT.resolve())):
            return jsonify({"ok": False, "error": "Acceso denegado"}), 403
        if not p.exists() or not p.is_file():
            return jsonify({"ok": False, "error": "Archivo no encontrado"}), 404
        return send_from_directory(p.parent, p.name, as_attachment=False)

    # Descarga de archivos de salida
    @app.get("/download")
    def download():
        out_dir = request.args.get("out")
        fname = request.args.get("file")
        if not out_dir or not fname:
            flash("Parámetros de descarga inválidos", "error")
            return redirect(url_for("index"))
        d = Path(out_dir)
        return send_from_directory(d, fname, as_attachment=True)

    # Servir figuras de salida
    @app.get("/fig/<path:subpath>")
    def serve_fig(subpath: str):
        # subpath esperado: outputs_ui/labX/figures/<file>
        print(f"DEBUG: serve_fig requested subpath: {subpath}")
        p = Path(subpath)
        # Asegurar path absoluto
        # Si subpath empieza con /, Path(subpath) lo toma como absoluto y (ROOT / p) ignora ROOT
        # Debemos asegurar que sea relativo a ROOT.
        if p.is_absolute():
            try:
                p = p.relative_to(ROOT)
            except ValueError:
                # Si no es relativo a ROOT, forzamos
                pass

        full_path = (ROOT / p).resolve()
        print(f"DEBUG: serve_fig full_path: {full_path}")

        directory = full_path.parent
        fname = full_path.name
        if not full_path.exists():
            print(f"ERROR: File not found: {full_path}")
            return "File not found", 404

        return send_file(full_path)

    return app



app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=True)
