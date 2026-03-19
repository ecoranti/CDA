from __future__ import annotations

import os
import shutil
import threading
import uuid
import subprocess
import tempfile
from pathlib import Path
import sys
from typing import Optional

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
from src.audio_utils import load_wav_mono, uniform_quantize, mu_law_quantize, plot_hist_bits
from src.bits_utils import ints_to_bits
from src.scrambling import scramble
from src.huffman import encode
from src import lab3_demod


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = "cda-key"

    ROOT = Path(__file__).resolve().parent.parent


    DEFAULTS = {
            "audio": str(ROOT / "data/voice.wav"),
            "text": str(ROOT / "data/sample_text.txt"),
            "out_lab1": str(ROOT / "outputs_ui/lab1"),
            "out_lab2": str(ROOT / "outputs_ui/lab2"),
            "out_lab3": str(ROOT / "outputs_ui/lab3"),
        }

    OUTPUTS_ROOT = ROOT / "outputs_ui"

    def _ts_dir(base: Path) -> Path:
        import datetime
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
            if (base_dir / "params.json").exists() and _has_iq(base_dir):
                return [(".", base_dir)]
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
                               mu: int = 255,
                               lfsr_seed: int = 0b1010110011,
                               lfsr_taps: tuple[int, ...] = (9, 6),
                               lfsr_bitwidth: int = 10) -> list[int]:
        # source: 'audio'|'text'|'concat'
        # method: 'scrambling'|'huffman'
        bits_audio: list[int] = []
        bits_text: list[int] = []

        if source in {"audio", "concat"}:
            x, _ = load_wav_mono(audio, target_fs=fs)
            if quantizer == "mulaw":
                qA, _ = mu_law_quantize(x, bits=n_bits, mu=mu, xmin=-1, xmax=1)
            else:
                qA, _ = uniform_quantize(x, bits=n_bits, xmin=-1, xmax=1)
            base_bits = ints_to_bits(qA, n_bits)
            if method == "scrambling":
                bits_audio = [int(b) for b in scramble(base_bits, seed=lfsr_seed, taps=lfsr_taps, bitwidth=lfsr_bitwidth)]
            else:
                bits_huff, _, _ = encode(qA.tolist())
                bits_audio = [int(b) for b in bits_huff]

        if source in {"text", "concat"}:
            txt = open(text, "r", encoding="utf-8").read()
            b = txt.encode("utf-8")
            bytes_list = list(b)
            base_bits_b = []
            for val in bytes_list:
                for bit in range(7, -1, -1):
                    base_bits_b.append((val >> bit) & 1)
            if method == "scrambling":
                bits_text = [int(b) for b in scramble(base_bits_b, seed=lfsr_seed, taps=lfsr_taps, bitwidth=lfsr_bitwidth)]
            else:
                bits_huff_b, _, _ = encode(bytes_list)
                bits_text = [int(b) for b in bits_huff_b]

        if source == "audio":
            return bits_audio
        if source == "text":
            return bits_text
        # concat
        return (bits_audio or []) + (bits_text or [])

    def _generate_formateo_outputs(out_dir: Path, audio: str, text_path: str, fs: int, n_bits: int, quantizer: str,
                                  mu: int, lfsr_seed: int, lfsr_taps: tuple[int, ...], lfsr_bitwidth: int,
                                  hist_bins: int = 50, entropy_step_a: int = 10000, entropy_step_b: int = 500,
                                  sqnr_mus: list[int] | None = None) -> dict:
        """Genera las figuras de Formateo dentro de una carpeta específica y retorna rutas relativas."""
        out_dir.mkdir(parents=True, exist_ok=True)
        figdir = lab1.ensure_dirs(str(out_dir))

        rows = []
        rows.extend(lab1.process_audio(
            audio, figdir, fs_target=fs, n_bits=n_bits, quantizer=quantizer,
            mu_val=mu, lfsr_seed=lfsr_seed, lfsr_taps=lfsr_taps, lfsr_bitwidth=lfsr_bitwidth,
            hist_bins=hist_bins, entropy_step=entropy_step_a,
        ))
        rows.extend(lab1.process_text(
            text_path, figdir, lfsr_seed=lfsr_seed, lfsr_taps=lfsr_taps,
            lfsr_bitwidth=lfsr_bitwidth, entropy_step=entropy_step_b,
        ))

        # SQNR/MSE comparativos (igual que Formateo)
        try:
            x, _ = lab1.load_wav_mono(audio, target_fs=fs)
            xhat_uni = lab1_sqnr.eval_uniform(x, bits=n_bits)
            mse_uni = lab1_sqnr.mse(x, xhat_uni)
            sqnr_uni = lab1_sqnr.sqnr_db(x, xhat_uni)

            mu_vals = sqnr_mus or [1, 25, 50, 100, 255]
            mses, sqnrs = [], []
            for mu_val in mu_vals:
                xhat_mu = lab1_sqnr.eval_mulaw(x, mu=mu_val, bits=n_bits)
                mses.append(lab1_sqnr.mse(x, xhat_mu))
                sqnrs.append(lab1_sqnr.sqnr_db(x, xhat_mu))

            import matplotlib.pyplot as plt
            labels = ["Uniforme"] + [f"µ={mu_val}" for mu_val in mu_vals]
            plt.figure()
            plt.bar(range(len([sqnr_uni] + sqnrs)), [sqnr_uni] + sqnrs)
            plt.xticks(range(len(labels)), labels)
            plt.ylabel("SQNR (dB)")
            plt.title("Comparación de SQNR ({} bits)".format(n_bits))
            plt.tight_layout()
            plt.savefig(out_dir / "sqnr_comparacion.png", dpi=140)
            plt.close()

            plt.figure()
            plt.bar(range(len([mse_uni] + mses)), [mse_uni] + mses)
            plt.xticks(range(len(labels)), labels)
            plt.ylabel("MSE")
            plt.title("Comparación de MSE ({} bits)".format(n_bits))
            plt.tight_layout()
            plt.savefig(out_dir / "mse_comparacion.png", dpi=140)
            plt.close()
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
        for name in ["sqnr_comparacion.png", "mse_comparacion.png"]:
            p = out_dir / name
            if p.exists():
                try:
                    figs.append(str(p.relative_to(ROOT)))
                except Exception:
                    figs.append(str(p))
        return {"out": str(out_dir), "figs": figs}


    def _validate_lab1_inputs(audio: str, text: str, out_dir: str, fs: int, n_bits: int, quantizer: str,
                              mu: int, lfsr_seed: int, lfsr_bitwidth: int, hist_bins: int,
                              entropy_step_a: int, entropy_step_b: int, lfsr_taps: tuple[int, ...], sqnr_mus: list[int]):
        if not os.path.isfile(audio):
            raise ValueError(f"Audio no existe: {audio}")
        if not os.path.isfile(text):
            raise ValueError(f"Texto no existe: {text}")
        if fs <= 0:
            raise ValueError("fs debe ser > 0")
        if not (1 <= n_bits <= 16):
            raise ValueError("n_bits debe estar entre 1 y 16")
        if quantizer not in {"uniform", "mulaw", "both"}:
            raise ValueError("quantizer inválido")
        if mu <= 0:
            raise ValueError("mu debe ser > 0")
        if not (1 <= lfsr_bitwidth <= 32):
            raise ValueError("lfsr_bitwidth debe estar entre 1 y 32")
        if hist_bins <= 0:
            raise ValueError("hist_bins debe ser > 0")
        if entropy_step_a <= 0 or entropy_step_b <= 0:
            raise ValueError("entropy steps deben ser > 0")
        for t in lfsr_taps:
            if t < 0 or t >= lfsr_bitwidth:
                raise ValueError("lfsr_taps fuera de rango para bitwidth")
        if not sqnr_mus:
            raise ValueError("sqnr_mu list vacío")
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
                "desc": "Formateo y ecualización del histograma (scrambling/Huffman)",
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
        audio = request.form.get("audio") or DEFAULTS["audio"]
        text = request.form.get("text") or DEFAULTS["text"]
        base_out = request.form.get("out") or DEFAULTS["out_lab1"]
        fs = int(request.form.get("fs") or 16000)
        n_bits = int(request.form.get("n_bits") or 8)
        quantizer = request.form.get("quantizer") or "mulaw"
        mu = _parse_int_auto(request.form.get("mu"), 255)
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
        sqnr_mus_str = (request.form.get("sqnr_mus") or "1,25,50,100,255").strip()
        try:
            sqnr_mus = [int(v.strip()) for v in sqnr_mus_str.split(",") if v.strip() != ""]
        except Exception:
            sqnr_mus = [1, 25, 50, 100, 255]

        # Crear subcarpeta con timestamp para reproducibilidad
        out_dir = str(_ts_dir(Path(base_out)))

        try:
            _validate_lab1_inputs(audio, text, out_dir, fs, n_bits, quantizer,
                                  mu, lfsr_seed, lfsr_bitwidth, hist_bins, entropy_step_a, entropy_step_b, lfsr_taps, sqnr_mus)
            log_path = Path(out_dir) / "run_log.txt"
            _append_log(log_path, f"Formateo start: audio={audio}, text={text}, fs={fs}, n_bits={n_bits}, quantizer={quantizer}")

            figdir = lab1.ensure_dirs(out_dir)

            rows = []
            rows.extend(lab1.process_audio(audio, figdir, fs_target=fs, n_bits=n_bits, quantizer=quantizer,
                                           mu_val=mu, lfsr_seed=lfsr_seed, lfsr_taps=lfsr_taps, lfsr_bitwidth=lfsr_bitwidth,
                                           hist_bins=hist_bins, entropy_step=entropy_step_a))
            rows.extend(lab1.process_text(text, figdir, lfsr_seed=lfsr_seed, lfsr_taps=lfsr_taps,
                                          lfsr_bitwidth=lfsr_bitwidth, entropy_step=entropy_step_b))

            try:
                x, _ = lab1.load_wav_mono(audio, target_fs=fs)
                xhat_uni = lab1_sqnr.eval_uniform(x, bits=n_bits)
                mse_uni = lab1_sqnr.mse(x, xhat_uni)
                sqnr_uni = lab1_sqnr.sqnr_db(x, xhat_uni)

                mu_vals = sqnr_mus
                mses, sqnrs = [], []
                for mu in mu_vals:
                    xhat_mu = lab1_sqnr.eval_mulaw(x, mu=mu, bits=n_bits)
                    mses.append(lab1_sqnr.mse(x, xhat_mu))
                    sqnrs.append(lab1_sqnr.sqnr_db(x, xhat_mu))

                import matplotlib.pyplot as plt
                labels = ["Uniforme"] + [f"µ={mu}" for mu in mu_vals]
                # SQNR plot
                import os as _os
                plt.figure()
                plt.bar(range(len([sqnr_uni] + sqnrs)), [sqnr_uni] + sqnrs)
                plt.xticks(range(len(labels)), labels)
                plt.ylabel("SQNR (dB)")
                plt.title("Comparación de SQNR ({} bits)".format(n_bits))
                plt.tight_layout()
                plt.savefig(_os.path.join(out_dir, "sqnr_comparacion.png"), dpi=140)
                plt.close()

                # MSE plot
                plt.figure()
                plt.bar(range(len([mse_uni] + mses)), [mse_uni] + mses)
                plt.xticks(range(len(labels)), labels)
                plt.ylabel("MSE")
                plt.title("Comparación de MSE ({} bits)".format(n_bits))
                plt.tight_layout()
                plt.savefig(_os.path.join(out_dir, "mse_comparacion.png"), dpi=140)
                plt.close()

                import pandas as pd
                rows_sqnr = [("Uniforme", sqnr_uni, mse_uni)] + [
                    (f"µ={mu}", s, m) for mu, s, m in zip(mu_vals, sqnrs, mses)
                ]
                df_sqnr = pd.DataFrame(rows_sqnr, columns=["Cuantizador", "SQNR (dB)", "MSE"])
                df_sqnr.to_csv(os.path.join(out_dir, "sqnr_mse_resumen.csv"), index=False)
            except Exception as ex:
                _append_log(log_path, f"[WARN] SQNR/MSE: {ex}")
                flash(f"[WARN] No se pudo calcular SQNR/MSE: {ex}", "warning")

            lab1_report.save_metrics_csv(out_dir, rows)
            lab1_report.write_markdown(out_dir)
            try:
                from src.report import write_pdf as _write_pdf
                _write_pdf(out_dir)
            except Exception as _e:
                _append_log(Path(out_dir) / "run_log.txt", f"[WARN] PDF no generado: {_e}")
            # Guardar params
            _write_json(Path(out_dir) / "params.json", {
                "audio": audio,
                "text": text,
                "fs": fs,
                "n_bits": n_bits,
                "quantizer": quantizer,
                "mu": mu,
                "lfsr_seed": lfsr_seed,
                "lfsr_taps": list(lfsr_taps),
                "lfsr_bitwidth": lfsr_bitwidth,
                "hist_bins": hist_bins,
                "entropy_step_a": entropy_step_a,
                "entropy_step_b": entropy_step_b,
                "sqnr_mus": sqnr_mus,
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
        figs = []
        if figures_dir.exists():
            for p in sorted(figures_dir.glob("*.png")):
                figs.append(p.name)
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
        return render_template("lab1_results.html", out=out_dir, figs=figs, files=files, log_text=log_text)

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
        mod = request.form.get("modulation") or "QPSK"
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
            if mod.upper() not in {"BPSK", "QPSK"}:
                raise ValueError("Modulación no soportada (use BPSK o QPSK)")
            if eye_span < 1:
                raise ValueError("eye_span debe ser >= 1")
            if eye_traces <= 0:
                raise ValueError("eye_traces debe ser > 0")
            params = lab2_rrc.Lab2Params(
                out_dir=out_dir,
                n_bits=bits_len,
                modulation=mod.upper(),
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
                "modulation": mod.upper(),
                "sps": sps,
                "rolloff": alpha,
                "span": span,
                "seed": seed,
                "out": out_dir,
                "eye_span": eye_span,
                "eye_traces": eye_traces,
            })
            _append_log(Path(out_dir) / "run_log.txt", f"Modulación run OK: {params}")
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
        for name in ["iq_time.png", "constellation.png", "spectrum.png", "rrc_impulse.png", "eye_diagram.png", "l1_bits_hist.png", "bits_iq_transition.png"]:
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
        base_out = data.get("out_dir") or DEFAULTS["out_lab2"]
        out_dir = str(_ts_dir(Path(base_out)))
        base_abs = (ROOT / out_dir).resolve() if not Path(out_dir).is_absolute() else Path(out_dir).resolve()
        # Lab2 params
        mod = (data.get("modulation") or "QPSK").upper()
        sps = int(data.get("sps") or 8)
        alpha = float(data.get("rolloff") or 0.25)
        span = int(data.get("span") or 8)
        seed = int(data.get("seed") or 0)
        # Lab1 params
        l1 = data.get("lab1") or {}
        audio = l1.get("audio") or DEFAULTS["audio"]
        text = l1.get("text") or DEFAULTS["text"]
        fs = int(l1.get("fs") or 16000)
        n_bits = int(l1.get("n_bits") or 8)
        quantizer = (l1.get("quantizer") or "mulaw").lower()
        source = (l1.get("source") or "audio").lower()  # audio|text|concat
        method = (l1.get("method") or "scrambling").lower()  # scrambling|huffman
        l1_mu = _parse_int_auto(l1.get("mu"), 255)
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
            formateo_dir, audio, text, fs, n_bits, quantizer,
            l1_mu, l1_seed, l1_taps, l1_bitwidth
        )
        try:
            if sps <= 0:
                raise ValueError("sps debe ser > 0")
            if mod not in {"BPSK", "QPSK"}:
                raise ValueError("Modulación no soportada (use BPSK o QPSK)")
            if not (1 <= n_bits <= 16):
                raise ValueError("n_bits (Formateo) debe estar entre 1 y 16")
            if quantizer not in {"mulaw", "uniform"}:
                raise ValueError("quantizer (Formateo) inválido")
            if method not in {"scrambling", "huffman", "both"}:
                raise ValueError("method (Formateo) inválido")
            # Construir bits desde Formateo con parámetros avanzados
            import numpy as _np

            def _run_one(label: str, src: str, method_use: str, out_dir_use: Path):
                bits_local = _build_bits_from_lab1(
                    audio, text, fs, n_bits, quantizer, src, method_use,
                    mu=l1_mu, lfsr_seed=l1_seed, lfsr_taps=l1_taps, lfsr_bitwidth=l1_bitwidth
                )
                if len(bits_local) == 0:
                    raise ValueError(f"La secuencia de bits ({tag}) está vacía")

                out_dir_use.mkdir(parents=True, exist_ok=True)
                bf = out_dir_use / "bits_from_lab1.bin"
                _np.asarray(bits_local, dtype=_np.uint8).ravel().tofile(bf)
                bf_alias = out_dir_use / "bits_formateo.bin"
                _safe_alias(bf, bf_alias)

                try:
                    plot_hist_bits(bits_local, f"Bits (Formateo→Modulación) [{tag}]", str(out_dir_use / "l1_bits_hist.png"), as_probability=True)
                except Exception:
                    pass

                from src.bits_utils import bits_entropy_stats as _bes
                p0, p1, H, var = _bes(bits_local)
                audit = {
                    "bits": {"count": len(bits_local), "p0": float(p0), "p1": float(p1), "H": float(H), "var": float(var),
                             "source": src, "method": method_use}
                }

                import hashlib
                if src in {"audio", "concat"}:
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
                if src in {"text", "concat"}:
                    try:
                        h = hashlib.md5()
                        b = open(text, 'rb').read()
                        h.update(b)
                        preview = open(text, 'r', encoding='utf-8', errors='ignore').read(200)
                        audit["text"] = {"path": text, "md5": h.hexdigest(), "bytes": len(b), "preview": preview}
                    except Exception as _e:
                        audit["text_error"] = str(_e)

                _write_json(out_dir_use / "params.json", {
                    "lab2": {"modulation": mod, "sps": sps, "rolloff": alpha, "span": span, "seed": seed},
                    "lab1": {"audio": audio, "text": text, "fs": fs, "n_bits": n_bits, "quantizer": quantizer,
                             "source": src, "method": method_use, "mu": l1_mu, "lfsr_seed": l1_seed,
                             "lfsr_taps": list(l1_taps), "lfsr_bitwidth": l1_bitwidth},
                    "n_bits_effective": len(bits_local),
                    "out": str(out_dir_use),
                    "l1_metrics": audit["bits"],
                })

                params = lab2_rrc.Lab2Params(
                    out_dir=str(out_dir_use), n_bits=len(bits_local), modulation=mod, sps=sps, rolloff=alpha, span=span, seed=seed
                )
                paths = lab2_rrc.run_lab2(params, bits=_np.array(bits_local, dtype=_np.uint8))
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
                audit["params"] = {"mu": l1_mu, "lfsr_seed": l1_seed, "lfsr_taps": list(l1_taps), "lfsr_bitwidth": l1_bitwidth,
                                    "fs": fs, "n_bits": n_bits, "quantizer": quantizer, "source": src, "method": method_use}
                return {"paths": gen, "audit": audit, "n_bits": len(bits_local), "out": str(out_dir_use), "label": label}

            source_label = "Audio" if source == "audio" else ("Texto" if source == "text" else "Concat")
            methods = ["scrambling", "huffman"] if method == "both" else [method]

            if source == "separate":
                runs = {}
                for src in ["audio", "text"]:
                    src_label = "Audio" if src == "audio" else "Texto"
                    for m in methods:
                        key = f"{src}_{m}" if method == "both" else src
                        label = f"{src_label} - {m.title()}" if method == "both" else src_label
                        out_dir_use = base_abs / src / m if method == "both" else base_abs / src
                        runs[key] = _run_one(label, src, m, out_dir_use)
                _write_json(base_abs / "params.json", {
                    "mode": "separate",
                    "lab2": {"modulation": mod, "sps": sps, "rolloff": alpha, "span": span, "seed": seed},
                    "lab1": {"audio": audio, "text": text, "fs": fs, "n_bits": n_bits, "quantizer": quantizer,
                             "source": source, "method": method, "mu": l1_mu, "lfsr_seed": l1_seed,
                             "lfsr_taps": list(l1_taps), "lfsr_bitwidth": l1_bitwidth},
                    "out": str(base_abs),
                    "runs": {k: v["out"] for k, v in runs.items()},
                })
                return jsonify({"ok": True, "out": str(base_abs), "mode": "separate", "runs": runs, "formateo": formateo_payload})

            if method == "both":
                runs = {}
                for m in methods:
                    key = m
                    label = f"{source_label} - {m.title()}"
                    out_dir_use = base_abs / m
                    runs[key] = _run_one(label, source, m, out_dir_use)
                _write_json(base_abs / "params.json", {
                    "mode": "both",
                    "lab2": {"modulation": mod, "sps": sps, "rolloff": alpha, "span": span, "seed": seed},
                    "lab1": {"audio": audio, "text": text, "fs": fs, "n_bits": n_bits, "quantizer": quantizer,
                             "source": source, "method": method, "mu": l1_mu, "lfsr_seed": l1_seed,
                             "lfsr_taps": list(l1_taps), "lfsr_bitwidth": l1_bitwidth},
                    "out": str(base_abs),
                    "runs": {k: v["out"] for k, v in runs.items()},
                })
                return jsonify({"ok": True, "out": str(base_abs), "mode": "both", "runs": runs, "formateo": formateo_payload})

            # Modo normal (una sola fuente o concat)
            result = _run_one(source_label, source, method, base_abs)
            return jsonify({"ok": True, "out": str(base_abs), "paths": result["paths"], "n_bits": result["n_bits"], "audit": result["audit"], "formateo": formateo_payload})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    # ---------- API Endpoints ----------
    @app.post("/api/lab1/run")
    def api_lab1_run():
        data = request.get_json(force=True) or {}
        audio = data.get("audio") or DEFAULTS["audio"]
        text = data.get("text") or DEFAULTS["text"]
        base_out = data.get("out") or DEFAULTS["out_lab1"]
        fs = int(data.get("fs") or 16000)
        n_bits = int(data.get("n_bits") or 8)
        quantizer = data.get("quantizer") or "mulaw"
        mu = _parse_int_auto(data.get("mu"), 255)
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
        sqnr_mus_str = (data.get("sqnr_mus") or "1,25,50,100,255").strip()
        try:
            sqnr_mus = [int(v.strip()) for v in sqnr_mus_str.split(",") if v.strip() != ""]
        except Exception:
            sqnr_mus = [1, 25, 50, 100, 255]
        out_dir = str(_ts_dir(Path(base_out)))
        try:
            _validate_lab1_inputs(audio, text, out_dir, fs, n_bits, quantizer,
                                  mu, lfsr_seed, lfsr_bitwidth, hist_bins, entropy_step_a, entropy_step_b, lfsr_taps, sqnr_mus)
            figdir = lab1.ensure_dirs(out_dir)
            rows = []
            rows.extend(lab1.process_audio(audio, figdir, fs_target=fs, n_bits=n_bits, quantizer=quantizer,
                                           mu_val=mu, lfsr_seed=lfsr_seed, lfsr_taps=lfsr_taps, lfsr_bitwidth=lfsr_bitwidth,
                                           hist_bins=hist_bins, entropy_step=entropy_step_a))
            rows.extend(lab1.process_text(text, figdir, lfsr_seed=lfsr_seed, lfsr_taps=lfsr_taps,
                                          lfsr_bitwidth=lfsr_bitwidth, entropy_step=entropy_step_b))
            lab1_report.save_metrics_csv(out_dir, rows)
            lab1_report.write_markdown(out_dir)
            try:
                from src.report import write_pdf as _write_pdf
                _write_pdf(out_dir)
            except Exception as _e:
                pass
            _write_json(Path(out_dir) / "params.json", {
                "audio": audio,
                "text": text,
                "fs": fs,
                "n_bits": n_bits,
                "quantizer": quantizer,
                "mu": mu,
                "lfsr_seed": lfsr_seed,
                "lfsr_taps": list(lfsr_taps),
                "lfsr_bitwidth": lfsr_bitwidth,
                "hist_bins": hist_bins,
                "entropy_step_a": entropy_step_a,
                "entropy_step_b": entropy_step_b,
                "sqnr_mus": sqnr_mus,
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
            _write_json(p, {
                "job_id": job_id,
                "state": state,
                "progress": progress,
                "message": message,
                "out_dir": out_dir,
            })
        except Exception:
            pass


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
        sps: int,
        seed: int,
        progress_cb=None,
    ) -> str:
        out_dir = str(_ts_dir(Path(base_out)))
        use_l2_chain = True
        if trials <= 0:
            raise ValueError("trials_per_ebn0 debe ser > 0")
        l2 = lab2_path or _latest_lab2_output_dir(DEFAULTS["out_lab2"])
        if subrun:
            l2 = str(Path(l2).joinpath(subrun))
        if not l2:
            raise ValueError("No hay salida de Modulación disponible. Ejecutá Modulación primero.")
        l2_base = Path(l2)
        runs = _collect_lab2_runs(l2_base)
        if not runs:
            raise ValueError("No se encontraron corridas válidas en Modulación.")
        multi = len(runs) > 1 or (runs and runs[0][0] != ".")

        if multi:
            runs_meta = []
            total = len(runs)
            for i, (rel, rdir) in enumerate(runs, 1):
                if progress_cb:
                    progress_cb(i - 1, total, rel)
                out_sub = Path(out_dir) / rel
                out_sub.mkdir(parents=True, exist_ok=True)
                _ = lab3_demod.run_simulation_from_file(
                    lab2_dir=str(rdir),
                    out_dir=str(out_sub),
                    ebn0_start=eb_start,
                    ebn0_end=eb_end,
                    ebn0_step=eb_step,
                    trials_per_ebn0=trials,
                    seed=seed,
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
                "modulation": mod,
                "eb_start": eb_start,
                "eb_end": eb_end,
                "eb_step": eb_step,
                "trials_per_ebn0": trials,
                "sps": sps,
                "seed": seed,
                "out": out_dir,
                "runs": [r[0] for r in runs]
            })

            _append_log(Path(out_dir) / "run_log.txt", f"Canal y Rx run OK: mode=lab2_chain_multi runs={len(runs_meta)}")
            return out_dir

        # Single run
        rdir = runs[0][1]
        _ = lab3_demod.run_simulation_from_file(
            lab2_dir=str(rdir),
            out_dir=out_dir,
            ebn0_start=eb_start,
            ebn0_end=eb_end,
            ebn0_step=eb_step,
            trials_per_ebn0=trials,
            seed=seed,
        )

        _write_json(Path(out_dir) / "params.json", {
            "mode": "lab2_chain" if use_l2_chain else "standalone",
            "lab2_path": str(rdir) if use_l2_chain else None,
            "n_bits": n_bits,
            "modulation": mod,
            "eb_start": eb_start,
            "eb_end": eb_end,
            "eb_step": eb_step,
            "trials_per_ebn0": trials,
            "sps": sps,
            "seed": seed,
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
        mod = request.values.get("modulation") or "QPSK"
        n_bits = int(request.values.get("n_bits") or 10000)
        eb_start = float(request.values.get("eb_start") or 0.0)
        eb_end = float(request.values.get("eb_end") or 12.0)
        eb_step = float(request.values.get("eb_step") or 2.0)
        trials = int(request.values.get("trials_per_ebn0") or 20)
        sps = int(request.values.get("sps") or 8)
        seed = int(request.values.get("seed") or 42)

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
                sps=sps,
                seed=seed,
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
        mod = data.get("modulation") or "QPSK"
        n_bits = int(data.get("n_bits") or 10000)
        eb_start = float(data.get("eb_start") or 0.0)
        eb_end = float(data.get("eb_end") or 12.0)
        eb_step = float(data.get("eb_step") or 2.0)
        trials = int(data.get("trials_per_ebn0") or 20)
        sps = int(data.get("sps") or 8)
        seed = int(data.get("seed") or 42)

        job_id = uuid.uuid4().hex
        out_dir = str(_ts_dir(Path(base_out)))
        _write_lab3_status(out_dir, job_id, "running", 0.0, "Iniciando simulación")

        def _worker():
            try:
                def _progress(done, total, label):
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
                    sps=sps,
                    seed=seed,
                    progress_cb=_progress,
                )
                _write_lab3_status(out_dir, job_id, "done", 1.0, "Completado")
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

    @app.get("/lab3/results")
    def lab3_results():
        out_dir = request.args.get("out") or DEFAULTS["out_lab3"]
        p = Path(out_dir)

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
        
        other = []
        for name in ["params.json", "run_log.txt", "informe_lab3.md", "informe_lab3.pdf", "runs.json"]:
            if (p / name).exists():
                other.append(name)
                
        return render_template("lab3_results.html", out=out_dir, ber_plot=ber_plot, ber_csv=ber_csv, other=other, runs=runs)




    @app.post("/api/lab3/run_single")
    def api_lab3_run_single():
        data = request.get_json(force=True) or {}
        
        # --- Mode A: File-Based Integration (Lab 2 Output) ---
        if "lab2_path" in data and data["lab2_path"]:
            try:
                l2_path = data["lab2_path"]
                ebn0 = float(data.get("ebn0", 10.0))
                # Output dir
                out_base = data.get("out") or DEFAULTS["out_lab3"]
                out_ts = _ts_dir(Path(out_base))
                
                res = lab3_demod.run_from_file(l2_path, ebn0, str(out_ts))
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
        sps = int(data.get("sps") or 8)
        alpha = float(data.get("rolloff") if data.get("rolloff") is not None else (data.get("alpha") or 0.25))
        span = int(data.get("span") or 8)
        seed = int(data.get("seed") or 42)
        ebn0 = float(data.get("ebn0") or 10.0)
        
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
                quantizer = (l1.get("quantizer") or "mulaw").lower()
                source = (l1.get("source") or "audio").lower()
                method = (l1.get("method") or "scrambling").lower()
                l1_mu = _parse_int_auto(l1.get("mu"), 255)
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
                                             mu=l1_mu, lfsr_seed=l1_seed, lfsr_taps=l1_taps, lfsr_bitwidth=l1_bitwidth)
                
                if bits:
                    l1_audit = {
                        "source": source, "method": method, "count": len(bits),
                        "audio": audio if source in ["audio","concat"] else None,
                        "text": text if source in ["text","concat"] else None
                    }
            except Exception as e:
                 return jsonify({"ok": False, "error": f"Formateo integration error: {e}"}), 400

        # If no bits from Lab 1, use n_bits random
        n_bits_sim = len(bits) if bits else int(data.get("n_bits") or 10000)
        
        try:
            params = lab3_demod.Lab3Params(
                out_dir=str(base_abs),
                n_bits=n_bits_sim,
                modulation=mod,
                sps=sps,
                rolloff=alpha,
                span=span,
                ebn0_start=ebn0, # Use start as THE value for single run
                seed=seed
            )
            
            import numpy as _np
            bits_arr = _np.array(bits, dtype=_np.uint8) if bits else None
            
            res = lab3_demod.run_single(params, bits=bits_arr)
            
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
        n_bits = int(data.get("n_bits") or 2000)
        sps = int(data.get("sps") or 8)
        alpha = float(data.get("rolloff") or 0.25)
        span = int(data.get("span") or 8)
        seed = int(data.get("seed") or 0)
        try:
            params = lab2_rrc.Lab2Params(out_dir=out_dir, n_bits=n_bits, modulation=mod, sps=sps, rolloff=alpha, span=span, seed=seed)
            paths = lab2_rrc.run_lab2(params)
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
