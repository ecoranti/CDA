from __future__ import annotations

import os
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
from src import sqnr_eval as lab1_sqnr
from src import lab2_rrc
from src.audio_utils import load_wav_mono, uniform_quantize, mu_law_quantize, plot_hist_bits
from src.bits_utils import ints_to_bits, bits_to_bytes, bits_entropy_stats
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
                "title": "Formateo (Lab 1)",
                "desc": "Formateo y ecualización del histograma (scrambling/Huffman)",
                "link": url_for("lab1_page"),
            },
            {
                "id": "lab2",
                "title": "Modulación + RRC (Lab 2)",
                "desc": "Transmisor digital, mapeo IQ y pulso de Nyquist",
                "link": url_for("lab2_page"),
            },
            {
                "id": "lab3",
                "title": "Demodulación Digital (Lab 3)",
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
            _append_log(log_path, f"Lab1 start: audio={audio}, text={text}, fs={fs}, n_bits={n_bits}, quantizer={quantizer}")

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
            flash(f"Error al ejecutar Lab 1: {e}", "error")
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
            _append_log(Path(out_dir) / "run_log.txt", f"Lab2 run OK: {params}")
        except Exception as e:
            flash(f"Error al ejecutar Lab 2: {e}", "error")
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
        try:
            if sps <= 0:
                raise ValueError("sps debe ser > 0")
            if mod not in {"BPSK", "QPSK"}:
                raise ValueError("Modulación no soportada (use BPSK o QPSK)")
            if not (1 <= n_bits <= 16):
                raise ValueError("n_bits (Lab1) debe estar entre 1 y 16")
            if quantizer not in {"mulaw", "uniform"}:
                raise ValueError("quantizer (Lab1) inválido")
            # Construir bits desde lab1 con parámetros avanzados
            bits = _build_bits_from_lab1(audio, text, fs, n_bits, quantizer, source, method,
                                         mu=l1_mu, lfsr_seed=l1_seed, lfsr_taps=l1_taps, lfsr_bitwidth=l1_bitwidth)
            if len(bits) == 0:
                raise ValueError("La secuencia de bits construida está vacía")

            # Guardar bits binarios para trazabilidad
            bin_bytes = bytes(bits_to_bytes(bits))
            bf = base_abs / "bits_from_lab1.bin"
            bf.write_bytes(bin_bytes)
            # Histograma y métricas para auditar origen de bits
            try:
                plot_hist_bits(bits, "Bits (Lab1→Lab2)", str(base_abs / "l1_bits_hist.png"), as_probability=True)
            except Exception:
                pass
            from src.bits_utils import bits_entropy_stats as _bes
            p0, p1, H, var = _bes(bits)
            audit = {"bits": {"count": len(bits), "p0": float(p0), "p1": float(p1), "H": float(H), "var": float(var),
                               "source": source, "method": method}}
            # Info de archivos de entrada (md5, tamaño, duración)
            import hashlib
            if source in {"audio", "concat"}:
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
            if source in {"text", "concat"}:
                try:
                    h = hashlib.md5()
                    b = open(text, 'rb').read()
                    h.update(b)
                    preview = open(text, 'r', encoding='utf-8', errors='ignore').read(200)
                    audit["text"] = {"path": text, "md5": h.hexdigest(), "bytes": len(b), "preview": preview}
                except Exception as _e:
                    audit["text_error"] = str(_e)
            # Parámetros
            _write_json(base_abs / "params.json", {
                "lab2": {"modulation": mod, "sps": sps, "rolloff": alpha, "span": span, "seed": seed},
                "lab1": {"audio": audio, "text": text, "fs": fs, "n_bits": n_bits, "quantizer": quantizer,
                         "source": source, "method": method, "mu": l1_mu, "lfsr_seed": l1_seed,
                         "lfsr_taps": list(l1_taps), "lfsr_bitwidth": l1_bitwidth},
                "n_bits_effective": len(bits),
                "out": str(base_abs),
                "l1_metrics": audit["bits"],
            })

            params = lab2_rrc.Lab2Params(
                out_dir=str(base_abs), n_bits=len(bits), modulation=mod, sps=sps, rolloff=alpha, span=span, seed=seed
            )
            import numpy as _np
            paths = lab2_rrc.run_lab2(params, bits=_np.array(bits, dtype=_np.uint8))
            gen = {}
            for k, v in paths.items():
                p = Path(v)
                try:
                    gen[k] = str(p.relative_to(ROOT))
                except Exception:
                    gen[k] = str(p)
            try:
                gen["bits_bin"] = str(bf.relative_to(ROOT)) if bf.exists() else str(bf)
            except Exception:
                gen["bits_bin"] = str(bf)
            # agregar histograma si existe
            l1_hist = base_abs / "l1_bits_hist.png"
            if l1_hist.exists():
                try:
                    gen["l1_bits_hist"] = str(l1_hist.relative_to(ROOT))
                except Exception:
                    gen["l1_bits_hist"] = str(l1_hist)
            audit["params"] = {"mu": l1_mu, "lfsr_seed": l1_seed, "lfsr_taps": list(l1_taps), "lfsr_bitwidth": l1_bitwidth,
                                "fs": fs, "n_bits": n_bits, "quantizer": quantizer, "source": source, "method": method}
            return jsonify({"ok": True, "out": str(base_abs), "paths": gen, "n_bits": len(bits), "audit": audit})
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

    @app.get("/lab3")
    def lab3_page():
        return render_template(
            "lab3.html",
            defaults=DEFAULTS,
        )

    @app.post("/lab3/run")
    def lab3_run():
        base_out = request.form.get("out") or DEFAULTS["out_lab3"]
        out_dir = str(_ts_dir(Path(base_out)))
        mod = request.form.get("modulation") or "QPSK"
        n_bits = int(request.form.get("n_bits") or 10000)
        eb_start = float(request.form.get("eb_start") or 0.0)
        eb_end = float(request.form.get("eb_end") or 12.0)
        eb_step = float(request.form.get("eb_step") or 2.0)
        sps = int(request.form.get("sps") or 8)
        seed = int(request.form.get("seed") or 42)

        try:
            params = lab3_demod.Lab3Params(
                out_dir=out_dir,
                n_bits=n_bits,
                modulation=mod,
                sps=sps,
                ebn0_start=eb_start,
                ebn0_end=eb_end,
                ebn0_step=eb_step,
                seed=seed
            )
            res = lab3_demod.run_simulation(params)
            
            # Save params for UI
            _write_json(Path(out_dir) / "params.json", {
                "n_bits": n_bits,
                "modulation": mod,
                "eb_start": eb_start, 
                "eb_end": eb_end,
                "eb_step": eb_step,
                "sps": sps,
                "seed": seed,
                "out": out_dir
            })
            _append_log(Path(out_dir) / "run_log.txt", f"Lab3 run OK: {params}")

        except Exception as e:
            flash(f"Error al ejecutar Lab 3: {e}", "error")
            return redirect(url_for("lab3_page"))

        return redirect(url_for("lab3_results", out=out_dir))

    @app.get("/lab3/results")
    def lab3_results():
        out_dir = request.args.get("out") or DEFAULTS["out_lab3"]
        p = Path(out_dir)
        
        # Check specific files
        ber_plot_path = p / "ber_curve.png"
        ber_plot = str(ber_plot_path) if ber_plot_path.exists() else None
        ber_csv = "ber_results.csv" if (p / "ber_results.csv").exists() else None
        
        other = []
        for name in ["params.json", "run_log.txt"]:
            if (p / name).exists():
                other.append(name)
                
        return render_template("lab3_results.html", out=out_dir, ber_plot=ber_plot, ber_csv=ber_csv, other=other)

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
        base_out = data.get("out_dir") or DEFAULTS["out_lab3"]
        out_dir = str(_ts_dir(Path(base_out)))
        base_abs = (ROOT / out_dir).resolve() if not Path(out_dir).is_absolute() else Path(out_dir).resolve()
        
        # Params
        mod = (data.get("modulation") or "QPSK").upper()
        sps = int(data.get("sps") or 8)
        alpha = float(data.get("rolloff") or 0.25)
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
                 # If lab1 fail, we fall back to random? or error?
                 # Let's just log and proceed with random if bits empty
                 _append_log(Path(out_dir)/"run_log.txt", f"Lab1 integration warning: {e}")

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
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
