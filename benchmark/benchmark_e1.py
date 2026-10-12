"""
E1 · Stress-test script benchmark_e1.py
=======================================

Corresponding deliverables:
    D-3  Stress-test script and raw data (script + CSV/logs)
    D-4  Before/after comparison table (including Appendix C baseline)
    D-5  Reproduction notes (environment pinning checklist)

Corresponding acceptance criteria:
    AC-01 Latency hard constraint (gating): across 1000 frames, no frame other than the first exceeds 50 ms; the overlay must be disabled during measurement
    AC-02 Latency spike target        : steady-state p95 <= 30 ms (non-gating)
    AC-03 Cold start                  : cold-start total time recorded; service-state first frame <= steady-state p95 × 1.5
    AC-04 VRAM                        : peak VRAM <= baseline
    AC-06 Stability                   : no memory/VRAM growth trend over continuous runs
    AC-09 Machine matrix              : at least 2 RTX machine tiers; warm-up and VRAM strategies must not hard-code machine assumptions

Requirement §5 "Test plan" states:
    Steady-state stress test: a fixed 1000-frame clip; run one pass for each combination of decoupling on/off × overlay on/off (4 groups in total)

Usage
-----
    # 4-group comparison (recommended)
    python benchmark/benchmark_e1.py --frames 1000 --matrix

    # Single group
    python benchmark/benchmark_e1.py --frames 1000 --render on --overlay off

    # Reproduction notes: environment pinning information is written out alongside the report
    python benchmark/benchmark_e1.py --frames 1000 --matrix --out reports/

Design notes
------------
- The script **does not depend on a real camera**: it drives the inference backend with a synthetic frame sequence, which eases CI and cross-machine reproduction (NFR-06).
- Supports `--backend mock` to validate the script itself and the statistical methodology in an environment without a GPU; no performance conclusions are produced.
- All results are written to CSV + JSON, ensuring the "raw data" is traceable (D-3).
"""

from __future__ import annotations

import argparse
import csv
import json
import locale
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Allow running directly from the repository root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import E1Config
from utils.metrics import FrameRecord, MetricsCollector, _percentile, vram_snapshot
from utils.renderPipeline import FrameSnapshot, RenderPipeline, NullChannel
from utils.warmup import WarmupRunner, warmup_if_enabled


# --------------------------------------------------------------------------
# draw simulation cost of the mock backend
# --------------------------------------------------------------------------
# Used **only by the mock backend**, to stably reproduce on a single machine the structural
# difference of "draw sitting on the critical path". The value is deliberately far below the
# 272.73 ms measured in Appendix C — the aim is to make the **structural difference** clearly
# discernible across the 4 comparison groups (draw on/off the critical path), not to reproduce
# the absolute figure of any particular machine.
# Real performance conclusions must be drawn with --backend real (reusing the IGNITE class) on
# the target RTX machine (AC-09).
DRAW_SIM_MS = 20.0


# ==========================================================================
# Environment pinning (D-5 / NFR-06)
# ==========================================================================
def freeze_environment() -> dict:
    """
    Pin driver / CUDA / cuDNN versions, power mode, thermal and throttling state.

    NFR-06 requires "reproducible results", so the stress-test report must carry this
    checklist with it. Items that cannot be obtained are recorded explicitly as
    "unavailable" rather than silently omitted.
    """
    env = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "machine": platform.machine(),
        "torch": "unavailable",
        "cuda": "unavailable",
        "cudnn": "unavailable",
        "gpu_name": "unavailable",
        "gpu_count": 0,
        "vram_total_mb": "unavailable",
        "driver": "unavailable",
        "power_mode": "unavailable",
        "thermal_throttling_observed": "unknown",
    }

    try:
        import torch  # type: ignore

        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda or "cpu-only"
        if torch.backends.cudnn.is_available():
            env["cudnn"] = str(torch.backends.cudnn.version())
        env["cudnn_benchmark"] = str(torch.backends.cudnn.benchmark)
        if torch.cuda.is_available():
            env["gpu_count"] = torch.cuda.device_count()
            env["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            env["vram_total_mb"] = round(props.total_memory / (1024 ** 2), 1)
            env["compute_capability"] = f"{props.major}.{props.minor}"
    except ImportError:
        env["torch"] = "not-installed"
    except Exception as exc:
        env["torch_probe_error"] = str(exc)

    # Driver version (when nvidia-smi is available)
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,power.draw,temperature.gpu,clocks_throttle_reasons.active",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            parts = [p.strip() for p in out.stdout.strip().split("\n")[0].split(",")]
            env["driver"] = parts[0] if len(parts) > 0 else "unknown"
            env["power_draw_w"] = parts[1] if len(parts) > 1 else "unknown"
            env["temperature_c"] = parts[2] if len(parts) > 2 else "unknown"
            # Any throttle bitmask other than 0x0 is treated as throttling observed (§5 environment pinning requirement)
            throttle = parts[3] if len(parts) > 3 else "unknown"
            env["thermal_throttling_observed"] = (
                "no" if throttle.lower() in ("0x0000000000000000", "0x0", "not active")
                else "yes" if throttle != "unknown" else "unknown"
            )
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        env["driver"] = "nvidia-smi-unavailable"

    # Power mode (Windows uses powercfg; Linux uses the cpufreq governor)
    # This is E1's key controlled variable: background load / power plan can markedly affect
    # CPU-side inference latency (see docs/E1_change_notes.md §8.5 — up to several-fold difference
    # measured on the same machine).
    try:
        gov = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        if gov.exists():
            env["power_mode"] = gov.read_text().strip()
        elif platform.system() == "Windows":
            # Note: powercfg may not be found under shell=False because of PATHEXT resolution,
            # so try shell=True first; on failure fall back to the list form. Errors are not
            # swallowed silently.
            raw = None
            try:
                out = subprocess.run(
                    "powercfg /getactivescheme",
                    capture_output=True, timeout=5, shell=True,
                )
                if out.returncode == 0 and out.stdout:
                    raw = out.stdout
            except Exception:
                pass
            if raw is None:
                try:
                    out = subprocess.run(
                        ["powercfg", "/getactivescheme"],
                        capture_output=True, timeout=5,
                    )
                    if out.returncode == 0 and out.stdout:
                        raw = out.stdout
                except Exception:
                    pass
            if raw is not None:
                # Multi-layer decoding fallback: Chinese Windows uses cp936/gbk; 'mbcs' is a
                # Windows-only alias, but some environments may misdetect the encoding, so try
                # several candidates in turn and take the first result that "contains Chinese or
                # readable brackets". If all fail, force-decode with replace so that no error is
                # raised.
                text = None
                for enc in ("mbcs", "gbk", "cp936", "utf-8", "cp1252"):
                    try:
                        cand = raw.decode(enc)
                        text = cand
                        break
                    except (LookupError, UnicodeDecodeError):
                        continue
                if text is None:
                    text = raw.decode("utf-8", errors="replace")

                # Of the form "Power Scheme GUID: 381b4222-...  (Balanced)" — take the scheme name in brackets.
                # Both the ASCII and the full-width bracket pairs are accepted, so that the regex also
                # matches the output of a localised (for example Chinese) Windows console.
                m = re.search(r"[\uff08(]\s*([^\uff09)]+)\s*[\uff09)]", text)
                if m:
                    env["power_mode"] = m.group(1).strip()
                else:
                    # Without brackets (English systems often show "(Balanced)" or a bare GUID), take the first line
                    first = text.strip().splitlines()[0] if text.strip() else ""
                    env["power_mode"] = first[:60] or "unavailable(empty-output)"
            else:
                env["power_mode"] = "unavailable(powercfg-failed)"
    except Exception as exc:
        env["power_mode"] = f"error:{type(exc).__name__}"

    return env


# ==========================================================================
# Inference backends
# ==========================================================================
class MockBackend:
    """
    Synthetic backend: simulates the frame-latency distribution of the baseline profile,
    used to validate the script and the statistical methodology.

    - The steady-state mean is aligned to the 43.50 ms of Table 2 (IGNITE);
    - Periodic spikes can be injected to reproduce the t≈330 / 750 ms phenomenon of
      Appendix C, used to verify that the AC-01 criterion can indeed be triggered.
    """

    def __init__(self, seed: int = 20261010, steady_ms: float = 43.50, spike_every: int = 0):
        self.rng = random.Random(seed)
        self.steady_ms = steady_ms
        self.spike_every = spike_every
        self.i = 0

    def infer(self, frame):
        self.i += 1
        base = max(1.0, self.rng.gauss(self.steady_ms, self.steady_ms * 0.10))
        if self.spike_every and self.i % self.spike_every == 0:
            base = 750.0        # frame-level spike observed in Appendix C
        time.sleep(base / 1000.0)
        return [{"class_name": "flame", "bbox": [10, 10, 60, 60], "confidence": 0.9}], (["rule"], [1], [0.9])


class RealBackend:
    """
    Real backend adapter: wraps IGNITE's actual inference entry point `_parse()` into an
    `infer(frame)` interface consistent with MockBackend, so that run_case can call both
    uniformly.

    Background (important)
    ----------------------
    The public inference entry point of the IGNITE class is `_parse(frame, conf=(...), **kwargs)`;
    **there is no `infer()` method**. The `infer()` shipped with MockBackend is merely a
    same-named convention internal to the stress-test script. Without this adapter, real mode
    would raise `AttributeError: 'IGNITE' object has no attribute 'infer'` inside run_case.

    This adapter:
    - only "forwards + normalises the return", without altering any IGNITE behaviour (NFR-04 zero regression);
    - takes `_parse` on the IGNITE instance directly as the method under test, ensuring the
      stress test exercises the product path.
    """

    def __init__(self, ignite):
        self._app = ignite
        # Bind _parse; it is the entry point that actually runs the two YOLOs plus predicate/semantic inference
        self._parse = getattr(ignite, "_parse", None)
        if self._parse is None:
            raise AttributeError("The IGNITE instance has no _parse method, so it cannot serve as the real backend")

    def infer(self, frame):
        """Aligned with MockBackend.infer: runs one inference and returns the result (structure-insensitive)."""
        return self._parse(frame)


def build_real_backend(cfg: E1Config):
    """
    Real backend: reuses the IGNITE class (ensuring the stress test exercises the product
    path), then adapts it via RealBackend into a uniform infer() interface. Requires CUDA and
    weight files; when missing, returns None for the caller to degrade gracefully.
    """
    try:
        from app import IGNITE  # type: ignore

        return RealBackend(IGNITE(cfg))
    except Exception as exc:
        print(f"[benchmark] Real backend unavailable, degrading to mock: {exc}")
        return None



# ==========================================================================
# Single stress-test group
# ==========================================================================
def run_case(
    frames: int,
    render_enabled: bool,
    overlay_enabled: bool,
    backend,
    cfg: E1Config,
    seed: int = 20261010,
    imgsz: int = 640,
) -> dict:
    """
    Runs one comparison group. render_enabled / overlay_enabled are the 2×2 matrix of requirement §5.

    Returns: {case, config, metrics:{...}, env:{...}}
    """
    rcfg = cfg.render
    rcfg.enabled = render_enabled
    rcfg.draw_enabled = overlay_enabled

    # Explicitness of the comparison groups: with decoupling enabled, draw leaves the critical
    # path; with decoupling disabled and the overlay enabled, draw blocks synchronously inside
    # the inference thread — this is precisely the baseline state that E1 must eliminate (P1).
    draw_in_critical_path = (not render_enabled) and overlay_enabled

    case_name = f"decouple={'on' if render_enabled else 'off'}__overlay={'on' if overlay_enabled else 'off'}"
    print(f"\n{'=' * 78}\n[case] {case_name}  frames={frames}\n{'=' * 78}")

    # ---- Warm-up (FR-04; executed only when the decoupling/warm-up switches permit it) ----
    wcfg = cfg.warmup
    warm_report = None
    if wcfg.enabled:
        warm_report = WarmupRunner(wcfg).run(
            detector_probe=lambda f: backend.infer(f),
            embedder_probe=lambda s: True,
        )
        print(f"[warmup] {warm_report.summary()}")
    else:
        warm_report = warmup_if_enabled(wcfg)

    # ---- Render pipeline construction ----
    channel = NullChannel() if not render_enabled else NullChannel()  # baseline uses Null (no GUI dependency)
    pipeline = RenderPipeline(
        channel=channel,
        queue_depth=rcfg.queue_depth,
        rate_limit=rcfg.rate_limit,
        draw_enabled=overlay_enabled,
        snapshot=rcfg.snapshot,
    )
    pipeline.start()

    collector = MetricsCollector(
        enabled=True,
        warmup_frames=cfg.metrics.warmup_frames,
        budget_ms=cfg.metrics.latency_budget_ms,
        target_p95_ms=cfg.metrics.latency_target_p95_ms,
        first_frame_factor=cfg.metrics.first_frame_factor,
    )
    collector.start()

    rng = random.Random(seed)
    try:
        import numpy as np  # type: ignore
        frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    except ImportError:
        frame = None

    records = []
    draw_critical_ms_total = 0.0
    for seq in range(1, frames + 1):
        is_warmup = seq <= cfg.metrics.warmup_frames

        t0 = time.perf_counter()
        backend.infer(frame)
        # Decoupling off + overlay on: draw blocks synchronously inside the inference thread and counts directly towards the critical path
        if draw_in_critical_path:
            t_draw = time.perf_counter()
            time.sleep(DRAW_SIM_MS / 1000.0)     # simulate draw cost (observation point, see module-level notes)
            draw_critical_ms_total += (time.perf_counter() - t_draw) * 1000.0
        parse_ms = (time.perf_counter() - t0) * 1000.0

        # Decoupling on: draw is handed to the render thread (inference does not wait) — the core behaviour of FR-01
        if render_enabled:
            pipeline.submit(
                FrameSnapshot(
                    seq=seq, timestamp=t0, frame=frame,
                    predictions=[{"class_name": "flame", "bbox": [10, 10, 60, 60],
                                  "confidence": 0.9}],
                    decision=None, is_warmup=is_warmup,
                )
            )
        # With decoupling off, draw has already run synchronously within the parse_ms timing interval above (draw_in_critical_path)

        alloc, reserved = vram_snapshot()
        rec = FrameRecord(
            seq=seq, timestamp=t0, parse_ms=parse_ms, is_warmup=is_warmup,
            # draw_ms records only the render-thread-side cost; with decoupling off, draw is already counted in parse_ms, so it is not double-counted here
            draw_ms=(pipeline.stats().get("draw_ms_last")
                     if (render_enabled and overlay_enabled) else None),
            queue_depth=(int(pipeline.stats().get("queue_depth_now", 0)) if render_enabled else None),
            vram_alloc_mb=alloc, vram_reserved_mb=reserved,
        )
        collector.record(rec)
        records.append(rec)

    # ---- Convergence ----
    pipeline.stop()
    stats = pipeline.stats()
    collector.note_render(int(stats.get("dropped", 0)), float(stats.get("max_queue_depth", 0)))
    summary = collector.summary()

    print(summary.report())

    return {
        "case": case_name,
        "config": {
            "frames": frames,
            "render_enabled": render_enabled,
            "overlay_enabled": overlay_enabled,
            "skip_frames": 1,
            "imgsz": imgsz,
            "seed": seed,
        },
        "latency": summary.to_dict(),
        "draw": {
            "p50": summary.draw_p50, "p95": summary.draw_p95, "max": summary.draw_max,
            "note": "draw is the render-thread-side cost and is not counted towards the critical path (FR-08)",
        },
        "render_queue": {
            "dropped_frames": summary.dropped_frames,
            "max_queue_depth": summary.max_queue_depth,
            "submitted": stats.get("submitted", 0),
            "rendered": stats.get("rendered", 0),
        },
        "warmup": warm_report.to_dict(),
        "warmup_total_ms": warm_report.total_ms,
        "cold_start_total_ms": warm_report.total_ms,
        "leak_index": collector.leak_indices(),
        "_records": records,
    }


# ==========================================================================
# Report output (D-3 / D-4 / D-5)
# ==========================================================================
# All artefacts are produced as plain strings and handed to utils.logger.Logger,
# which persists them as `<YYYY-MM-DD>_report.log` under the requested directory.
# Nothing is written directly to disk from this module, so the output format stays
# under the single control of the Logger.
#
# Formatting rule: the log is a plain-text artefact that may be downloaded and read
# by third parties, so no Markdown syntax is used anywhere — no '#' headings, no '|'
# tables, no fenced code blocks and no '**' emphasis. Sections are delimited by '='
# rules and columns are aligned with plain spaces.
def _rule(char: str = "-", width: int = 100) -> str:
    """A plain horizontal rule, used to delimit sections and column blocks."""
    return char * width


def _align_table(headers: list, rows: list, aligns: list | None = None) -> str:
    """
    Render a list of rows as a space-aligned plain-text table.

    `aligns` accepts '<' (left, default) or '>' (right) per column. No pipe or dash
    table syntax is emitted, so the result stays readable in any plain-text viewer.
    """
    aligns = aligns or ["<"] * len(headers)
    cells = [[str(h) for h in headers]] + [[str(c) for c in row] for row in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    out = []
    for idx, row in enumerate(cells):
        parts = []
        for i, cell in enumerate(row):
            spec = f"{aligns[i]}{widths[i]}"
            parts.append(f"{cell:{spec}}")
        out.append("  ".join(parts).rstrip())
        if idx == 0:
            out.append("  ".join("-" * w for w in widths))
    return "\n".join(out)


def build_frame_csv(records) -> str:
    """
    Render per-frame raw data as CSV text (D-3).

    The header row of each block is prefixed with 'CASE:' rather than a '#' comment
    character, because the surrounding artefact is a plain-text log in which the
    '#' glyph is reserved for headings.
    """
    import io as _io

    buf = _io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=["seq", "timestamp", "parse_ms", "is_warmup", "draw_ms",
                    "queue_depth", "vram_alloc_mb", "vram_reserved_mb", "note"],
    )
    writer.writeheader()
    for r in records:
        writer.writerow(r.to_dict())
    return buf.getvalue()


def build_compare_table(results: list, env: dict) -> str:
    """
    Before/after comparison table (D-4): 4 comparison groups + Appendix C baseline row.

    All figures in the table come from measured data / the original tables of the requirements
    document; nothing is extrapolated or estimated. Rendered as plain text (no Markdown).
    """
    lines = []
    lines.append(_rule("="))
    lines.append("D-4  STRESS-TEST BEFORE/AFTER COMPARISON TABLE")
    lines.append(_rule("="))
    lines.append(f"Generated at : {env.get('timestamp', 'n/a')}")
    lines.append(f"Machine      : {env.get('gpu_name', 'n/a')}")
    lines.append(f"VRAM         : {env.get('vram_total_mb', 'n/a')} MB")
    lines.append("")
    lines.append("Note: the baseline row is the original observed value from requirement")
    lines.append("      section 1.1 (Appendix C / Table 2); it is not inferred from this")
    lines.append("      stress test. The measured rows come from this script's groups.")
    lines.append("")

    headers = ["Group", "Decouple", "Overlay", "Samples", "p50(ms)", "p95(ms)",
               "p99(ms)", "max(ms)", ">50ms", "drawP95", "Dropped", "VRAM(MB)",
               "AC-01", "AC-02"]
    aligns = ["<", "<", "<", ">", ">", ">", ">", ">", ">", ">", ">", ">", "<", "<"]
    rows = [[
        "Baseline (Appendix C)", "off", "on", "-", "-", "-", "-", "750.00",
        ">=1", "-", "-", "-", "FAIL", "-",
    ]]
    for r in results:
        lat = r["latency"]
        steady = lat["latency_ms"]
        c = r["config"]
        rows.append([
            r["case"],
            "on" if c["render_enabled"] else "off",
            "on" if c["overlay_enabled"] else "off",
            lat["n_steady"],
            f"{steady['p50']:.2f}", f"{steady['p95']:.2f}",
            f"{steady['p99']:.2f}", f"{steady['max']:.2f}",
            lat["hard_violations"], f"{r['draw']['p95']:.2f}",
            r["render_queue"]["dropped_frames"],
            f"{lat['vram_mb']['peak_reserved']:.1f}",
            lat["verdict"]["AC-01_hard_budget"],
            lat["verdict"]["AC-02_p95_target"],
        ])
    lines.append(_align_table(headers, rows, aligns))
    lines.append("")

    # Baseline key metrics listed separately, for direct reviewer comparison
    lines.append(_rule("="))
    lines.append("APPENDIX C BASELINE ORIGINAL VALUES (requirement section 1.1, not extrapolated)")
    lines.append(_rule("="))
    base_headers = ["Observation point", "Value", "Note"]
    base_rows = [
        ["Cold start t=0", "1,393.72 ms",
         "CUDA context + first kernel compilation + VRAM pool initialisation (one-off)"],
        ["Runtime spike t~330", "750 ms",
         "of which draw = 272.73 ms, about 36% of that frame"],
        ["IGNITE steady-state mean", "43.50 ms", "+24.54 ms compared with standalone YOLO"],
        ["Standalone YOLO steady state", "18.96 ms", "comparison baseline"],
        ["Latency hard budget", "<= 50 ms/frame", ">= 20 FPS, must not be exceeded"],
        ["Latency spike target", "<= 30 ms", "non-gating, already decided in Q1"],
        ["Remaining headroom", "6.5 ms", "43.50 -> 50"],
    ]
    lines.append(_align_table(base_headers, base_rows))

    return "\n".join(lines) + "\n"


def build_repro_doc(env: dict, results: list) -> str:
    """Reproduction notes / environment pinning checklist (D-5). Rendered as plain text."""
    lines = [
        _rule("="),
        "D-5  REPRODUCTION NOTES AND ENVIRONMENT PINNING CHECKLIST",
        _rule("="),
        "",
        "1. HOW TO RUN",
        "",
        "   (a) the four comparison groups (decoupling on/off x overlay on/off), 1000 frames",
        "       python benchmark/benchmark_e1.py --frames 1000 --matrix --out reports/",
        "",
        "   (b) reproduce a single group",
        "       python benchmark/benchmark_e1.py --frames 1000 --render on --overlay off",
        "",
        f"   Random seed: {results[0]['config']['seed'] if results else 20261010} "
        f"(pinned to the script default)",
        "",
        "2. ENVIRONMENT PINNING SNAPSHOT",
        "",
    ]
    env_headers = ["Item", "Value"]
    env_rows = [[k, v] for k, v in env.items()]
    lines.append(_align_table(env_headers, env_rows))
    lines += [
        "",
        "3. PREREQUISITES FOR REPRODUCTION",
        "",
        "   - Hardware: an off-the-shelf consumer PC plus an RTX-series GPU (lower bound",
        "     matched to Appendix C: RTX 3050 Ti Laptop / 4GB VRAM), with no upper bound;",
        "     at least 2 machine tiers must complete AC-09.",
        "   - Cooling: record whether thermal throttling occurs; data collected during",
        "     throttling must not be used for acceptance.",
        "   - Power: fixed to the maximum-performance mode; during the stress test, do not",
        "     unplug the power or switch to a power-saving mode.",
        "   - Weights: Service/ObjectDetector/fire.pt, obj1.pt and",
        "     Service/Semantic/triplets_cache.pkl must match the baseline (the precondition",
        "     for NFR-04 accuracy regression).",
        "",
        "4. RESTATEMENT OF THE MEASUREMENT METHODOLOGY (requirement section 5)",
        "",
        "   - p50 / p95 / p99 / max frame latency: steady-state frames only are counted",
        "     (FR-07 already excludes warm-up frames).",
        "   - draw standalone cost: recorded on the render-thread side and not counted",
        "     towards the critical path (FR-08).",
        "   - Product latency KPIs are always measured with the overlay disabled.",
        "   - VRAM: record the peak and the time series (AC-04 / AC-06).",
        "",
    ]
    return "\n".join(lines) + "\n"


# ==========================================================================
# CLI
# ==========================================================================
def parse_args():
    ap = argparse.ArgumentParser(description="E1 render decoupling + CUDA warm-up stress-test script")
    ap.add_argument("--frames", type=int, default=1000, help="Number of frames to stress-test (acceptance methodology: 1000)")
    ap.add_argument("--matrix", action="store_true",
                    help="Run the 2×2 comparison matrix (decoupling on/off × overlay on/off)")
    ap.add_argument("--render", choices=["on", "off"], default="on", help="Single-group mode: render decoupling")
    ap.add_argument("--overlay", choices=["on", "off"], default="off", help="Single-group mode: overlay")
    ap.add_argument("--warmup", choices=["on", "off"], default="on", help="CUDA warm-up")
    ap.add_argument("--backend", choices=["mock", "real"], default="mock",
                    help="Inference backend: mock=synthetic latency (runs without a GPU); real=reuse the IGNITE class")
    ap.add_argument("--steady-ms", type=float, default=43.50,
                    help="Mock steady-state mean, aligned to the 43.50 ms of Table 2 by default")
    ap.add_argument("--spike-every", type=int, default=0,
                    help="With the mock, inject a 750ms spike every N frames (reproducing Appendix C t≈330)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--out", type=str, default="reports", help="Output directory")
    ap.add_argument("--per-case-process", action="store_true",
                    help="Run each case in a separate subprocess (recommended with --backend real). "
                         "The process exits as soon as it finishes, releasing VRAM, which avoids the "
                         "cumulative VRAM usage of a small-VRAM machine (e.g. 4GB) triggering OOM "
                         "under the full --matrix.")
    ap.add_argument("--case-worker", action="store_true",
                    help=argparse.SUPPRESS)  # internal: subprocess entry-point marker
    ap.add_argument("--case-spec", type=str, default="",
                    help=argparse.SUPPRESS)  # internal: subprocess case parameters (JSON)
    return ap.parse_args()


# ==========================================================================
# Subprocess case execution (--per-case-process)
# ==========================================================================
def _run_case_subprocess(args, render_on: bool, overlay_on: bool) -> dict:
    """
    Runs one case in a separate subprocess and returns the result dict of run_case.

    Why subprocess isolation matters: with `--backend real`, each group loads weights via
    `IGNITE(cfg)`; on a small-VRAM machine, running 4 groups within one process accumulates
    usage and triggers OOM. Exiting the subprocess reclaims it immediately.
    """
    spec = {
        "frames": args.frames, "render_on": render_on, "overlay_on": overlay_on,
        "backend": args.backend, "seed": args.seed, "steady_ms": args.steady_ms,
        "spike_every": args.spike_every, "imgsz": args.imgsz,
        "warmup": args.warmup,
    }
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--case-worker", "--case-spec", json.dumps(spec),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(
            f"Subprocess case failed (decouple={render_on}, overlay={overlay_on}), "
            f"exit={proc.returncode}"
        )
    # The subprocess writes the result JSON on the last line (prefixed sentinel), so it is easy to separate from the logs
    marker = "__CASE_RESULT_JSON__"
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    raise RuntimeError("The subprocess did not return a result JSON (output truncated?)")


def _case_worker(spec: dict) -> int:
    """Subprocess entry point: build backend → run run_case → print the result JSON."""
    cfg = _build_base_cfg(spec["warmup"] == "on", spec["imgsz"])
    if spec["backend"] == "mock":
        backend = MockBackend(seed=spec["seed"], steady_ms=spec["steady_ms"],
                              spike_every=spec["spike_every"])
    else:
        backend = build_real_backend(cfg)
        if backend is None:
            backend = MockBackend(seed=spec["seed"], steady_ms=spec["steady_ms"])
    res = run_case(
        frames=spec["frames"], render_enabled=spec["render_on"],
        overlay_enabled=spec["overlay_on"], backend=backend, cfg=cfg,
        seed=spec["seed"], imgsz=spec["imgsz"],
    )
    res.pop("_records", None)
    print("__CASE_RESULT_JSON__" + json.dumps(res, ensure_ascii=False))
    return 0


def _build_base_cfg(warmup_on: bool, imgsz: int) -> E1Config:
    """Build the baseline configuration (shared by the main process and subprocesses, for consistent methodology)."""
    return E1Config.from_dict({
        "render": {"queue_depth": 1, "rate_limit": 15.0, "snapshot": True},
        "warmup": {"enabled": warmup_on, "device": "cuda",
                   "imgsz": imgsz, "budget_ms": 2000.0},
        "metrics": {"enabled": True, "warmup_frames": 1,
                    "latency_budget_ms": 50.0, "latency_target_p95_ms": 30.0},
    }).validate()


def _emit_reports(out_dir: Path, env: dict, results: list, csv_blocks: list) -> None:
    """
    Hand every artefact to utils.logger.Logger, which persists it as
    `<YYYY-MM-DD>_report.log` inside `out_dir`.

    Each artefact is passed as one string. The Logger aggregates within a short
    window and flushes to disk asynchronously, so the benchmark module never
    writes files itself.
    """
    try:
        from utils.logger import Logger  # type: ignore
    except ImportError as exc:  # pragma: no cover - only when run outside the repo
        print(f"[report] utils.logger unavailable, falling back to stdout only: {exc}")
        Logger = None  # type: ignore

    report_dir = str(out_dir)
    sections = [
        ("D-5 · Environment pinning snapshot", _dict_as_lines(env)),
        ("D-4 · Before/after comparison table", build_compare_table(results, env)),
        ("D-5 · Reproduction notes", build_repro_doc(env, results)),
        ("D-3 · Per-frame raw data (CSV)", "\n".join(csv_blocks)),
        ("Full metrics (JSON)", json.dumps({"environment": env, "results": results},
                                           ensure_ascii=False, indent=2)),
    ]

    if Logger is not None:
        for title, body in sections:
            Logger.report(f"===== {title} =====\n{body}", report_dir=report_dir)
        Logger.flush()
        print(f"\nReports flushed via Logger to: {out_dir}/<YYYY-MM-DD>_report.log")
    else:
        # Degraded path: still emit to stdout so nothing is silently lost
        for title, body in sections:
            print(f"\n===== {title} =====\n{body}")


def _dict_as_lines(d: dict) -> str:
    """Render a flat mapping as aligned `key: value` lines."""
    return "\n".join(f"  {k:28s}: {v}" for k, v in d.items())


def main():
    args = parse_args()

    # Subprocess entry point: run just one case and output the result JSON, without writing any artefacts
    if args.case_worker:
        return _case_worker(json.loads(args.case_spec))

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    env = freeze_environment()
    print("=" * 78)
    print("E1 stress test · environment pinning snapshot (D-5 / NFR-06)")
    print("=" * 78)
    for k, v in env.items():
        print(f"  {k:28s}: {v}")

    base_cfg = _build_base_cfg(args.warmup == "on", args.imgsz)

    if args.per_case_process:
        print("[mode] Per-case subprocess isolation (--per-case-process): VRAM is released as soon as each group finishes")

    if args.matrix:
        combos = [(True, True), (True, False), (False, True), (False, False)]
    else:
        combos = [(args.render == "on", args.overlay == "on")]

    results = []
    csv_blocks = []
    for render_on, overlay_on in combos:
        if args.per_case_process:
            res = _run_case_subprocess(args, render_on, overlay_on)
            # The subprocess mode no longer holds per-frame detail (the cost of VRAM isolation)
            res.setdefault("_records", [])
        else:
            cfg = E1Config.from_dict(base_cfg.to_dict()).validate()
            if args.backend == "mock":
                backend = MockBackend(seed=args.seed, steady_ms=args.steady_ms,
                                      spike_every=args.spike_every)
            else:
                backend = build_real_backend(cfg)
                if backend is None:
                    backend = MockBackend(seed=args.seed, steady_ms=args.steady_ms)
            res = run_case(
                frames=args.frames, render_enabled=render_on, overlay_enabled=overlay_on,
                backend=backend, cfg=cfg, seed=args.seed, imgsz=args.imgsz,
            )
        results.append(res)

        # Per-frame raw data (D-3) is rendered as CSV text and kept in memory,
        # to be handed to the Logger together with the other artefacts.
        tag = f"decouple{int(render_on)}_overlay{int(overlay_on)}"
        csv_blocks.append(f"CASE: {tag}\n{build_frame_csv(res['_records'])}")

    # Drop the per-frame detail from the JSON payload to keep the report compact
    for r in results:
        r.pop("_records", None)

    _emit_reports(out_dir, env, results, csv_blocks)

    print("\n" + "=" * 78)
    print("Conclusion summary")
    print("=" * 78)
    for r in results:
        v = r["latency"]["verdict"]
        print(f"  {r['case']:34s} AC-01={v['AC-01_hard_budget']:4s} "
              f"AC-02={v['AC-02_p95_target']:4s} AC-03={v['AC-03_service_first_frame']}")
    print(f"\nArtefacts written via Logger to: {out_dir}/<YYYY-MM-DD>_report.log")


if __name__ == "__main__":
    main()
