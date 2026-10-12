"""
E1 · Per-frame metrics and stability collection metrics
=======================================================

Corresponding requirement items:
    FR-07  First-frame exception labelling: the first frame / warm-up frames are
           explicitly labelled in the metrics and excluded from steady-state statistics
    NFR-01 Dual latency bands: hard constraint any frame <= 50 ms (the first frame
           excepted); stretch target steady-state p95 <= 30 ms
    NFR-06 Reproducibility: stress-test scripts, random seed, driver/CUDA versions and
           power mode pinned
    AC-01  No frame other than the first exceeds 50 ms in a 1000-frame stress test; the
           overlay is disabled during measurement
    AC-02  Steady-state p95 <= 30 ms (non-gating)
    AC-06  30 min of continuous running with no memory/VRAM growth trend (leak check)

Metric definitions (requirements §5)
------------------------------------
    p50 / p95 / p99 / max frame latency, per-item draw duration, render queue depth,
    peak VRAM, per-stage warm-up duration
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ==========================================================================
# Per-frame record
# ==========================================================================
@dataclass
class FrameRecord:
    """
    Complete metrics for a single frame.

    Note the separation of `parse_ms` and `draw_ms` (FR-08):
    - `parse_ms` is the inference critical-path duration, **excluding** draw; the product
      latency KPI refers to it alone;
    - `draw_ms` is the render-thread-side duration, for observation only, not entering
      the critical path.
    """

    seq: int
    timestamp: float
    parse_ms: float
    is_warmup: bool = False
    draw_ms: Optional[float] = None
    queue_depth: Optional[int] = None
    vram_alloc_mb: Optional[float] = None
    vram_reserved_mb: Optional[float] = None
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq": self.seq,
            "timestamp": round(self.timestamp, 6),
            "parse_ms": round(self.parse_ms, 3),
            "is_warmup": self.is_warmup,
            "draw_ms": None if self.draw_ms is None else round(self.draw_ms, 3),
            "queue_depth": self.queue_depth,
            "vram_alloc_mb": None if self.vram_alloc_mb is None else round(self.vram_alloc_mb, 2),
            "vram_reserved_mb": None if self.vram_reserved_mb is None else round(self.vram_reserved_mb, 2),
            "note": self.note,
        }


# ==========================================================================
# Aggregate statistics
# ==========================================================================
def _percentile(sorted_values: List[float], q: float) -> float:
    """
    Linearly interpolated percentile (consistent with the numpy.percentile default
    convention, avoiding a numpy dependency).
    q is in the range 0~100.
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    idx = (len(sorted_values) - 1) * (q / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = idx - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


@dataclass
class MetricsSummary:
    """Statistical summary of a stress test / run, mapping directly to the AC-01 / AC-02 criteria."""

    n_total: int = 0
    n_steady: int = 0
    n_warmup: int = 0

    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    max: float = 0.0
    mean: float = 0.0
    min: float = 0.0

    first_frame_ms: Optional[float] = None

    budget_ms: float = 50.0
    target_p95_ms: float = 30.0
    hard_violations: int = 0          # number of steady-state frames with parse_ms > budget_ms
    first_frame_factor: float = 1.5

    draw_p50: float = 0.0
    draw_p95: float = 0.0
    draw_max: float = 0.0
    dropped_frames: int = 0
    max_queue_depth: float = 0.0

    vram_peak_alloc_mb: float = 0.0
    vram_peak_reserved_mb: float = 0.0

    duration_s: float = 0.0

    # ---------------- Criteria ----------------
    @property
    def ac01_pass(self) -> bool:
        """AC-01 (gating): no frame other than the first exceeds 50 ms."""
        return self.hard_violations == 0

    @property
    def ac02_pass(self) -> bool:
        """AC-02 (non-gating): steady-state p95 <= 30 ms."""
        return self.p95 <= self.target_p95_ms

    @property
    def ac03_service_first_frame_pass(self) -> bool:
        """AC-03: serving-state first frame <= steady-state p95 x 1.5."""
        if self.first_frame_ms is None:
            return False
        ceiling = self.p95 * self.first_frame_factor if self.p95 > 0 else self.target_p95_ms
        return self.first_frame_ms <= max(ceiling, self.target_p95_ms)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_total": self.n_total,
            "n_steady": self.n_steady,
            "n_warmup": self.n_warmup,
            "latency_ms": {
                "p50": round(self.p50, 3),
                "p95": round(self.p95, 3),
                "p99": round(self.p99, 3),
                "max": round(self.max, 3),
                "mean": round(self.mean, 3),
                "min": round(self.min, 3),
                "first_frame": None if self.first_frame_ms is None
                else round(self.first_frame_ms, 3),
            },
            "thresholds": {
                "budget_ms": self.budget_ms,
                "target_p95_ms": self.target_p95_ms,
                "first_frame_factor": self.first_frame_factor,
            },
            "hard_violations": self.hard_violations,
            "draw_ms": {
                "p50": round(self.draw_p50, 3),
                "p95": round(self.draw_p95, 3),
                "max": round(self.draw_max, 3),
            },
            "render_queue": {
                "dropped_frames": self.dropped_frames,
                "max_queue_depth": self.max_queue_depth,
            },
            "vram_mb": {
                "peak_alloc": round(self.vram_peak_alloc_mb, 2),
                "peak_reserved": round(self.vram_peak_reserved_mb, 2),
            },
            "duration_s": round(self.duration_s, 3),
            "verdict": {
                "AC-01_hard_budget": "PASS" if self.ac01_pass else "FAIL",
                "AC-02_p95_target": "PASS" if self.ac02_pass else "FAIL",
                "AC-03_service_first_frame": "PASS" if self.ac03_service_first_frame_pass else "FAIL",
            },
        }

    def report(self) -> str:
        """Human-readable verdict lines."""
        lines = [
            f"samples: total={self.n_total} steady={self.n_steady} warmup={self.n_warmup}",
            f"critical-path latency(ms): p50={self.p50:.2f} p95={self.p95:.2f} "
            f"p99={self.p99:.2f} max={self.max:.2f} mean={self.mean:.2f}",
            f"draw(ms): p95={self.draw_p95:.2f} max={self.draw_max:.2f} "
            f"(observation only, not counted in the critical path)",
            f"hard constraint(<= {self.budget_ms:.0f} ms): violations={self.hard_violations} "
            f"-> AC-01 {'PASS' if self.ac01_pass else 'FAIL'}",
            f"stretch target(p95 <= {self.target_p95_ms:.0f} ms): "
            f"-> AC-02 {'PASS' if self.ac02_pass else 'FAIL'}",
        ]
        if self.first_frame_ms is not None:
            lines.append(
                f"serving-state first frame={self.first_frame_ms:.2f} ms -> AC-03 "
                f"{'PASS' if self.ac03_service_first_frame_pass else 'FAIL'}"
            )
        return "\n".join(lines)


# ==========================================================================
# Collector
# ==========================================================================
class MetricsCollector:
    """
    Per-frame metrics collector.

    Threading model: written only from the inference thread (single writer), with a
    separate snapshot taken when reading statistics, avoiding lock contention.
    FR-08: when the overlay is disabled, draw_ms is always recorded as None, ensuring
    "zero latency accounted".
    """

    def __init__(
        self,
        enabled: bool = True,
        warmup_frames: int = 1,
        budget_ms: float = 50.0,
        target_p95_ms: float = 30.0,
        first_frame_factor: float = 1.5,
    ):
        self.enabled = enabled
        self.warmup_frames = warmup_frames
        self.budget_ms = budget_ms
        self.target_p95_ms = target_p95_ms
        self.first_frame_factor = first_frame_factor

        self._records: List[FrameRecord] = []
        self._t_start: Optional[float] = None
        self._dropped = 0
        self._max_queue = 0.0

    # ------------------------------------------------------------------
    def start(self) -> None:
        self._t_start = time.perf_counter()

    def record(self, record: FrameRecord) -> None:
        if not self.enabled:
            return
        self._records.append(record)

    def note_render(self, dropped: int, max_queue_depth: float) -> None:
        """Back-fill render-side statistics from the render pipeline."""
        self._dropped = dropped
        self._max_queue = max(self._max_queue, max_queue_depth)

    # ------------------------------------------------------------------
    def summary(self) -> MetricsSummary:
        s = MetricsSummary(
            budget_ms=self.budget_ms,
            target_p95_ms=self.target_p95_ms,
            first_frame_factor=self.first_frame_factor,
        )
        if not self._records:
            return s

        all_parse = [r.parse_ms for r in self._records]
        steady = [r for r in self._records if not r.is_warmup]
        steady_parse = sorted(r.parse_ms for r in steady)

        s.n_total = len(self._records)
        s.n_steady = len(steady)
        s.n_warmup = len(self._records) - len(steady)

        if steady_parse:
            s.p50 = _percentile(steady_parse, 50)
            s.p95 = _percentile(steady_parse, 95)
            s.p99 = _percentile(steady_parse, 99)
            s.max = float(steady_parse[-1])
            s.min = float(steady_parse[0])
            s.mean = float(statistics.fmean(steady_parse))
            s.hard_violations = sum(1 for v in steady_parse if v > self.budget_ms)
        elif all_parse:
            s.max = float(max(all_parse))

        # serving-state first frame = the first non-warm-up frame
        if steady:
            s.first_frame_ms = steady[0].parse_ms

        draw_vals = sorted(r.draw_ms for r in self._records if r.draw_ms is not None)
        if draw_vals:
            s.draw_p50 = _percentile(draw_vals, 50)
            s.draw_p95 = _percentile(draw_vals, 95)
            s.draw_max = float(draw_vals[-1])

        vram_alloc = [r.vram_alloc_mb for r in self._records if r.vram_alloc_mb is not None]
        vram_reserved = [r.vram_reserved_mb for r in self._records if r.vram_reserved_mb is not None]
        s.vram_peak_alloc_mb = max(vram_alloc) if vram_alloc else 0.0
        s.vram_peak_reserved_mb = max(vram_reserved) if vram_reserved else 0.0

        s.dropped_frames = self._dropped
        s.max_queue_depth = self._max_queue
        s.duration_s = (time.perf_counter() - self._t_start) if self._t_start else 0.0
        return s

    # ------------------------------------------------------------------
    def leak_indices(self) -> Dict[str, float]:
        """
        AC-06 helper: a rough characterisation of the growth trend via the ratio of
        "mean of the second half / mean of the first half".
        A return value > 1.05 suggests possible growth and requires manual review (no
        automatic determination, to avoid false positives).
        """
        vram = [r.vram_alloc_mb for r in self._records if r.vram_alloc_mb is not None]
        if len(vram) < 4:
            return {"vram_growth_ratio": 0.0, "samples": float(len(vram))}
        half = len(vram) // 2
        first = statistics.fmean(vram[:half]) or 1e-9
        second = statistics.fmean(vram[half:])
        return {
            "vram_growth_ratio": round(second / first, 4),
            "samples": float(len(vram)),
        }

    def reset(self) -> None:
        self._records.clear()
        self._dropped = 0
        self._max_queue = 0.0
        self._t_start = None

    @property
    def records(self) -> List[FrameRecord]:
        return self._records


def vram_snapshot(torch_module: Optional[Any] = None):
    """
    Read the current VRAM usage, returning (allocated_mb, reserved_mb).
    Returns (None, None) in a non-CUDA environment, which the caller must tolerate.
    """
    torch = torch_module
    if torch is None:
        try:
            import torch as _torch  # type: ignore

            torch = _torch
        except ImportError:
            return None, None
    try:
        if not torch.cuda.is_available():
            return None, None
        return (
            torch.cuda.memory_allocated(0) / (1024 ** 2),
            torch.cuda.memory_reserved(0) / (1024 ** 2),
        )
    except Exception:
        return None, None


__all__ = [
    "FrameRecord",
    "MetricsSummary",
    "MetricsCollector",
    "vram_snapshot",
]
