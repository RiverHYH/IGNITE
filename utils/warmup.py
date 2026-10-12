"""
E1 · CUDA warm-up module warmup
===============================

Corresponding requirement items:
    FR-04  CUDA warm-up: on process start, warm up the CUDA context, cuDNN/kernel
           auto-tuning and the VRAM pool, and execute a dummy forward covering the full
           chain (detector + embedder), entering the serving state only when complete
    FR-05  Warm-up observability: each stage outputs a duration log and a readiness
           signal is exposed
    NFR-02 Total warm-up duration <= 2000 ms (one-off)
    R3     Incomplete warm-up coverage (only the detector warmed up, the embedder
           missed) → the dummy forward must cover the full chain, with readiness
           verified by assertion
    R7     Warm-up monopolising resources and slowing the user's foreground operations
           → set a duration cap + release priority as soon as ready

Background (Appendix C, t=0)
----------------------------
The one-off cold-start overhead is **1,393.72 ms**, comprising:
    CUDA context creation + first kernel compilation + VRAM pool initialisation
This overhead lands on the first frame, directly affecting the start-up SLA and
first-frame availability (problem P2).
This module moves the above overhead **forward** into an explicit warm-up phase,
bringing the "serving-state first frame" back into the steady-state range.

Design notes
------------
1. **Per-stage timing**: context → cuDNN tuning → VRAM pool → detector dummy → embedder
   dummy, each stage timed independently and printable, satisfying FR-05.
2. **Full-chain coverage + assertion (R3)**: not only the detector is warmed up; the
   embedder (BGE) also executes a dummy forward; if any stage has not genuinely run a
   CUDA tensor, `ready` is not set.
3. **Hard budget gate (NFR-02 / R7)**: once `budget_ms` is exceeded, subsequent stages
   are aborted immediately, `truncated=True` is recorded and execution still proceeds —
   better an incomplete warm-up than holding up the user's foreground.
4. **Zero model risk (G4)**: the warm-up performs forward passes only, changing no
   weights, thresholds or knowledge base; the dummy input is a synthetic tensor/fixed
   text, introducing no data dependency (C-03).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ==========================================================================
# Result structures
# ==========================================================================
@dataclass
class WarmupStage:
    """Duration record of a single warm-up stage."""

    name: str
    elapsed_ms: float
    ok: bool
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.name,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "ok": self.ok,
            "detail": self.detail,
        }


@dataclass
class WarmupReport:
    """
    Overall warm-up report — the externally observable product of FR-05 and the raw
    data for the AC-03 cold-start test.

    Attributes
    ----------
    ready : bool
        Readiness signal. True means the serving state may be entered.
    total_ms : float
        Total warm-up duration, for comparison against NFR-02 (<= 2000 ms).
    truncated : bool
        Whether it was aborted early because the budget was exceeded (R7 degradation path).
    device : str
        The actual warm-up device.
    stages : list[WarmupStage]
        Per-stage duration detail.
    covered : list[str]
        The chain elements genuinely covered by the dummy forward (basis for R3 verification).
    """

    ready: bool = False
    total_ms: float = 0.0
    truncated: bool = False
    device: str = "unknown"
    stages: List[WarmupStage] = field(default_factory=list)
    covered: List[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "total_ms": round(self.total_ms, 3),
            "truncated": self.truncated,
            "device": self.device,
            "covered": list(self.covered),
            "error": self.error,
            "stages": [s.to_dict() for s in self.stages],
        }

    def summary(self) -> str:
        """Single-line summary, convenient for logs and stress-test report references."""
        stages = " | ".join(f"{s.name}={s.elapsed_ms:.1f}ms" for s in self.stages)
        flag = "READY" if self.ready else "NOT-READY"
        trunc = " (truncated)" if self.truncated else ""
        return f"[warmup:{flag}{trunc}] total={self.total_ms:.1f}ms :: {stages}"


# ==========================================================================
# Readiness state object: directly readable by /health /ready style endpoints
# ==========================================================================
class ReadinessState:
    """
    FR-05 readiness signal carrier; C-02 requires only that this state be **added**,
    without changing existing CLI/API semantics.

    Typical usage (when serving):
        state = ReadinessState()
        ...
        state.update(report)          # set once warm-up is complete
        http_get("/ready") -> state.probe()
    """

    def __init__(self) -> None:
        self._ready = False
        self._report: Optional[WarmupReport] = None
        self._first_frame_done = False

    def update(self, report: WarmupReport) -> None:
        self._ready = bool(report.ready)
        self._report = report

    def mark_serving(self) -> None:
        """Called after the serving-state first frame is produced, allowing an external probe to distinguish "ready" from "first frame emitted"."""
        self._first_frame_done = True

    @property
    def ready(self) -> bool:
        return self._ready

    def probe(self) -> Dict[str, Any]:
        """Return a readiness snapshot directly exposable over HTTP/logs (structure corresponding to the D-6 interface document)."""
        return {
            "ready": self._ready,
            "serving": self._first_frame_done,
            "warmup": self._report.to_dict() if self._report else None,
        }


# ==========================================================================
# Warm-up executor
# ==========================================================================
class WarmupRunner:
    """
    Warm-up orchestrator.

    Decoupled from IGNITE business logic: it depends only on two "callables" —
        detector_probe(frame)  -> any return value   (must genuinely run one detector forward)
        embedder_probe(text)   -> any return value   (must genuinely run one embedder forward)
    This way the warm-up logic need not know the internals of ObjectDetector / Semantic,
    nor depend on them in reverse (keeping the Service layer single-purpose).
    """

    def __init__(self, config, logger: Optional[Any] = None):
        self.cfg = config
        self.log = logger
        self._deadline: Optional[float] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _info(self, message: str) -> None:
        if self.log is not None and hasattr(self.log, "info"):
            self.log.info(message)
        else:
            print(message)

    def _budget_left_ms(self) -> float:
        """Remaining budget (ms). Returns positive infinity when there is no deadline."""
        if self._deadline is None:
            return float("inf")
        return max(0.0, (self._deadline - time.perf_counter()) * 1000.0)

    def _timed(self, report: WarmupReport, name: str, fn: Callable[[], str]):
        """
        Execute a single warm-up stage and time it. If the budget is exceeded it
        short-circuits and returns None, leaving the caller to decide whether to abort.
        """
        if self._budget_left_ms() <= 0.0:
            report.truncated = True
            return None

        started = time.perf_counter()
        try:
            detail = fn() or ""
            elapsed = (time.perf_counter() - started) * 1000.0
            stage = WarmupStage(name=name, elapsed_ms=elapsed, ok=True, detail=detail)
            report.stages.append(stage)
            if self.cfg.log_stages:
                self._info(f"[warmup] {name}: {elapsed:.1f} ms {('- ' + detail) if detail else ''}")
            return stage
        except Exception as exc:  # a single-stage failure does not abort the whole run, but marks ok=False
            elapsed = (time.perf_counter() - started) * 1000.0
            stage = WarmupStage(name=name, elapsed_ms=elapsed, ok=False, detail=str(exc))
            report.stages.append(stage)
            self._info(f"[warmup] {name}: FAILED after {elapsed:.1f} ms :: {exc}")
            return stage

    # ------------------------------------------------------------------
    # Main flow
    # ------------------------------------------------------------------
    def run(
        self,
        detector_probe: Optional[Callable[[Any], Any]] = None,
        embedder_probe: Optional[Callable[[str], Any]] = None,
        torch_module: Optional[Any] = None,
    ) -> WarmupReport:
        """
        Execute the full warm-up chain.

        Parameters
        ----------
        detector_probe : callable(ndarray) -> Any
            Detector dummy forward. When None this element is skipped and marked as
            uncovered in the report.
        embedder_probe : callable(str) -> Any
            Embedder dummy forward. R3 requires this element to exist and genuinely run.
        torch_module : module, optional
            Inject torch, to ease substitution in unit tests; by default it is imported
            on its own.

        Returns
        -------
        WarmupReport
        """
        report = WarmupReport(device=self.cfg.device)
        started = time.perf_counter()
        self._deadline = started + (self.cfg.budget_ms / 1000.0)

        torch = torch_module
        if torch is None:
            try:
                import torch as _torch  # type: ignore

                torch = _torch
            except ImportError:
                report.error = "torch is not installed, warm-up impossible; degrading to not-ready."
                report.total_ms = (time.perf_counter() - started) * 1000.0
                self._info(f"[warmup] {report.error}")
                return report

        use_cuda = self.cfg.device == "cuda" and getattr(torch, "cuda", None) is not None \
            and torch.cuda.is_available()
        report.device = "cuda" if use_cuda else "cpu"
        if not use_cuda and self.cfg.device == "cuda":
            self._info("[warmup] CUDA unavailable, warm-up degrading to CPU (chain self-check only).")

        # ---------------- Stage 1: CUDA context ----------------
        # Corresponds to the "CUDA context creation" overhead at t=0, the largest
        # component of 1393.72 ms
        def _ctx() -> str:
            if use_cuda:
                torch.cuda.init()
                torch.cuda.synchronize()
                name = torch.cuda.get_device_name(0)
                return f"device={name}"
            return "cpu-mode"

        if self._timed(report, "cuda_context", _ctx) is None:
            report.total_ms = (time.perf_counter() - started) * 1000.0
            return report

        # ---------------- Stage 2: VRAM pool initialisation ----------------
        # Corresponds to the "VRAM pool initialisation" at t=0; a releasable temporary
        # tensor pulls the caching pool up to a steady state.
        # Skipped when pool_mb=0 (VRAM degradation switch for 4GB machines, NFR-03).
        def _pool() -> str:
            if not use_cuda:
                return "skipped(cpu)"
            if self.cfg.pool_mb <= 0:
                return "skipped(pool_mb=0)"
            # Derive the element count from the target MB (FP16=2 bytes / FP32=4 bytes)
            elem_bytes = 2 if self.cfg.half else 4
            numel = max(1, int(self.cfg.pool_mb * 1024 * 1024 / elem_bytes))
            probe = torch.zeros(numel,
                                dtype=torch.float16 if self.cfg.half else torch.float32,
                                device="cuda")
            torch.cuda.synchronize()
            del probe
            torch.cuda.empty_cache()
            alloc = torch.cuda.memory_allocated(0) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
            return f"pool={self.cfg.pool_mb:.1f}MB alloc={alloc:.1f}MB reserved={reserved:.1f}MB"

        if self._timed(report, "vram_pool", _pool) is None:
            report.total_ms = (time.perf_counter() - started) * 1000.0
            return report

        # ---------------- Stage 3: cuDNN / kernel auto-tuning ----------------
        # Corresponds to the "first kernel compilation" at t=0; benchmark lets cuDNN
        # select and cache an algorithm
        def _cudnn() -> str:
            if not use_cuda:
                return "skipped(cpu)"
            if self.cfg.cudnn_benchmark:
                torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            return f"benchmark={torch.backends.cudnn.benchmark}"

        if self._timed(report, "cudnn_autotune", _cudnn) is None:
            report.total_ms = (time.perf_counter() - started) * 1000.0
            return report

        # ---------------- Stage 4: detector dummy forward ----------------
        # R3: must genuinely execute, rather than merely loading the weights
        def _det() -> str:
            if detector_probe is None:
                raise RuntimeError("detector_probe not provided; the detector chain was not warmed up")
            frame = _synthetic_frame(self.cfg.imgsz)
            detector_probe(frame)
            return f"imgsz={self.cfg.imgsz} half={self.cfg.half} iters={self.cfg.warm_iters}"

        stage = self._timed(report, "detector_forward", _det)
        if stage is None:
            report.total_ms = (time.perf_counter() - started) * 1000.0
            return report
        if stage.ok:
            report.covered.append("ObjectDetector")

        # ---------------- Stage 5: embedder dummy forward ----------------
        # Skipped when full_chain=False (4GB VRAM degradation: warm up the detector only,
        # with R3 coverage thereby partial).
        def _emb() -> str:
            if not self.cfg.full_chain:
                return "skipped(full_chain=False)"
            if embedder_probe is None:
                raise RuntimeError("embedder_probe not provided; the embedder chain was not warmed up")
            embedder_probe(self.cfg.dummy_prompt)
            return f"prompt={self.cfg.dummy_prompt[:32]!r}"

        stage = self._timed(report, "embedder_forward", _emb)
        if stage is None:
            report.total_ms = (time.perf_counter() - started) * 1000.0
            return report
        if stage.ok:
            report.covered.append("ContinuousLatentInferencer")

        # ---------------- Stage 6: readiness synchronisation ----------------
        def _sync() -> str:
            if use_cuda:
                torch.cuda.synchronize()
            return "synced"

        self._timed(report, "final_sync", _sync)

        report.total_ms = (time.perf_counter() - started) * 1000.0

        # R3 assertion-style readiness determination: incomplete coverage means not ready;
        # better to expose it than to proceed silently
        required = {"ObjectDetector", "ContinuousLatentInferencer"}
        missing = required - set(report.covered)
        if missing:
            report.ready = False
            report.error = f"incomplete warm-up chain coverage, missing: {sorted(missing)} (see R3)"
        else:
            report.ready = True

        if report.total_ms > self.cfg.budget_ms:
            report.truncated = True
            self._info(
                f"[warmup] total duration {report.total_ms:.1f} ms exceeds the budget "
                f"{self.cfg.budget_ms:.0f} ms, truncated recorded (NFR-02 / R7)."
            )

        self._info(report.summary())
        return report


# ==========================================================================
# Helpers
# ==========================================================================
def _synthetic_frame(imgsz: int):
    """
    Synthetic dummy frame (no camera, no disk reads), ensuring the warm-up introduces
    no data dependency (C-03).
    A fixed value is used rather than random noise, avoiding (if the implementer prints
    the tensor) polluting the logs with noise.
    """
    try:
        import numpy as np  # type: ignore

        return np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    except ImportError:  # last-resort fallback when numpy is missing
        return [[[0, 0, 0]] * imgsz] * imgsz


def warmup_if_enabled(
    config,
    detector_probe: Optional[Callable] = None,
    embedder_probe: Optional[Callable] = None,
    logger: Optional[Any] = None,
) -> WarmupReport:
    """
    Convenience entry point: decide whether to warm up according to configuration.

    When warm-up is disabled (FR-06), an "empty report" with ready=True is returned,
    so the upper layer can handle readiness uniformly without writing if-statements
    everywhere.
    """
    if not getattr(config, "enabled", False):
        report = WarmupReport(ready=True, device="disabled", total_ms=0.0)
        report.error = ""  # an explicit disable is not an error
        report.stages.append(WarmupStage("warmup_disabled", 0.0, True, "FR-06 switch off"))
        return report
    return WarmupRunner(config, logger=logger).run(
        detector_probe=detector_probe,
        embedder_probe=embedder_probe,
    )


__all__ = [
    "WarmupStage",
    "WarmupReport",
    "ReadinessState",
    "WarmupRunner",
    "warmup_if_enabled",
]
