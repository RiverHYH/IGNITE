"""
E1 · Render decoupling + CUDA warm-up — configuration module
============================================================

Corresponding requirement items:
    FR-02  Render rate limiting (visualisation frequency independently configurable,
           by default <= inference frame rate, and not amplified with it)
    FR-03  Frame-drop policy latest-wins (fixed cap on queue depth, default 1)
    FR-06  Regression switches (render decoupling and warm-up can each be disabled
           independently, for A/B and offline-replay comparisons)
    FR-08  Metrics isolation (with the overlay disabled, zero time and zero latency
           are accounted)
    NFR-02 Total warm-up duration <= 2000 ms
    NFR-03 Extra VRAM overhead introduced by decoupling <= 10% of the original budget
           (hard constraint on 4GB VRAM)
    C-01   Target machine class = commercially available consumer RTX series; the
           warm-up and VRAM strategy must be parameterised, with hard-coded machine
           models forbidden

Design notes
------------
1. All switches are **disabled by default** (`enabled=False`). The default behaviour
   is identical to the baseline frame by frame, guaranteeing zero risk to C-02
   (no change to existing CLI/API semantics) and NFR-04 (unchanged accuracy).
   During acceptance they are enabled explicitly via configuration, which yields the
   "decoupling on/off x overlay on/off" comparison groups required by AC-01.
2. Configuration precedence: explicit arguments > environment variables (IGNITE_*) >
   defaults. No training dependency or data dependency is introduced (C-03).
3. No machine-model allow-list is consulted; all thresholds are generic tunables
   (C-01 / AC-09).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict, fields
from typing import Any, Dict


# --------------------------------------------------------------------------
# Environment-variable parsing helpers
# --------------------------------------------------------------------------
_TRUTHY = {"1", "true", "yes", "on", "y", "t"}


def _env(key: str, default: Any, cast=None) -> Any:
    """Read the IGNITE_<key> environment variable, falling back to default when absent."""
    raw = os.environ.get(f"IGNITE_{key.upper()}")
    if raw is None or raw == "":
        return default
    if cast is bool:
        return raw.strip().lower() in _TRUTHY
    if cast is not None:
        try:
            return cast(raw)
        except (TypeError, ValueError):
            return default
    return raw


# --------------------------------------------------------------------------
# Render decoupling configuration
# --------------------------------------------------------------------------
@dataclass
class RenderConfig:
    """
    Render decoupling / rate limiting / frame-drop policy.

    Attributes
    ----------
    enabled : bool
        FR-01 master switch. When False, falls back to the baseline
        "same-thread synchronous draw" path.
    draw_enabled : bool
        FR-08 overlay switch. When False, draw is not executed at all, **zero time
        and zero latency are accounted**. Product-latency KPIs are always measured
        with draw_enabled=False.
    rate_limit : float
        Render rate cap (per second). <=0 means unlimited (follow the streaming frame
        rate). FR-02: default 15.0, below the 30 FPS camera frame rate, avoiding draw
        being amplified with the inference frame rate.
    queue_depth : int
        FR-03 queue depth cap. Fixed at 1, i.e. latest-wins; >1 is for debugging only,
        and the default is strictly 1, with no backlog permitted.
    snapshot : bool
        NFR-05 single writer + double buffering: what is handed to the render thread
        is a shallow copy of the frame snapshot. The render thread only reads it while
        the inference thread keeps reusing the original buffer, avoiding data races
        and tearing.
    coordinate_space : str
        "bgr" | "rgb". The frame is converted uniformly before being passed to the
        render thread, and no colour-space assumption is made inside the render thread.
    backend : str
        "thread" | "process". Default is a thread (under the GIL, both cv2 and numpy
        release it, meeting the P1 decoupling requirement); "process" is reserved for
        R1/R8 mitigation: multiple processes can fully isolate GPU/CPU contention.
    name : str
        cv2 window name, kept consistent with the baseline ("IGNITE").
    """

    enabled: bool = False
    draw_enabled: bool = False
    rate_limit: float = 15.0
    queue_depth: int = 1
    snapshot: bool = True
    coordinate_space: str = "bgr"
    backend: str = "thread"
    name: str = "IGNITE"


# --------------------------------------------------------------------------
# CUDA warm-up configuration
# --------------------------------------------------------------------------
@dataclass
class WarmupConfig:
    """
    CUDA context / cuDNN / VRAM pool / full-chain dummy forward warm-up.

    Attributes
    ----------
    enabled : bool
        FR-06 master warm-up switch.
    budget_ms : float
        NFR-02 upper bound on total warm-up duration (one-off). On timeout the warm-up
        is terminated immediately and execution proceeds, avoiding R7 "warm-up
        monopolising resources and slowing the user's foreground operations".
    device : str
        Target device, "cuda" | "cpu". On cpu the warm-up degrades to an environment
        self-check.
    imgsz : int
        Side length of the dummy forward input, which must match the deployment phase,
        otherwise kernel auto-tuning is invalidated.
    half : bool
        Whether to warm up in FP16. Must match the inference precision used at
        deployment (the baseline CLI passes half=True).
    dummy_prompt : str
        R3 mitigation: dummy text covering the embedder, ensuring the full chain of
        detector + embedder is warmed up, rather than only the detector.
    warm_iters : int
        Number of dummy forward repetitions, used to stabilise kernel auto-tuning and
        expand the VRAM pool to a steady state.
    cudnn_benchmark : bool
        Whether to enable cuDNN benchmark to trigger algorithm auto-tuning.
    log_stages : bool
        FR-05 per-stage duration logging.
    pool_mb : float
        NFR-03 target size of the VRAM pool warm-up (MB). The allocation is released
        immediately after the warm-up phase, used to induce the CUDA caching allocator
        to expand to the deployment steady state. On low-VRAM machines (e.g. 4GB) it
        can be reduced or set to 0 to skip, acting as a VRAM degradation switch.
    full_chain : bool
        R3 mitigation: whether to execute the full-chain dummy forward of detector +
        embedder. On 4GB machines a full chain would exhaust VRAM, so it can be set to
        False to warm up only the detector.
    """

    enabled: bool = False
    budget_ms: float = 2000.0
    device: str = "cuda"
    imgsz: int = 640
    half: bool = True
    dummy_prompt: str = "A flame is located inside the furniture fabric."
    warm_iters: int = 3
    cudnn_benchmark: bool = True
    log_stages: bool = True
    pool_mb: float = 2.5
    full_chain: bool = True


# --------------------------------------------------------------------------
# Observation / metrics configuration
# --------------------------------------------------------------------------
@dataclass
class MetricsConfig:
    """
    FR-05 warm-up observability / FR-07 first-frame exception labelling / stress-test
    protocol.

    Attributes
    ----------
    enabled : bool
        Whether to record per-frame metrics.
    warmup_frames : int
        FR-07 the first N frames are explicitly labelled "warm-up frames" and excluded
        from steady-state statistics.
    latency_budget_ms : float
        NFR-01 hard constraint: any frame <= 50 ms (the first frame excepted).
    latency_target_p95_ms : float
        NFR-01 stretch target: steady-state p95 <= 30 ms (non-gating).
    first_frame_factor : float
        NFR-02 serving-state first frame <= steady-state p95 x this factor (1.5).
    leak_window_s : float
        AC-06 long-duration stability observation window (30 min).
    """

    enabled: bool = True
    warmup_frames: int = 1
    latency_budget_ms: float = 50.0
    latency_target_p95_ms: float = 30.0
    first_frame_factor: float = 1.5
    leak_window_s: float = 1800.0


# --------------------------------------------------------------------------
# Top-level configuration
# --------------------------------------------------------------------------
@dataclass
class E1Config:
    """Top-level E1 configuration aggregate."""

    render: RenderConfig = field(default_factory=RenderConfig)
    warmup: WarmupConfig = field(default_factory=WarmupConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    # ------------------------------------------------------------------
    # Construction and serialisation
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, payload: Dict[str, Any] | None) -> "E1Config":
        """Build from a nested dict; fields not provided retain their defaults."""
        cfg = cls()
        if not payload:
            return cfg
        for section in ("render", "warmup", "metrics"):
            sub = payload.get(section)
            if not isinstance(sub, dict):
                continue
            target = getattr(cfg, section)
            valid = {f.name for f in fields(target)}
            for key, value in sub.items():
                if key in valid:
                    setattr(target, key, value)
        return cfg

    @classmethod
    def from_yaml(cls, path: str) -> "E1Config":
        """
        Load from YAML. A clear message is raised when PyYAML is missing, with no
        silent fallback. If PyYAML is not installed in the project, from_dict can be
        used to pass a dict directly.
        """
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - actionable hint when a dependency is missing
            raise ImportError(
                "Reading YAML configuration requires PyYAML; please run `pip install pyyaml` "
                "first, or use E1Config.from_dict(...) to pass a configuration dict directly."
            ) from exc
        with open(path, "r", encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}
        return cls.from_dict(payload)

    @classmethod
    def from_env(cls) -> "E1Config":
        """
        Build from environment variables, making it easy for CI / stress-test scripts to
        switch comparison groups without changing code.

        Example:
            IGNITE_RENDER_ENABLED=1
            IGNITE_RENDER_DRAW_ENABLED=0
            IGNITE_WARMUP_ENABLED=1
            IGNITE_WARMUP_BUDGET_MS=2000
        """
        cfg = cls()
        cfg.render.enabled = _env("render_enabled", cfg.render.enabled, bool)
        cfg.render.draw_enabled = _env("render_draw_enabled", cfg.render.draw_enabled, bool)
        cfg.render.rate_limit = _env("render_rate_limit", cfg.render.rate_limit, float)
        cfg.render.queue_depth = _env("render_queue_depth", cfg.render.queue_depth, int)
        cfg.render.snapshot = _env("render_snapshot", cfg.render.snapshot, bool)
        cfg.render.backend = _env("render_backend", cfg.render.backend, str)

        cfg.warmup.enabled = _env("warmup_enabled", cfg.warmup.enabled, bool)
        cfg.warmup.budget_ms = _env("warmup_budget_ms", cfg.warmup.budget_ms, float)
        cfg.warmup.device = _env("warmup_device", cfg.warmup.device, str)
        cfg.warmup.imgsz = _env("warmup_imgsz", cfg.warmup.imgsz, int)
        cfg.warmup.half = _env("warmup_half", cfg.warmup.half, bool)
        cfg.warmup.pool_mb = _env("warmup_pool_mb", cfg.warmup.pool_mb, float)
        cfg.warmup.full_chain = _env("warmup_full_chain", cfg.warmup.full_chain, bool)

        cfg.metrics.enabled = _env("metrics_enabled", cfg.metrics.enabled, bool)
        cfg.metrics.warmup_frames = _env("metrics_warmup_frames", cfg.metrics.warmup_frames, int)
        return cfg

    @classmethod
    def resolve(
        cls,
        config: "E1Config | Dict[str, Any] | str | None" = None,
        **overrides: Any,
    ) -> "E1Config":
        """
        Unified resolution entry point, precedence:
        overrides > config argument > environment variables > defaults.

        config accepts an E1Config instance / dict / yaml path / None.
        overrides take the form render_enabled=True, warmup_enabled=True.
        """
        if isinstance(config, E1Config):
            cfg = config
        elif isinstance(config, dict):
            cfg = cls.from_dict(config)
        elif isinstance(config, str):
            cfg = cls.from_yaml(config)
        else:
            cfg = cls.from_env()

        for key, value in overrides.items():
            if "_" not in key:
                continue
            section, attr = key.split("_", 1)
            if section in ("render", "warmup", "metrics") and hasattr(getattr(cfg, section), attr):
                setattr(getattr(cfg, section), attr, value)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        """Convenient for writing into stress-test reports / reproduction notes (D-5)."""
        return asdict(self)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> "E1Config":
        """
        Parameter validity checks, surfacing configuration errors as early as possible.
        Only generic consistency checks are performed, with no machine-model assumptions
        (C-01).
        """
        if self.render.queue_depth < 1:
            raise ValueError("render.queue_depth must be >= 1 (latest-wins default 1).")
        if self.render.queue_depth > 1 and self.render.enabled:
            raise ValueError(
                "FR-03 requires the queue depth cap to be fixed at 1 (no backlog); "
                ">1 is for debugging only, please adjust it after disabling decoupling."
            )
        if self.render.backend not in ("thread", "process"):
            raise ValueError("render.backend supports only 'thread' | 'process'.")
        if self.render.coordinate_space not in ("bgr", "rgb"):
            raise ValueError("render.coordinate_space supports only 'bgr' | 'rgb'.")
        if self.warmup.budget_ms <= 0:
            raise ValueError("warmup.budget_ms must be positive (NFR-02 <= 2000 ms).")
        if self.warmup.imgsz <= 0:
            raise ValueError("warmup.imgsz must be a positive integer.")
        if self.metrics.warmup_frames < 0:
            raise ValueError("metrics.warmup_frames must not be negative.")
        return self

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def draw_in_critical_path(self) -> bool:
        """
        Whether draw still lies on the inference critical path.
        True only when decoupling is disabled and the overlay is enabled — precisely
        the state E1 aims to eliminate.
        """
        return (not self.render.enabled) and self.render.draw_enabled


__all__ = [
    "RenderConfig",
    "WarmupConfig",
    "MetricsConfig",
    "E1Config",
]
