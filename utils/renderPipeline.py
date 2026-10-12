"""
E1 · Render decoupling module renderPipeline
============================================

Corresponding requirement items:
    FR-01  Render decoupling: rendering must be decoupled from the inference loop into
           a separate thread/process; the inference thread must not wait for draw to return
    FR-02  Render rate limiting: the visualisation drawing frequency is independently
           configurable, by default <= inference frame rate, and not amplified with it
    FR-03  Frame-drop policy latest-wins: when rendering blocks, old frames are
           discarded/overwritten, with no backlog queue allowed; the queue depth cap is fixed
    FR-08  Metrics isolation: with the overlay disabled, zero time and zero latency are
           accounted; the interface must retain the openness to "be promoted to a product UI"
    NFR-05 Thread safety: data transfer between rendering and inference must be thread-safe
           (snapshot/double buffering), with no data races or tearing

Design notes
------------
1. **Zero waiting on the inference thread**: `submit()` performs only a single
   non-blocking put and never joins / never blocks. If rendering is slow, frames are
   dropped, in line with FR-03 latest-wins.
2. **Single writer (R2 mitigation)**: once the inference thread has produced a frame
   snapshot, ownership of that snapshot transfers to the render thread; the inference
   thread no longer touches it and the render thread only reads it. Hence no global lock
   is needed to protect the pixel data.
3. **Zero-cost disable (FR-08)**: when `draw_enabled=False`, `submit()` returns directly
   at the first check, building no snapshot, taking no lock and timing nothing — a
   genuine zero time and zero latency accounting.
4. **Interface promotable to a UI (FR-08 / R8)**: the `RenderChannel` abstraction and the
   frame snapshot structure are exposed, so wiring up a product UI / live stream in the
   future only requires replacing the sink — a configuration change rather than a rewrite.
"""

from __future__ import annotations

import abc
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

try:  # cv2 is needed only for actual rendering; its absence allows pure-logic testing
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore


# ==========================================================================
# Frame snapshot: the sole data carrier between the render thread and the
# inference thread
# ==========================================================================
@dataclass
class FrameSnapshot:
    """
    A single renderable frame snapshot.

    Ownership convention (R2: single writer made explicit)
    ------------------------------------------------------
    - the inference thread creates and fills the snapshot, then hands it over via
      RenderChannel;
    - after handover the inference thread **must not** read or write `frame` again;
    - the render thread is the sole reader of `frame`, so reading pixels needs no locking.
    """

    seq: int                      # inference frame sequence number, for metric alignment and deduplication
    timestamp: float              # time of inference production (time.perf_counter)
    frame: Any                    # ndarray, BGR or RGB (see RenderConfig.coordinate_space)
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    decision: Any = None          # final decision tuple, for the overlay or a debug UI
    is_warmup: bool = False       # FR-07: warm-up frame explicitly labelled

    def meta(self) -> Dict[str, Any]:
        """Lightweight meta-information, avoiding carrying whole-frame pixels into logs."""
        return {
            "seq": self.seq,
            "timestamp": round(self.timestamp, 6),
            "n_predictions": len(self.predictions),
            "is_warmup": self.is_warmup,
        }


# ==========================================================================
# Render channel abstraction: the open interface by which FR-08 can be
# "promoted to a product UI"
# ==========================================================================
class RenderChannel(abc.ABC):
    """
    Render channel interface.

    The current iteration only needs a "local window sink"; in the future, if the
    overlay is promoted to a product UI, or frames need to be pushed to RTSP / WebRTC /
    a message channel, a new implementation class need only be added, with no change to
    the inference-side call site — that is the "configuration change rather than a
    rewrite" required by FR-08.
    """

    @abc.abstractmethod
    def consume(self, snapshot: FrameSnapshot) -> None:
        """Consume one frame snapshot within the render thread. The implementer is responsible for the actual drawing and presentation."""

    def on_start(self) -> None:  # pragma: no cover - optional hook
        """Preparation action when the render thread starts."""

    def on_stop(self) -> None:  # pragma: no cover - optional hook
        """Clean-up action when the render thread exits."""


class LocalWindowChannel(RenderChannel):
    """
    Local cv2 window sink — the default debug visualisation channel for this iteration.

    Note: the window handle is created and destroyed only within the render thread, and
    the inference thread never touches the GUI, avoiding random crashes caused by
    cross-thread GUI calls.
    """

    def __init__(self, window_name: str = "IGNITE", size=(960, 540), draw_fn: Optional[Callable] = None):
        self.window_name = window_name
        self.size = size
        self._draw_fn = draw_fn
        self._ready = False

    def on_start(self) -> None:
        if cv2 is None:
            return
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, *self.size)
        self._ready = True

    def consume(self, snapshot: FrameSnapshot) -> None:
        if cv2 is None:
            return
        # Draw within this thread: the cost of draw only affects the render thread and
        # no longer enters the inference critical path
        frame = snapshot.frame
        if self._draw_fn is not None and snapshot.predictions:
            frame = self._draw_fn(frame, snapshot.predictions, in_place=True)
        if self._ready:
            cv2.imshow(self.window_name, frame)

    def on_stop(self) -> None:
        if cv2 is None:
            return
        if self._ready:
            cv2.destroyWindow(self.window_name)
            self._ready = False


class NullChannel(RenderChannel):
    """
    Empty sink — used for baseline comparison and headless stress testing.

    Used for the "decoupling on / overlay off" comparison group: frames are delivered
    as usual but no drawing is performed, thereby demonstrating that the latency gain
    comes from "moving draw off the critical path" rather than from "not drawing at all".
    """

    def consume(self, snapshot: FrameSnapshot) -> None:
        return None


# ==========================================================================
# Render pipeline: dedicated thread + bounded latest-wins queue
# ==========================================================================
class RenderPipeline:
    """
    Dedicated render thread carrying draw and presentation.

    Key guarantees
    --------------
    - `submit()` is non-blocking: when the queue is full the oldest frame is
      overwritten (latest-wins), and the inference thread is never held up by draw.
    - when `draw_enabled=False`, `submit()` returns immediately at zero overhead (FR-08).
    - thread safety: the queue itself is thread-safe; snapshot ownership is unique and
      the pixel data has no shared writes.
    """

    def __init__(
        self,
        channel: RenderChannel,
        queue_depth: int = 1,
        rate_limit: float = 15.0,
        draw_enabled: bool = True,
        snapshot: bool = True,
    ):
        if queue_depth < 1:
            raise ValueError("queue_depth must be >= 1 (FR-03 latest-wins default 1)")

        self.channel = channel
        self.draw_enabled = draw_enabled
        self.snapshot = snapshot
        self.rate_limit = rate_limit
        self._min_interval = (1.0 / rate_limit) if rate_limit and rate_limit > 0 else 0.0

        self._queue: "queue.Queue[FrameSnapshot]" = queue.Queue(maxsize=queue_depth)
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Internal statistics (render side only, for the stress-test report)
        self._lock = threading.Lock()
        self._stats: Dict[str, float] = {
            "submitted": 0,
            "dropped": 0,
            "rendered": 0,
            "draw_ms_last": 0.0,
            "draw_ms_total": 0.0,
            "draw_ms_max": 0.0,
            "max_queue_depth": 0,
        }
        self._last_render_ts = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> "RenderPipeline":
        """Start the render thread (idempotent)."""
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="IGNITE-Render", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 2.0) -> None:
        """
        Stop the render thread and drain the queue.

        Called only at thread lifecycle boundaries (start/exit), and **not on the
        per-frame path**, so it does not violate the constraint that "the inference
        thread must not wait for draw to return".
        """
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        # Drain residual snapshots, releasing the large arrays they hold
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    # ------------------------------------------------------------------
    # Producer side: called by the inference thread
    # ------------------------------------------------------------------
    def submit(self, snapshot: FrameSnapshot) -> bool:
        """
        Submit one frame without blocking.

        Returns
        -------
        bool
            Whether it was successfully enqueued. False means the frame was dropped
            (the render side cannot keep up, latest-wins).

        This method **never blocks**, satisfying FR-01 "the inference thread must not
        wait for draw to return".
        """
        # FR-08: overlay disabled = zero time and zero latency accounted, return at the
        # first check
        if not self.draw_enabled:
            return False

        with self._lock:
            self._stats["submitted"] += 1

        # FR-02: rate limiting. Drop directly when less than the minimum interval has
        # elapsed since the last actual render, avoiding draw being amplified with the
        # inference frame rate
        if self._min_interval > 0.0:
            now = time.perf_counter()
            if (now - self._last_render_ts) < self._min_interval:
                with self._lock:
                    self._stats["dropped"] += 1
                return False
            self._last_render_ts = now

        # FR-03: latest-wins — when the queue is full, evict the oldest frame and insert
        # the newest one
        if self._queue.full():
            try:
                self._queue.get_nowait()
                with self._lock:
                    self._stats["dropped"] += 1
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait(snapshot)
        except queue.Full:
            with self._lock:
                self._stats["dropped"] += 1
            return False

        with self._lock:
            self._stats["max_queue_depth"] = max(
                self._stats["max_queue_depth"], float(self._queue.qsize())
            )
        return True

    # ------------------------------------------------------------------
    # Consumer side: internal to the render thread
    # ------------------------------------------------------------------
    def _loop(self) -> None:
        self.channel.on_start()
        try:
            while self._running:
                try:
                    snapshot = self._queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                started = time.perf_counter()
                try:
                    self.channel.consume(snapshot)
                except Exception as exc:  # a rendering failure must not take down the inference loop
                    _safe_log(f"[renderPipeline] rendering exception, swallowed to protect the inference loop: {exc}")
                elapsed_ms = (time.perf_counter() - started) * 1000.0

                with self._lock:
                    self._stats["rendered"] += 1
                    self._stats["draw_ms_last"] = elapsed_ms
                    self._stats["draw_ms_total"] += elapsed_ms
                    self._stats["draw_ms_max"] = max(self._stats["draw_ms_max"], elapsed_ms)
        finally:
            self.channel.on_stop()

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, float]:
        """Render-side statistics snapshot: per-item draw duration, dropped frames, peak queue depth."""
        with self._lock:
            data = dict(self._stats)
        rendered = data.get("rendered", 0) or 0
        data["draw_ms_mean"] = (data["draw_ms_total"] / rendered) if rendered else 0.0
        data["queue_depth_now"] = float(self._queue.qsize())
        return data

    @property
    def running(self) -> bool:
        return self._running


def _safe_log(message: str) -> None:
    """Fallback logging within the render thread, avoiding a second error when the logger is not ready."""
    try:
        from utils.logger import Logger  # type: ignore

        Logger.debug(message)
    except Exception:
        print(message)


def make_default_pipeline(config, draw_fn: Optional[Callable] = None) -> RenderPipeline:
    """
    Construct the default render pipeline from an E1Config.

    When config disables decoupling, a NullChannel pipeline with draw_enabled=False is
    returned, so the caller can safely call submit() without branching.
    """
    if getattr(config, "enabled", False):
        channel: RenderChannel = LocalWindowChannel(
            window_name=getattr(config, "name", "IGNITE"), draw_fn=draw_fn
        )
    else:
        channel = NullChannel()

    return RenderPipeline(
        channel=channel,
        queue_depth=getattr(config, "queue_depth", 1),
        rate_limit=getattr(config, "rate_limit", 15.0),
        draw_enabled=getattr(config, "draw_enabled", False),
        snapshot=getattr(config, "snapshot", True),
    )


__all__ = [
    "FrameSnapshot",
    "RenderChannel",
    "LocalWindowChannel",
    "NullChannel",
    "RenderPipeline",
    "make_default_pipeline",
]
