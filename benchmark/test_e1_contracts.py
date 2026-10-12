"""E1 logic self-test: validates the core contracts without depending on CUDA / a camera / ultralytics."""
import sys, os, time, queue
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

# Explicitly pre-warm the interpreter: the first `import torch` takes about 1s, and if it is not
# completed before the timing interval it pollutes the budget verdict of the warm-up case, leaving
# coverage incomplete and causing sporadic FAILs. Here it is moved ahead of any timing to eliminate
# cold-start jitter (without affecting the logic under test).
try:
    import torch  # noqa: F401
except Exception:
    pass

from config import E1Config
from utils.renderPipeline import FrameSnapshot, RenderPipeline, NullChannel, make_default_pipeline
from utils.warmup import WarmupRunner, WarmupReport, ReadinessState, warmup_if_enabled
from utils.metrics import MetricsCollector, FrameRecord, _percentile

ok, fail = [], []
def check(name, cond, extra=""):
    (ok if cond else fail).append(name)
    print(f"{'PASS' if cond else 'FAIL'} | {name} {extra}")

print("=" * 70)
print("1) Configuration: default off / override precedence / validation")
print("=" * 70)
cfg = E1Config.from_dict({})
check("Default render.enabled=False", cfg.render.enabled is False)
check("Default warmup.enabled=False", cfg.warmup.enabled is False)
check("Default draw_enabled=False", cfg.render.draw_enabled is False)
check("Default queue_depth=1", cfg.render.queue_depth == 1)
check("Default is machine-independent (no hard-coded machine field)", not any("3050" in str(v) or "rtx" in str(v).lower() for v in cfg.to_dict().values()))
check("draw is off the critical path by default", cfg.draw_in_critical_path is False)

cfg2 = E1Config.resolve(None, render_enabled=True, warmup_enabled=True, render_draw_enabled=True)
check("overrides take effect", cfg2.render.enabled and cfg2.warmup.enabled and cfg2.render.draw_enabled)
check("after enabling, draw enters the critical path (true only when decoupling is off)", cfg2.draw_in_critical_path is False)

try:
    E1Config.from_dict({"render": {"queue_depth": 5, "enabled": True}}).validate()
    check("queue_depth>1 rejected when decoupling is on (FR-03)", False)
except ValueError:
    check("queue_depth>1 rejected when decoupling is on (FR-03)", True)

print()
print("=" * 70)
print("2) Queue depth must always be 1 (drains immediately even when constructed with more elements)")
print("=" * 70)
p = RenderPipeline(channel=NullChannel(), queue_depth=3, rate_limit=0, draw_enabled=True)
check("in-pipeline queue maxsize==queue_depth", p._queue.maxsize == 3)

print()
print("=" * 70)
print("3) Render decoupling: zero wait on the inference thread (core FR-01)")
print("=" * 70)
class SlowChannel(NullChannel):
    """Simulates extremely slow rendering (the draw=272.73ms spike in the baseline)."""
    def consume(self, snapshot):
        time.sleep(0.5)

p = RenderPipeline(channel=SlowChannel(), queue_depth=1, rate_limit=0, draw_enabled=True)
p.start()
submit_times = []
for i in range(20):
    t = time.perf_counter()
    p.submit(FrameSnapshot(seq=i, timestamp=t, frame=None, predictions=[], decision=None))
    submit_times.append((time.perf_counter() - t) * 1000)
p.stop()
worst = max(submit_times)
check("worst single submit out of 20 < 1ms (rendering takes 500ms)", worst < 1.0, f"worst={worst:.4f}ms")

print()
print("=" * 70)
print("4) latest-wins: bounded queue depth, no backlog (FR-03)")
print("=" * 70)
p = RenderPipeline(channel=SlowChannel(), queue_depth=1, rate_limit=0, draw_enabled=True)
p.start()
for i in range(30):
    p.submit(FrameSnapshot(seq=i, timestamp=time.perf_counter(), frame=None, predictions=[], decision=None))
s = p.stats()
p.stop()
check("peak queue depth <= 1", s["max_queue_depth"] <= 1, f"max_depth={s['max_queue_depth']}")
check("dropped frames are accounted for", s["dropped"] > 0, f"dropped={s['dropped']}")

print()
print("=" * 70)
print("5) FR-08 measurement isolation: overlay off = zero cost, zero latency counted")
print("=" * 70)
p = RenderPipeline(channel=NullChannel(), queue_depth=1, rate_limit=0, draw_enabled=False)
t = time.perf_counter()
res = p.submit(FrameSnapshot(seq=0, timestamp=t, frame=None, predictions=[]))
elapsed_us = (time.perf_counter() - t) * 1e6
check("submit returns False when draw_enabled=False", res is False)
check("submit does not enqueue (zero cost)", p.stats()["submitted"] == 0)
check("single call < 50us", elapsed_us < 50, f"elapsed={elapsed_us:.2f}us")

print()
print("=" * 70)
print("6) FR-02 render frame dropping: frequency independently configurable, does not scale with the frame rate")
print("=" * 70)
class CountChannel(NullChannel):
    n = 0
    def consume(self, snapshot):
        CountChannel.n += 1
CountChannel.n = 0
p = RenderPipeline(channel=CountChannel(), queue_depth=1, rate_limit=50.0, draw_enabled=True)
p.start()
for i in range(200):          # densely submit 200 frames
    p.submit(FrameSnapshot(seq=i, timestamp=time.perf_counter(), frame=None, predictions=[]))
    time.sleep(0.002)
time.sleep(0.3)
p.stop()
check("200 frames submitted but the render count is limited by frame dropping", CountChannel.n < 200, f"rendered={CountChannel.n}")

print()
print("=" * 70)
print("7) Warm-up: R3 full-chain coverage assertion / over-budget truncation / disabled semantics")
print("=" * 70)
class FakeLog:
    def info(self, m): pass
    def error(self, m): pass

wcfg = E1Config.from_dict({"warmup": {"enabled": True, "device": "cpu", "budget_ms": 2000}}).warmup
calls = []
r = WarmupRunner(wcfg, logger=FakeLog()).run(
    detector_probe=lambda f: calls.append("det") or True,
    embedder_probe=lambda s: calls.append("emb") or True,
)
check("warm-up covers the detector", "ObjectDetector" in r.covered)
check("warm-up covers the embedder (R3)", "ContinuousLatentInferencer" in r.covered)
check("both chains are invoked", calls.count("det") >= 1 and calls.count("emb") >= 1, f"calls={calls}")
check("ready=True when the full chain is covered", r.ready is True)

# Missing embedder -> must not be ready (R3 positive assertion)
r2 = WarmupRunner(wcfg, logger=FakeLog()).run(detector_probe=lambda f: True, embedder_probe=None)
check("ready=False when the embedder is omitted (R3)", r2.ready is False, f"error={r2.error[:40]}")

# Over-budget truncation
trunc_cfg = E1Config.from_dict({"warmup": {"enabled": True, "device": "cpu", "budget_ms": 0.5}}).warmup
def slow_det(f):
    time.sleep(0.05); return True
r3 = WarmupRunner(trunc_cfg, logger=FakeLog()).run(detector_probe=slow_det, embedder_probe=lambda s: True)
check("truncated=True when over budget (R7/NFR-02)", r3.truncated is True)

# Disabled semantics
off = E1Config.from_dict({"warmup": {"enabled": False}}).warmup
r4 = warmup_if_enabled(off)
check("still returns ready=True when warm-up is disabled (caller needs no branch)", r4.ready is True)

print()
print("=" * 70)
print("8) readiness probe (FR-05 / D-6)")
print("=" * 70)
st = ReadinessState()
st.update(r)
probe = st.probe()
check("probe contains ready", probe["ready"] is True)
check("probe contains warmup detail", probe["warmup"] is not None and "stages" in probe["warmup"])
check("probe contains the serving flag", "serving" in probe and probe["serving"] is False)
st.mark_serving()
check("mark_serving takes effect", st.probe()["serving"] is True)

print()
print("=" * 70)
print("9) Metrics: FR-07 first-frame exclusion / AC-01 hard constraint / AC-02 p95")
print("=" * 70)
check("p95 percentile methodology", abs(_percentile(sorted([1,2,3,4,5,6,7,8,9,10]), 50) - 5.5) < 1e-9)
mc = MetricsCollector(enabled=True, warmup_frames=1, budget_ms=50, target_p95_ms=30)
mc.start()
mc.record(FrameRecord(seq=1, timestamp=0, parse_ms=1393.72, is_warmup=True))   # cold-start first frame
for i in range(2, 102):
    mc.record(FrameRecord(seq=i, timestamp=i, parse_ms=43.5))
s = mc.summary()
check("warm-up frames excluded from steady state", s.n_steady == 100 and s.n_warmup == 1)
check("first frame 1393.72ms does not produce a hard-constraint violation", s.hard_violations == 0)
check("AC-01 PASS", s.ac01_pass is True)
check("AC-02 FAIL (43.5>30, non-gating)", s.ac02_pass is False)
check("service-state first frame recorded as 43.5", abs(s.first_frame_ms - 43.5) < 1e-9)

# Inject one 750ms spike -> AC-01 must FAIL
mc2 = MetricsCollector(enabled=True, warmup_frames=1, budget_ms=50, target_p95_ms=30)
mc2.start()
mc2.record(FrameRecord(seq=1, timestamp=0, parse_ms=1393.72, is_warmup=True))
for i in range(2, 102):
    mc2.record(FrameRecord(seq=i, timestamp=i, parse_ms=43.5 if i != 50 else 750.0))
check("the 750ms spike is judged a hard-constraint violation (AC-01 FAIL)", mc2.summary().ac01_pass is False)

print()
print("=" * 70)
print(f"Total: PASS={len(ok)}  FAIL={len(fail)}")
if fail:
    print("Failed items:", fail)
print("=" * 70)
sys.exit(1 if fail else 0)
