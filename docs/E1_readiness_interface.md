# E1 · readiness 就绪接口文档（D-6）

> 对应交付物 **D-6**，以及 **FR-05（预热可观测 + 对外暴露 readiness 信号）**、**C-02（仅新增状态，不改既有 CLI/API 语义）**。

## 1. 设计约束

| 约束 | 落实方式 |
|---|---|
| **C-02：不改既有 CLI/API 语义** | 现有 `activate()` / `_parse()` / 属性访问器签名与行为均未改动；readiness 为**纯新增**接口 |
| **FR-05：预热可观测** | 预热分阶段耗时见 `WarmupReport.stages`；就绪信号见 `ReadinessState` |
| **R3：覆盖不全必须暴露** | 全链路覆盖不全时 `ready=False`，并给出 `error`，不静默放行 |

## 2. 编程接口

```python
from app import IGNITE
from config import E1Config

app = IGNITE(E1Config.from_dict({
    "warmup": {"enabled": True},
    "render": {"enabled": True, "draw_enabled": False},
}))

# ---- 就绪探针 ----
probe = app.probe()
```

### 2.1 `probe()` 返回结构

```json
{
  "ready": true,
  "serving": false,
  "warmup": {
    "ready": true,
    "total_ms": 1843.2,
    "truncated": false,
    "device": "cuda",
    "covered": ["ObjectDetector", "ContinuousLatentInferencer"],
    "error": "",
    "stages": [
      {"stage": "cuda_context",      "elapsed_ms": 620.4, "ok": true, "detail": "device=NVIDIA GeForce RTX ..."},
      {"stage": "vram_pool",         "elapsed_ms": 180.7, "ok": true, "detail": "alloc=12.0MB reserved=64.0MB"},
      {"stage": "cudnn_autotune",    "elapsed_ms": 3.1,   "ok": true, "detail": "benchmark=True"},
      {"stage": "detector_forward",  "elapsed_ms": 742.9, "ok": true, "detail": "imgsz=640 half=True iters=3"},
      {"stage": "embedder_forward",  "elapsed_ms": 292.6, "ok": true, "detail": "prompt='A flame is located insid'"},
      {"stage": "final_sync",        "elapsed_ms": 1.2,   "ok": true, "detail": "synced"}
    ]
  }
}
```

| 字段 | 类型 | 语义 |
|---|---|---|
| `ready` | bool | **就绪信号**。`True` 表示全链路预热完成，可接受推理请求 |
| `serving` | bool | 是否已产出**服务态首帧**。用于区分「已就绪」与「已出帧」（配合 AC-03） |
| `warmup.total_ms` | float | 预热总耗时，**对照 NFR-02 的 2000 ms 上限** |
| `warmup.truncated` | bool | 是否因超预算被截断（R7 降级路径） |
| `warmup.covered` | list | 已被真实 dummy 前向覆盖的链路环节（R3 校验依据） |
| `warmup.stages[]` | list | **FR-05 的分阶段耗时明细**，可直接进日志/监控 |
| `warmup.error` | str | 未就绪原因；`ready=True` 时为空串 |

### 2.2 其他只读接口

| 接口 | 返回 | 用途 |
|---|---|---|
| `app.warmup_report` | `WarmupReport` 或 `None` | 完整预热报告对象；未启用预热时为关闭态报告 |
| `app.metrics_summary()` | `MetricsSummary` | 含 AC-01 / AC-02 / AC-03 判据（见 §4） |
| `app.render_stats()` | `dict` | draw 单项耗时、丢帧数、队列深度峰值 |
| `app.config` | `E1Config` | 当前生效配置，便于复现与排障 |

## 3. 服务化接入建议

`probe()` 是纯函数式读取，可直接挂到任意 Web 框架，无副作用：

```python
from flask import Flask, jsonify
app_web = Flask(__name__)

@app_web.get("/ready")
def ready():
    # 就绪探针：K8s readinessProbe / 负载均衡健康检查
    p = ignite_app.probe()
    return jsonify(p), (200 if p["ready"] else 503)

@app_web.get("/health")
def health():
    # 存活探针：进程活着即 200
    return jsonify({"alive": True}), 200

@app_web.get("/metrics")
def metrics():
    # 性能指标：AC-01/02/03 判据 + 渲染统计
    s = ignite_app.metrics_summary()
    return jsonify({**s.to_dict(), "render": ignite_app.render_stats()}), 200
```

**探针语义区分**（重要，避免误用）：

| 探针 | 语义 | 未就绪时 |
|---|---|---|
| `/health`（liveness） | 进程是否存活 | 不返回 5xx，否则会被重启 |
| `/ready`（readiness） | 全链路预热是否完成 | 返回 **503**，流量暂不导入 |

> 预热耗时上限 2000 ms（NFR-02），因此 `/ready` 由 `503` 转为 `200` 的窗口很短；
> 上游 LB 的重试间隔建议 <= 500 ms，避免就绪后仍有空窗。

## 4. 与验收标准的对应

| 验收项 | 接口依据 |
|---|---|
| **AC-01** 除首帧外无帧 > 50 ms | `metrics_summary().ac01_pass` |
| **AC-02** 稳态 p95 <= 30 ms（非门控） | `metrics_summary().ac02_pass` |
| **AC-03** 冷启动总耗时可记录；服务态首帧 <= 稳态 p95 × 1.5 | `warmup_report.total_ms` + `metrics_summary().ac03_service_first_frame_pass` |
| **AC-04** 显存峰值 | `metrics_summary().vram_peak_reserved_mb` |
| **AC-07** 交付完整性 | `benchmark/benchmark_e1.py --out reports/` 产出的报告套件 |
| **AC-08** 稳态共存（Q4） | `render_stats()` 中的 draw/队列指标；GPU 占用上限待标定 |

## 5. 日志输出样例（FR-05）

预热结束时会输出单行摘要，便于直接 grep：

```
[warmup:READY] total=1843.2ms :: cuda_context=620.4ms | vram_pool=180.7ms | cudnn_autotune=3.1ms | detector_forward=742.9ms | embedder_forward=292.6ms | final_sync=1.2ms
```

运行结束时的汇总（AC-07）：

```
IGNITE E1 运行汇总
样本: total=1001 steady=1000 warmup=1
关键路径时延(ms): p50=43.40 p95=44.12 p99=45.03 max=46.20 mean=43.38
draw(ms): p95=58.20 max=272.73 （仅观测，不计入关键路径）
硬约束(<= 50 ms): 违规帧=0 -> AC-01 PASS
冲击目标(p95 <= 30 ms): -> AC-02 FAIL
服务态首帧=44.05 ms -> AC-03 PASS
```

> 上例中 draw 的 `max=272.73 ms` 会被完整记录但**不计入**关键路径 —— 这正是 FR-08 的度量隔离，
> 也是 E1 相对基线的核心收益：draw 的极端耗时不再污染延迟指标。
