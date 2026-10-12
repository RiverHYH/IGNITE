# E1 · 配置项说明（D-2）

> 对应交付物 **D-2（配置项说明：FR-06 开关与默认值）**，以及 FR-02 / FR-03 / FR-08 / NFR-02 的可调参数。

## 1. 核心原则

| 原则 | 说明 | 对应条目 |
|---|---|---|
| **默认全关** | `render.enabled` 与 `warmup.enabled` 默认均为 `False`。不传配置时行为与基线**逐帧一致**，零回归风险 | C-02 / NFR-04 |
| **机型无关** | 所有阈值均为通用可调量，**不存在任何机型白名单或硬编码假设** | C-01 / AC-09 |
| **三级优先级** | 显式入参 `overrides` > `config` 入参 > 环境变量 `IGNITE_*` > 默认值 | NFR-06 |
| **独立开关** | 解耦与预热可分别启停，支持 2×2 对照组（FR-06） | FR-06 |

## 2. 配置加载方式

```python
from config import E1Config

# 方式一：显式字典（推荐用于压测对照）
cfg = E1Config.from_dict({
    "render": {"enabled": True, "draw_enabled": False},
    "warmup": {"enabled": True},
}).validate()

# 方式二：YAML 文件（需 PyYAML）
cfg = E1Config.from_yaml("config.yaml").validate()

# 方式三：环境变量（推荐用于 CI / 免改代码切组）
cfg = E1Config.from_env().validate()

# 方式四：统一入口 + 覆盖
cfg = E1Config.resolve(None, render_enabled=True, warmup_enabled=True).validate()

# 注入应用
from app import IGNITE
app = IGNITE(cfg)
```

## 3. 环境变量对照表

| 环境变量 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `IGNITE_RENDER_ENABLED` | bool | `false` | 渲染解耦总开关（FR-01） |
| `IGNITE_RENDER_DRAW_ENABLED` | bool | `false` | 叠加层开关（FR-08） |
| `IGNITE_RENDER_RATE_LIMIT` | float | `15.0` | 渲染降帧上限（次/秒），<=0 不限（FR-02） |
| `IGNITE_RENDER_QUEUE_DEPTH` | int | `1` | 队列深度上限，**必须为 1**（FR-03） |
| `IGNITE_RENDER_SNAPSHOT` | bool | `true` | 帧快照双缓冲（NFR-05） |
| `IGNITE_RENDER_BACKEND` | str | `thread` | `thread` \| `process` |
| `IGNITE_WARMUP_ENABLED` | bool | `false` | CUDA 预热总开关（FR-04） |
| `IGNITE_WARMUP_BUDGET_MS` | float | `2000` | 预热耗时上限（NFR-02） |
| `IGNITE_WARMUP_DEVICE` | str | `cuda` | `cuda` \| `cpu` |
| `IGNITE_WARMUP_IMGSZ` | int | `640` | dummy 前向边长，**须与部署期一致** |
| `IGNITE_WARMUP_HALF` | bool | `true` | 预热精度，**须与部署期一致** |
| `IGNITE_WARMUP_POOL_MB` | float | `2.5` | 显存池预热目标尺寸（MB），`0` = 跳过（低显存降级） |
| `IGNITE_WARMUP_FULL_CHAIN` | bool | `true` | `false` = 只预热检测器、跳过嵌入器（低显存降级） |
| `IGNITE_METRICS_ENABLED` | bool | `true` | 逐帧度量记录 |
| `IGNITE_METRICS_WARMUP_FRAMES` | int | `1` | 前 N 帧标记为预热帧并剔除（FR-07） |

## 4. `render` 段（渲染解耦）

| 字段 | 默认 | 需求条目 | 说明 |
|---|---|---|---|
| `enabled` | `False` | FR-01 | `True` 时 draw 与 cv2 呈现移入独立渲染线程，推理线程经非阻塞 `submit()` 投递 |
| `draw_enabled` | `False` | FR-08 | `False` 时 draw **完全不执行**，零耗时、零延迟计入。**产品延迟 KPI 必须在此状态下测量** |
| `rate_limit` | `15.0` | FR-02 | 渲染频率上限。默认 15 < 相机 30 FPS，确保 draw 不随推理帧率放大 |
| `queue_depth` | `1` | FR-03 | 固定 1 = latest-wins。`enabled=True` 且该值 >1 时 `validate()` 直接报错 |
| `snapshot` | `True` | NFR-05 | 投递帧快照，写者唯一，避免竞态与撕裂 |
| `backend` | `"thread"` | R1 | `process` 为 GPU/CPU 争用的彻底隔离路径预留 |
| `name` | `"IGNITE"` | — | cv2 窗口名，与基线一致 |

## 5. `warmup` 段（CUDA 预热）

| 字段 | 默认 | 需求条目 | 说明 |
|---|---|---|---|
| `enabled` | `False` | FR-06 | 预热总开关 |
| `budget_ms` | `2000.0` | NFR-02 / R7 | 超预算即截断（`truncated=True`）并放行，避免拖慢用户前台 |
| `device` | `"cuda"` | — | CUDA 不可用时自动降级 CPU 自检，不抛错 |
| `imgsz` | `640` | R3 | dummy 前向边长 |
| `half` | `True` | — | 与基线 `activate(..., half=True)` 对齐 |
| `dummy_prompt` | 见源码 | R3 | 嵌入器 dummy 文本，确保**全链路**被覆盖 |
| `warm_iters` | `3` | — | 重复次数，使内核调优与显存池进入稳态 |
| `cudnn_benchmark` | `True` | — | 触发 cuDNN 算法自动调优 |
| `pool_mb` | `2.5` | NFR-03 | 显存池预热目标尺寸（MB）。**低显存机型降级开关**：置 `0` 完全跳过池预热 |
| `full_chain` | `True` | R3 | 是否执行检测器 + 嵌入器全链路预热。**低显存机型降级开关**：置 `False` 只预热检测器（此时 R3 覆盖度为局部，`ready` 判定相应放宽） |

## 6. `metrics` 段（度量）

| 字段 | 默认 | 需求条目 | 说明 |
|---|---|---|---|
| `enabled` | `True` | — | 逐帧度量记录开关 |
| `warmup_frames` | `1` | FR-07 | 前 N 帧标记 `is_warmup=True`，从稳态统计中剔除 |
| `latency_budget_ms` | `50.0` | NFR-01 | AC-01 硬约束判据 |
| `latency_target_p95_ms` | `30.0` | NFR-01 | AC-02 冲击目标判据（非门控） |
| `first_frame_factor` | `1.5` | NFR-02 | 服务态首帧 <= 稳态 p95 × 该系数 |
| `leak_window_s` | `1800.0` | AC-06 | 长时稳定性观测窗口（30 min） |

## 7. 验收用推荐配置

```python
# AC-01 门控验收（产品延迟 KPI：叠加层必须关闭）
E1Config.from_dict({
    "render": {"enabled": True, "draw_enabled": False, "queue_depth": 1, "rate_limit": 15.0},
    "warmup": {"enabled": True, "device": "cuda", "imgsz": 640, "half": True, "budget_ms": 2000},
    "metrics": {"enabled": True, "warmup_frames": 1},
}).validate()
```

**四个对照组一键复现**（对应需求 §5「解耦开/关 × 叠加层开/关共 4 组」）：

```bash
python benchmark/benchmark_e1.py --frames 1000 --matrix --backend real --out reports/
```

## 8. 约束与校验行为

`validate()` 会拦截以下非法配置（早失败优于线上静默降级）：

| 条件 | 行为 |
|---|---|
| `queue_depth < 1` | 抛 `ValueError` |
| `queue_depth > 1` 且 `enabled=True` | 抛 `ValueError`，提示 FR-03 要求固定为 1 |
| `backend` 非 `thread`/`process` | 抛 `ValueError` |
| `budget_ms <= 0` 或 `imgsz <= 0` | 抛 `ValueError` |
| `warmup_frames < 0` | 抛 `ValueError` |

> 注：`queue_depth > 1` 在**解耦关闭**时放行，仅供离线回放调试；一旦开启解耦即被强制回 1。
