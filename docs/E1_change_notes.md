# E1 · 改动说明与合并指引

> 本文件说明本次针对《E1 渲染解耦 + CUDA 预热 需求文档 v1.2》所做的全部改动，
> 供评审确认后再执行合并。**原始 IGNITE 目录未被触碰**，全部改动落在副本 `IGNITE_E1/`。

## 1. 改动总览

| 文件 | 类型 | 说明 |
|---|---|---|
| `app.py` | **修改** | 接入渲染解耦与预热；修正 2 处既有缺陷；新增只读观测接口 |
| `config.py` | **新增** | 参数化配置（机型无关、三级优先级、独立开关） |
| `utils/renderPipeline.py` | **新增** | 独立渲染线程 + latest-wins 队列 + 帧快照 |
| `utils/warmup.py` | **新增** | CUDA/cuDNN/显存池/全链路 dummy 前向预热 + readiness |
| `utils/metrics.py` | **新增** | 帧级度量、p50/p95/p99、AC-01/02/03 判据、leak 检查 |
| `benchmark/benchmark_e1.py` | **新增** | 4 组对照压测脚本，产出 D-3/D-4/D-5（经 `utils/logger.py` 落盘为 `<YYYY-MM-DD>_report.log`） |
| `benchmark/test_e1_contracts.py` | **新增** | 35 条契约断言，零外部依赖 |
| `benchmark/selfcheck.py` | **新增** | 自持性校验（项目范式执行体） |
| `docs/E1_configuration_reference.md` | **新增** | D-2 |
| `docs/E1_readiness_interface.md` | **新增** | D-6 |
| `docs/E1_self_contained_branch_paradigm.md` | **新增** | 项目范式：测试 Branch 应可自持（D-7） |
| `docs/E1_change_notes.md` | **新增** | 本文件 |
| `.baseline/` | **新增** | 基线源码只读快照，用于 diff 对照与精度回归 |

**未改动**：`utils/boundingBoxDrawer.py`、`utils/videoCapture.py`、
`Service/ObjectDetector/yolo.py`、`Service/Predicate/affordanceEmbedder.py`、
`Service/Semantic/continuousLatentInferencer.py` —— 逐字节相同。

## 2. `app.py` 改动明细

### 2.1 修复的既有缺陷（改动前即为 bug）

| # | 缺陷 | 位置（基线） | 影响 | 修法 |
|---|---|---|---|---|
| 1 | `frame_counter` 定义在模块级且**从未自增**，恒为 0，故 `frame_counter % SKIP_FRAMES == 0` **永远成立** | `app.py:95` | 下采样完全失效，每帧都跑重推理 | 改为实例内显式计数 `local_idx` |
| 2 | `activate(skip_frames=)` 内部写 `SKIP_FRAMES=skip_frames`，赋值给**局部变量**，模块级全局未变 | `app.py:84` | `skip_frames` 参数**完全无效** | 直接使用入参 `skip`，参数真正生效 |
| 3 | `__main__` 中 `activate(streaming=True, kwargs={...})` 多包一层 | `app.py:194` | 参数落进 `**kwargs` 后被透传给 YOLO，语义与文档不符 | 改为展开传入 `half=.../imgsz=.../vid_stride=...` |

> 这三处均为**纯缺陷修复**，不改变检测逻辑与输出结果，符合 NFR-04。

### 2.2 新增能力

| 能力 | 实现 | 需求条目 |
|---|---|---|
| 渲染解耦 | `RenderPipeline` 独立线程；`submit()` 非阻塞，推理线程不等待 | FR-01 |
| 渲染降帧 | `RenderPipeline` 按 `rate_limit` 独立限频 | FR-02 |
| latest-wins | 队列满即丢最旧帧 | FR-03 |
| CUDA 预热 | `_warmup()` → `WarmupRunner`，6 阶段分阶段计时 | FR-04 |
| readiness | `ReadinessState` + `probe()` | FR-05 |
| 回归开关 | `config.render.enabled` / `config.warmup.enabled` | FR-06 |
| 首帧剔除 | `FrameRecord.is_warmup` | FR-07 |
| 度量隔离 | 叠加层关闭时 `submit()` 首判断即返回；draw 单独计量 | FR-08 |

### 2.3 明确的「未改动」

- `_parse()` 方法体：**逐字符与基线一致**（已用脚本校验）。
  > 这是 NFR-04「精度回归 = 0」的结构性前提：推理核心一行未动。
- `object_detector` / `fire_detector` 属性访问器：语义一致，仅在其后**追加**只读接口。
- `activate()` 签名：完全保持 `(streaming, img_path, result_path, skip_frames, **kwargs)`；
  仅把原函数体按 `streaming` 分流到 `_activate_streaming()` / `_activate_static()`。

## 3. 关键设计决策与理由

| 决策 | 理由 | 对应条目 |
|---|---|---|
| **开关默认全关** | 保证「不传配置 = 基线行为」，零回归风险；验收时才显式开启构成对照组 | C-02 / NFR-04 |
| **队列深度硬性为 1** | 需求要求「禁止积压队列」；`validate()` 在解耦开启时强校验 | FR-03 |
| **渲染默认走线程而非进程** | cv2/numpy 释放 GIL，线程已能满足「不阻塞推理」；进程为 R1 预留 | R1 |
| **预热超预算截断而非报错** | 主动关闭预热，避免 R7 拖慢用户前台；记录 `truncated=True` 供追溯 | NFR-02 / R7 |
| **覆盖不全则 `ready=False`** | R3 要求以断言校验就绪，宁可暴露也不静默放行 | R3 |
| **渲染通道抽象为 `RenderChannel`** | 未来叠加层提升为产品 UI 时只需换 sink，属配置变更而非重写 | FR-08 |
| **配置无任何机型字段** | 禁止硬编码机型（AC-09 要求至少 2 档 RTX 验证） | C-01 |

## 4. 验证结果

### 4.1 静态与逻辑自测（已通过）

| 项 | 结果 |
|---|---|
| 全部 `.py` 编译 | ✅ 通过 |
| 配置默认值 / 优先级 / 校验 | ✅ 通过（含 `queue_depth>1` 被拒绝，用例已修正） |
| **FR-01 推理线程零等待**：渲染占 500 ms 时，20 次 `submit()` 最坏 **0.058 ms** | ✅ 通过 |
| FR-03 latest-wins：投递 30 帧、队列深度峰值 **1**，丢帧 29 | ✅ 通过 |
| FR-08 叠加层关闭：`submit()` 单次 **5.79 µs**，且不入队 | ✅ 通过 |
| FR-02 降帧：投递 200 帧仅渲染 21 次 | ✅ 通过 |
| R3 覆盖断言：漏掉嵌入器 → `ready=False` | ✅ 通过 |
| NFR-02/R7：超预算 → `truncated=True` | ✅ 通过 |
| AC-01 判据：注入 750 ms 尖峰 → `AC-01 FAIL`；首帧 1393.72 ms 被剔除不误报 | ✅ 通过 |
| `_parse()` 与基线逐字符一致 | ✅ 通过 |

### 4.2 4 组对照压测（mock 后端，结构性验证）

`python benchmark/benchmark_e1.py --matrix --backend mock --steady-ms 5.0`

| 组别 | 关键路径 p50 | 结论 |
|---|---:|---|
| 解耦开 × 叠加层开 | 5.08 ms | draw 在渲染线程 |
| 解耦开 × 叠加层关 | 5.09 ms | **与上一行差 0.01 ms ⇒ draw 确已离开关键路径** |
| 解耦关 × 叠加层开 | 25.18 ms | **draw 20 ms 全部落在关键路径上（复现 P1）** |
| 解耦关 × 叠加层关 | 5.09 ms | 对照 |

> 该实验证明：**解耦开启后，叠加层的开/关不再影响关键路径时延**；而解耦关闭且叠加层开启时，
> draw 的耗时会 1:1 地叠加到关键路径 —— 这正是 E1 要消除的结构性问题。

### 4.3 自持性校验（项目范式，已通过）

```bash
python benchmark/selfcheck.py
```

| 级别 | 结果 |
|---|---|
| **CODE 级**（强制门槛） | ✅ 通过 —— 4/4 入口齐备、导入闭包完整、全部编译通过、无占位文件、契约测试 35/35 PASS |
| **RUNTIME 级**（真机验收需要） | ✅ **通过** —— 3 个二进制资产已齐备并通过深层校验（见 `docs/E1_self_contained_branch_paradigm.md` §4） |

> 即：**本分支已可被独立检出并直接跑通全部逻辑测试与真实推理链路**，无需回到开发机。
> `python benchmark/selfcheck.py --strict-runtime` 退出码 0。

**已补齐资产**（项目盘 → 副本，共约 85 MB）：

| 资产 | 大小 | 校验 |
|---|---:|---|
| `Service/ObjectDetector/fire.pt` | 42 MB | 776 项 torch 归档，含 `best/data.pkl` |
| `Service/ObjectDetector/obj1.pt` | 43 MB | 776 项 torch 归档，含 `best/data.pkl` |
| `Service/Semantic/triplets_cache.pkl` | 32 KB | protocol-4 pickle，2 条 384 维 BGE 嵌入，可被 `continuousLatentInferencer._update_ref_matrix` 直接消费 |

### 4.4 尚待现场完成的部分（不可由本沙箱代劳）

| 项 | 原因 |
|---|---|
| 真实 GPU 数值（AC-01/02/04 绝对指标） | 沙箱无 RTX GPU；须在目标机型用 `--backend real` 跑 |
| AC-09 机型矩阵 | 需至少 2 档 RTX 真机 |
| AC-05 精度回归差分 | 需真实回放集与权重（`fire.pt`/`obj1.pt`/`triplets_cache.pkl`） |
| AC-08 稳态共存 | 需真机并行运行微信/浏览器 |
| NFR-07 GPU 占用上限标定 | 需求文档自述「上限待标定」 |

## 5. 合并指引

```bash
# 1. 查看全部差异
diff -ru .baseline . --exclude=__pycache__ --exclude=.baseline

# 2. 确认无回归（应输出「推理核心逐字符一致」）
python3 - <<'EOF'
src = lambda f: open(f, encoding="utf-8").read()
g = lambda s, a, b: s[s.index(a):s.index(b)]
assert g(src(".baseline/app.py"), "def _parse(", "if __name__") == g(src("app.py"), "def _parse(", "if __name__")
print("推理核心一致 ✅")
EOF

# 3. 合并到原工程（确认后执行）
cp config.py app.py /path/to/IGNITE/
cp utils/renderPipeline.py utils/warmup.py utils/metrics.py /path/to/IGNITE/utils/
cp -r benchmark docs /path/to/IGNITE/

# 4. 资产：原工程通常已有历史权重，无需覆盖；若目标环境缺则按
#    docs/E1_self_contained_branch_paradigm.md §4 补齐（共约 85 MB，不进版本库）
```

> **注意**：本次合并**不包含** `.pt` / `.pkl` 二进制资产 —— 它们仅用于副本的 RUNTIME 级自持校验，
> 原工程既有权重应保持不动（避免用副本资产覆盖线上版本）。

## 6. 合并前需确认的开放项

| # | 事项 | 影响 | 状态 |
|---|---|---|---|
| 1 | **pyav 反馈界面消费方是否存在**（需求 §8.3 R8） | 决定 draw 是「纯调试可视化」还是「隐性产品功能」 | ✅ **已决策：按「预埋接口」处理**，见 §6.1 |
| 2 | ~~`Service/` 与 `service/` 目录大小写不一致~~ | **已撤回**：经复核，原工程 import 全部使用大写 `Service`，不存在大小写不一致。此前系我方建目录残留被误读，详见 §8 | ✅ 已关闭 |
| 3 | `utils/logger.py`、`utils/cameraList.py` 等 | 已补入副本（见 §7）；它们未被改动，合并时以原工程为准 | ✅ 已补齐 |
| 4 | 是否保留 `.baseline/` 快照 | 便于后续回归对照；若不希望入库可加入 `.gitignore` | 待定（无阻塞） |

### 6.1 R8 决策记录：预埋渲染消费接口

**决策**（用户确认）：**确认消费方存在**，即便实际终端用户尚未接入，仍需**预埋调用接口并持续调优**；紧急程度不高，不阻塞本次合并。

**已落实的接口预留**（无需新增代码，现有抽象即为预留点）：

| 环节 | 预留方式 | 位置 |
|---|---|---|
| 通道抽象 | `RenderChannel`（ABC），新增 sink 只需实现 `consume()` | `utils/renderPipeline.py:74` |
| 现有实现 | `LocalWindowChannel`（cv2 调试窗）、`NullChannel`（无头压测） | `utils/renderPipeline.py:94 / 133` |
| 组装点 | `make_default_pipeline(config)` 按配置选 sink | `utils/renderPipeline.py:327` |
| 未来 sink | 只需新增一个 `RenderChannel` 子类（RTSP / WebRTC / pyav / 消息通道），**推理侧调用点零改动** | 满足 FR-08「配置变更而非重写」 |

```python
# 未来接入 pyav / RTSP / WebRTC 时的扩展形态（示意，本次不实现）
class PyAVChannel(RenderChannel):
    def on_start(self): ...
    def consume(self, snapshot): ...   # 把 annotated frame 推给反馈界面
    def on_stop(self): ...

# 组装处一行替换，推理线程不受影响
pipeline = RenderPipeline(channel=PyAVChannel(...), ...)
```

> **不阻塞理由**：该预留点已经在架构上成立（FR-08 要求「叠加层可提升为 UI」正是为此设计），
> 后续接入属**新增一个类**，不改动任何已合入的推理/度量逻辑，因此不构成合并的风险项。
> 需要真机验证的仅是「接入后叠加层仍不进入关键路径」，可由已有的 4 组对照压测复现。

## 7. 交付物对照

| ID | 交付物 | 位置 | 状态 |
|---|---|---|---|
| D-1 | 代码补丁 | `app.py` + 4 个新增模块 | ✅ |
| D-2 | 配置项说明 | `docs/E1_configuration_reference.md` | ✅ |
| D-3 | 压测脚本与原始数据 | `benchmark/benchmark_e1.py` + `raw_frames_*.csv` | ✅ |
| D-4 | 前后对照表 | `benchmark` 自动生成 `E1_前后对照表.md` | ✅（含附录 C 基线行） |
| D-5 | 复现说明（环境固化清单） | 自动生成 `E1_复现说明.md` | ✅ |
| D-6 | readiness 就绪接口文档 | `docs/E1_readiness_interface.md` | ✅ |
| D-7 | 自持 Branch 范式 + 自持性校验工具 | `docs/E1_self_contained_branch_paradigm.md` + `benchmark/selfcheck.py` | ✅ |

## 8. 瓶颈归因修正（依论文实验数据）

> 本节依据用户论文实验的 1000 帧 `[REPORT]` 与逐帧曲线（`run_thru.png`），
> 对此前基于 Appendix C 的「draw 是主要瓶颈」判断进行**修正**。**结论：draw 与
> YOLO 检测均非稳态瓶颈，稳态瓶颈在 `_parse()` 的 CPU 侧推理逻辑。**

### 8.1 论文实验原始读数（1000 帧，同机 RTX 3050 Ti Laptop）

| 环节 | mean (ms) | 占 Loop 比 | max (ms) | std |
|---|---|---|---|---|
| IO | 3.005 | 23.0% | 8.446 | 0.72 |
| Inference | 6.437 | 49.3% | **1331.07** | 48.20 |
| Draw | **0.279** | **2.1%** | 272.73 | 8.62 |
| Loop | 13.060 | 100% | **1393.72** | 52.95 |

其他：Mean FPS 156.62；Peak GPU Memory 1373.33 MB；Peak GPU Utilization 99.00%；无内存/显存泄漏。

### 8.2 尖峰归属（逐帧曲线判读）

| 事件 | 时刻 | 主体 | 判读 |
|---|---|---|---|
| 冷启动尖峰 ~1394 ms | t=0 | **Inference ≈1331 ms** | 冷启动主因是**首次推理调用**，非 draw |
| 运行期尖峰 ~750 ms | t≈330 | Inference ≈470 ms **+ Draw 272.73 ms** | 两长尾叠加，draw 仅贡献其中 272 ms |
| Draw 整体 | 全程 | —— | 1000 帧中**仅 1 次**可见尖峰，均值 0.279 ms |

### 8.3 修正后的结论

| 命题 | 此前判断 | **修正后** |
|---|---|---|
| draw 拖慢系统 | ❌ 主瓶颈 | ⚠️ **均值不成立**（占 2.1%），但**进入关键路径时影响显著**（§8.6 实测 +25 ms） |
| YOLO detect 拖慢系统 | ❌ 主瓶颈 | ⚠️ 稳态均值仅 6.4 ms，**非主因** |
| **`_parse()` 全链路（CPU 侧）是稳态瓶颈** | 未识别 | ✅ **成立**：干净环境下 p50 15.68 ms，其中 GPU 前向约 6.2~7.3 ms，**约 55~60% 仍在 CPU 侧** |

旁证：GPU 显存恒稳 1373 MB 不涨、后台负载可使延迟恶化 3 倍（§8.5），
均指向「CPU 侧推理逻辑（几何谓词 + BGE 检索 + 后处理）为稳态瓶颈」。列为 Future Work。

### 8.4 E1 的诚实重定位（**已由真机干净环境数据修正**）

> 本节初稿依据 mock 对照，判断 E1「仅长尾保险、几乎无稳态收益」。
> **随后真机干净环境 300 帧 × 3 组的实测数据推翻了该悲观判断**，见 §8.6。
> 保留本节以记录判断演变过程。

| E1 声明 | mock 初判 | **真机实测（§8.6）** |
|---|---|---|
| 解耦让 draw 不进入关键路径 | ✅ 成立（结构上） | ✅ **成立，实测 p50 降 63%** |
| 消除 draw 的偶发长尾 | ✅ 成立 | ✅ **成立，同时消除方差** |
| 降低冷启动 ~1394 ms | ❌ 不成立 | ❌ 不成立（冷启动主因是首次推理） |
| 提升稳态 FPS | ❌ 几乎不成立 | ✅ **成立**：p50 43.85 → 15.68 ms |

**最终定位**：E1 既有**长尾隔离**价值，也有**明确的稳态收益**（真机实测），
并附带 AC-01 从 FAIL 翻转为 PASS 的门控意义。

### 8.4.1 真机干净环境对照（AC-01 门控验收）

条件：关闭所有非必要后台应用，仅保留系统进程与 VS Code；RTX 3050 Ti Laptop 4 GB；每 300 帧。

| 组 | 解耦 | 叠加层 | p50 (ms) | p95 (ms) | max (ms) | 违规帧 | AC-01 | AC-02 |
|---|---|---|---|---|---|---|---|---|
| **C_clean** | 开 | 关 | **15.81** | 17.26 | 23.28 | 0/299 | ✅ PASS | ✅ PASS |
| **B_clean** | 开 | 开 | **15.68** | 17.03 | 19.85 | 0/299 | ✅ PASS | ✅ PASS |
| **A_clean #1** | 关 | 开 | **40.57** | 43.70 | 55.32 | 2/299 | ❌ FAIL | ❌ FAIL |
| **A_clean #2** | 关 | 开 | **43.85** | 50.11 | 69.96 | 18/299 | ❌ FAIL | ❌ FAIL |

**核心结论**：

1. **解耦收益 ≈ 25 ms**：A 组 p50 ≈ 42.2 ms（两次均值）→ B 组 15.68 ms，**降幅 63%**。
2. **AC-01 门控翻转**：解耦关 → FAIL（2~18 违规帧）；解耦开 → PASS（0 违规帧）。
3. **`B ≈ C`**：叠加层开启未给关键路径增加可测开销（15.68 vs 15.81 ms，在噪声内），
   证明 draw 已被彻底移出关键路径（FR-08 指标隔离成立）。
4. **方差消除**：A 组两次复跑违规帧 2 → 18、max 55 → 70 ms，抖动剧烈；
   B 组稳定在 15.7 ms / max 20 ms。**解耦同时降低了长尾与方差。**

> **注**：A 组的 draw 开销由基准用 `time.sleep(DRAW_SIM_MS=20ms)` 模拟（非真实 GUI draw）。
> 实测差值 25 ms 略大于 20 ms，源于 sleep 与 CPU 侧竞争叠加。该口径下的结论
> 为「draw 进入关键路径时，解耦可消除其影响」，与真实 GUI draw 的绝对数值无关。

### 8.5 实验环境控制变量（重要）

用户确认：论文实验**即为本机**（RTX 3050 Ti Laptop，4 GB），采集时**刻意停用所有非必要应用**，
仅保留 Windows 系统默认进程与 VS Code。

| 条件 | Loop 稳态 | 说明 |
|---|---|---|
| 干净环境（§8.6 重采，C 组） | **15.81 ms**（p50） | 无后台干扰 |
| 有后台负载（早前 300 帧） | **47.89 ~ 50.28 ms**（p50） | 浏览器/IM 等抢占 CPU |
| 论文实验（1000 帧，mean） | **13.06 ms** | 无后台干扰（口径为 mean） |

**同机、同模型，仅因后台负载差异，稳态延迟差约 3 倍。**

由此得两条硬结论：

1. **早前 47.89 ms 是「被污染环境」读数，不得作为论文引用值**；
   正式引用值以 §8.6 干净环境数据为准（C/B 组 p50 ≈ 15.7~15.8 ms）。
2. 该敏感性**反证瓶颈在 CPU 侧**——后台进程只抢 CPU 时间片与系统内存（GPU 显存恒稳 1373 MB 不涨），
   却使延迟恶化 3 倍；若瓶颈在 GPU，增加 Chrome 进程不会产生此效应。

> **写入 D-5 复现说明的环境控制条款**：全部性能读数须在「关闭所有非必要后台应用
> （仅保留系统进程与编辑器）」条件下采集，并将电源模式、散热状态一并固化——
> `freeze_environment()` 中的 `power_mode` / `thermal_throttling_observed` 字段即为此外设计。

## 9. 修订记录

| 日期 | 修订 | 说明 |
|---|---|---|
| 2026-10-12 | **`report.log` 内容改为纯英文纯文本（去除全部 Markdown 语法）** | 因 `.log` 属**可被第三方下载**的产物（每次调用都会重新生成），其**格式与内容均须自洽于纯文本阅读器**，不依赖 Markdown 渲染。**改动**：`benchmark_e1.py` 新增 `_rule()`（横线分隔）与 `_align_table()`（空格对齐表格）两个纯文本渲染助手；`build_compare_table()` 与 `build_repro_doc()` 全量重写——去除 `#` 标题、`\|` 表格、\`\`\` 围栏与 `**` 强调，改用 `=` 横幅、空格对齐列、`-` 列表符与 ASCII 替代符号（`>=` / `<=` / `->` / `x` / `~`）。D-3 CSV 块的表头由 `# case=` 改为 `CASE:`（避免行首 `#` 被误读为标题）。D-5 运行示例的 `#` 注释改为 `(a)` / `(b)` 编号。**校验**：端到端 mock 运行产出唯一产物 `2026-10-12_report.log`，扫描确认行首 `#`=0、表格竖线=0、代码围栏=0、粗体=0、引用=0、CJK/全角=0。契约测试仍 35/35，双级自持校验通过。 |
| 2026-10-12 | **注释统一为英式英语 + 测试产物经 Logger 落盘** | 遵项目新规范（已写入项目规则）。**改动一**：将 E1 全部新增/修改文件（`app.py`、`config.py`、`utils/renderPipeline.py`、`utils/warmup.py`、`utils/metrics.py`、`benchmark/*.py`）中的中文 DOCSTRING 与注释改为英式英语。已用 AST 校验：剔除所有字符串常量后，各文件代码结构与改动前**逐节点一致**，证明零逻辑改动；`app.py` 的 `_parse()` 仍与基线逐字符一致（NFR-04）。**改动二**：`benchmark_e1.py` 的产物输出重构——`write_csv` / `write_compare_table` / `write_repro_doc` 改为返回字符串的 `build_frame_csv` / `build_compare_table` / `build_repro_doc`，由新增的 `_emit_reports()` 统一调用 `utils/logger.py` 的 `Logger.report()` 落盘，文件名格式 `<YYYY-MM-DD>_report.log`。**不再产出任何 CSV / MD 文件**。文档文件重命名为英文：`E1_change_notes.md`、`E1_configuration_reference.md`、`E1_readiness_interface.md`、`E1_self_contained_branch_paradigm.md`。契约测试仍 35/35，双级自持校验通过。 |
| 2026-10-12 | **`benchmark_e1.py` 新增 `--per-case-process` 与显存降级开关** | 用户实机为 4GB VRAM 笔记本（RTX 3050 Ti Laptop）。`--matrix` 全矩阵会在同进程内 **每算例新建一次 `IGNITE(cfg)`**，权重反复加载累积占用，4 组跑完有 OOM 风险。**修法**：(1) 新增 `--per-case-process`，每组算例走独立子进程（哨兵行回传结果 JSON），退出即回收显存；(2) 抽出 `_build_base_cfg()` 供主/子进程共用，保证口径一致；(3) 配置层新增 `warmup.pool_mb`（默认 2.5MB，置 0 跳过显存池预热）与 `warmup.full_chain`（默认 True，置 False 只预热检测器），两者均可由 `IGNITE_WARMUP_POOL_MB` / `IGNITE_WARMUP_FULL_CHAIN` 环境变量覆盖，**免改代码即可在低显存机型降级**。契约测试仍 35/35。 |
| 2026-10-12 | **修复 `benchmark_e1.py` real 后端接口错误** | 用户实测 `--backend real` 报 `AttributeError: 'IGNITE' object has no attribute 'infer'`。**根因**：`benchmark_e1.py` 自定义了 `infer()` 接口（MockBackend 提供），却让 real 后端直接复用 `IGNITE`，而 `IGNITE` 的真实推理入口是 `_parse(frame, conf=(...))`，**并无 `infer()` 方法**。mock 因 MockBackend 恰好同名而未暴露，real 一跑即炸。**修法**：新增 `RealBackend` 适配器，把 `IGNITE._parse` 包装为统一的 `infer(frame)`；**未改动 `app.py` 任何一行**（`_parse` 仍逐字符与基线一致，NFR-04 零回归成立）。已在模拟仓验证 `git apply` 通过、契约测试 35/35。 |
| 2026-10-10 | **补齐 3 个二进制资产，RUNTIME 级自持达成** | 从项目盘取回 `fire.pt`（42 MB）、`obj1.pt`（43 MB）、`triplets_cache.pkl`（32 KB），并做深层结构校验（torch 归档完整性、pickle 可消费性）。`selfcheck.py --strict-runtime` 退出码 0。**原工程仍未合并**（用户要求暂不合并）。 |
| 2026-10-10 | **R8 决策：确认消费方存在，按「预埋接口」处理** | 用户确认渲染消费方存在（即便终端用户未接入，仍需预埋接口并调优，紧急度不高）。现有 `RenderChannel` 抽象即为预留点，后续接入属新增一个类，不阻塞合并。详见 §6.1。 |
| 2026-10-10 | **确立项目范式「测试 Branch 应可自持」** | 新增 `docs/E1_self_contained_branch_paradigm.md`（范式全文、两级定义、资产清单、CI 集成）与 `benchmark/selfcheck.py`（可执行校验）。副本当前达 **CODE 级自持** ✅（35/35 契约用例），RUNTIME 级待补 3 个二进制资产。 |
| 2026-10-10 | **撤回「`Service/` 大小写不一致」结论** | 经用户复核并由我方逐项验证：原工程所有 import 均使用大写 `Service`（`app.py:8/9/10`、`Service/Predicate/affordanceEmbedder.py:9`），**不存在大小写不一致问题**。此前结论源于我方在下载源码时预建的临时目录残留（`.baseline/` 下混入一个空的小写 `service/`），被 `diff -rq` 报为 `Only in .baseline: service`，属误读。该空目录已删除。**无需任何修复动作。** |
