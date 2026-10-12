# 项目范式 · 测试 Branch 应可自持

> **范式（Norm）**：**任何一个测试 Branch 应可以自持（Self-Contained）。**
>
> 定义：一个测试 Branch 自持 = 在**干净的检出（clean checkout）**上，**不依赖分支外的任何文件**，
> 即可完成该分支承诺的全部测试。
>
> 本文件把该范式**从口头约定升级为可执行检查**，工具为 `benchmark/selfcheck.py`。
> 本文件即 `selfcheck.py` 输出中 `§4` 所指向的资产清单出处。

---

## §1 为什么立这条范式

E1 的验收方式决定了它必须自持：

| 事实 | 后果 |
|---|---|
| E1 的收益是**延迟指标**（AC-01 无帧 > 50 ms、AC-02 p95 ≤ 30 ms） | 必须在**至少 2 档 RTX 真机**上复现（AC-09） |
| 真机往往不是开发机，且常处于**离线 / 受限网络**环境 | 分支若依赖外部路径取源码与资产，验收现场无法开工 |
| 基线含**大体积二进制资产**（权重 `.pt`、向量缓存 `.pkl`） | 全量入库会撑爆仓库，故不能简单地「什么都塞进 git」 |

> 一句话：**验收发生在别人的机器上，所以分支必须能自己站起来。**
> 若一个分支必须回到开发机上才能跑，那它就不是一个可交付的验收单元，只是一份「依赖了我本人」的草稿。

---

## §2 两级自持定义

由于「完整自持」与「仓库体积可控」存在直接冲突，范式把自持性**分为两级**度量。
**CODE 级是强制门槛，RUNTIME 级视分支性质而定。**

| 级别 | 含义 | 判定内容 | 适用范围 |
|---|---|---|---|
| **CODE 级** | 所有 `.py` 与配置齐备，**逻辑测试可跑通** | 入口文件存在 + 包内导入闭包完整 + 全部可编译 + 无占位文件 + 契约测试全绿 | **所有测试分支，强制门槛** |
| **RUNTIME 级** | CODE 级之上，**二进制资产齐备**，可跑真实推理与压测 | CODE 级通过 + `ASSETS` 清单齐备 | **需要真机验收的分支** |

两级的关系：

```
RUNTIME 级自持  ⊃  CODE 级自持
     ↑                    ↑
  真机验收分支         所有分支（含纯逻辑分支）
```

> **设计取舍**：逻辑正确性是**代码属性**，应该在任何机器、任何时间点可验证，故设为强制门槛；
> 而推理性能是**硬件属性**，必须在特定 GPU 上验证，其资产无法随分支轻量分发，故降为「按需」。

**E1 副本当前状态**：CODE 级 ✅ 通过（35/35 契约用例）；RUNTIME 级 ✅ **通过**（3 个资产已齐备，见 §4）。

---

## §3 `selfcheck.py` 如何执行判定

工具**不执行被测文件**（仅对契约测试子进程做一次执行），以静态分析为主，故在无 GPU、无 `ultralytics` 的机器上同样可跑。

| 序 | 检查项 | 方法 | 不通过的后果 |
|---|---|---|---|
| 1 | **静态导入图闭包** | 从 4 个入口出发，用 `ast` 解析全部 `import` / `from ... import`；跳过 stdlib 与第三方白名单，凡**包内模块**必须物理存在（深度优先追踪传递依赖） | CODE 级 ❌ |
| 2 | **语法编译** | `py_compile` 遍历全部非隐藏 `.py` | CODE 级 ❌ |
| 3 | **占位/损坏文件** | 拦截 0 字节文件、以及开头为 `<?xml` 且含 `<Error>` 的**网盘错误页** | CODE 级 ❌ |
| 4 | **契约测试** | 子进程执行 `benchmark/test_e1_contracts.py`，校验退出码 | CODE 级 ❌ |
| 5 | **资产清单** | 按 `ASSETS` 核对文件存在且非空 | RUNTIME 级 ❌（不影响 CODE 级） |

**第 3 项为什么必要**：从项目盘拉取文件时，失败会留下一个 XML 错误页（HTTP 200 但内容为报错）。
这种文件**能通过「存在性」检查**，却会在导入或运行时炸掉，是最典型的「静默损坏」。
仅靠 `ls` 无法发现，必须在范式里显式拦截。

**白名单**：`THIRD_PARTY = {torch, torchvision, cv2, numpy, ultralytics, av, transformers, yaml, PIL, matplotlib, pandas}`。
白名单只回答「这个 import 算不算包内模块」，**不检查第三方是否真的已安装** ——
环境是否就绪由 `README.md` / `requirements.txt` 负责，自持性只关心「文件是否齐全」，两者职责分离。

### 用法

```bash
python benchmark/selfcheck.py                  # 人类可读报告
python benchmark/selfcheck.py --json           # 机器可读，供 CI 消费
python benchmark/selfcheck.py --strict-runtime # RUNTIME 缺失也判失败
python benchmark/selfcheck.py --skip-contract  # 跳过契约测试（仅做静态检查）
```

退出码：CODE 级不通过 → `1`；启用 `--strict-runtime` 且 RUNTIME 级不通过 → `1`；否则 `0`。

---

## §4 资产清单（RUNTIME 级自持需补齐）

以下二进制资产**不纳入版本库**（体积原因），但真机验收分支必须齐备。
清单与 `selfcheck.py` 的 `ASSETS` 常量**一一对应**，修改资产请同步改两处。

> **当前状态：✅ 三件已全部补齐并校验通过**（`selfcheck.py --strict-runtime` 退出码 0）。
> 累计约 85 MB（`fire.pt` 42M + `obj1.pt` 43M + `triplets_cache.pkl` 32K）。

| # | 路径 | 用途 | 获取来源 | 校验结果 | 缺失时的行为 |
|---|---|---|---|---|---|
| 1 | `Service/ObjectDetector/fire.pt` | 火焰检测权重（微调后） | 项目盘 `IGNITE/Service/ObjectDetector/fire.pt` | ✅ 42 MB；776 项 torch 归档，含 `best/data.pkl` | 无法加载检测器，检测链路直接不可用 |
| 2 | `Service/ObjectDetector/obj1.pt` | 环境物体检测权重（微调后） | 项目盘 `IGNITE/Service/ObjectDetector/obj1.pt` | ✅ 43 MB；776 项 torch 归档，含 `best/data.pkl` | 谓词推理缺少物体输入，精度类验收（AC-05）无法执行 |
| 3 | `Service/Semantic/triplets_cache.pkl` | 三元组向量缓存 | 项目盘 `IGNITE/Service/Semantic/triplets_cache.pkl` | ✅ 32 KB；protocol-4 pickle，2 条 `{status, triplet, embedding[1,384]}` | 可由 `common_knowledge.py` **重建**（更慢，且需 `transformers`）；延迟指标会被重建耗时污染 |

**资产有效性核验方法**（比「文件存在」更强的检查，补齐后建议手工跑一次）：

```bash
python3 - <<'EOF'
import pickle, zipfile
# 权重：应为 torch zip 容器且含 data.pkl
for w in ["Service/ObjectDetector/fire.pt", "Service/ObjectDetector/obj1.pt"]:
    with zipfile.ZipFile(w) as z:
        assert any(n.endswith("data.pkl") for n in z.namelist()), w
    print(w, "ok")
# 缓存：应能被 continuousLatentInferencer._update_ref_matrix 消费
ref = pickle.load(open("Service/Semantic/triplets_cache.pkl", "rb"))
assert all({"status", "triplet", "embedding"} <= set(it) for it in ref)
assert len({tuple(it["embedding"].squeeze().shape) for it in ref}) == 1
print("triplets_cache.pkl ok，条目数 =", len(ref))
EOF
```

> `selfcheck.py` 只校验「存在且非空」；上表第 4 列的深层校验（归档结构 / pickle 可消费性）是**人工补充**，
> 因为它们需要特定库（`zipfile`/`pickle`）且成本高于静态检查，不适合放进每次 CI。

### 补齐方式

```bash
# 假设已挂载/可访问项目盘
mkdir -p Service/ObjectDetector Service/Semantic
cp /path/to/IGNITE/Service/ObjectDetector/fire.pt            Service/ObjectDetector/
cp /path/to/IGNITE/Service/ObjectDetector/obj1.pt            Service/ObjectDetector/
cp /path/to/IGNITE/Service/Semantic/triplets_cache.pkl       Service/Semantic/

# 复核
python benchmark/selfcheck.py
# → 期望：RUNTIME 级自持 ✅ 通过
```

> **为什么权重不进 git**：单个 `.pt` 通常数十 MB 至数百 MB，`.pkl` 同理。
> 纳入版本库会让每次 clone 变慢、让 diff/branch 变重，且权重迭代与代码迭代节奏不同。
> 范式的立场是：**代码随分支走，资产按需分发** —— 但分发路径必须写在 §4，不能散落在聊天记录里。

---

## §5 CI 集成

把范式落进流水线，任何人都无法「忘记」：

```yaml
# .github/workflows/branch-selfcheck.yml（示意）
name: branch-selfcheck
on: [push, pull_request]
jobs:
  code-level:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      # CODE 级门槛：任何测试分支提交前必须通过
      - run: python benchmark/selfcheck.py || exit 1
```

**分层策略建议**

| 场景 | 建议做法 |
|---|---|
| PR 门禁（所有分支） | 只跑 CODE 级，快（秒级）、无需 GPU、无需资产 |
| 真机验收前的分支 | 另跑 `--strict-runtime`，缺资产直接拦下，避免到现场才发现（E1 现已 ✅ 通过） |
| 夜间回归 | CODE 级 + 真机 `--backend real` 压测（`benchmark/benchmark_e1.py --matrix`） |

---

## §6 如何把本范式套用到其他 Branch

新建任一测试 Branch 时，按下面 4 步即可达标：

1. **拷贝工具**：把 `benchmark/selfcheck.py` 复制到新分支的同名路径（该脚本无分支专属逻辑，仅 `ENTRYPOINTS` / `ASSETS` 两个常量需要按分支调整）。
2. **登记入口**：把该分支承诺「可跑」的文件写入 `ENTRYPOINTS`，逐条确保其导入闭包完整。
3. **登记资产**：若分支需要新的二进制/大文件，追加到 `ASSETS`，并**在本文档 §4 同步补一行**（路径 / 用途 / 来源 / 缺失行为）。
4. **跑一遍再提交**：`python benchmark/selfcheck.py`，CODE 级必须 ✅。

**契约测试的位置**：`selfcheck.py` 会执行 `benchmark/test_e1_contracts.py` 作为 CODE 级的操作性证明。
新分支若无同名脚本，要么沿用（把分支约束追加进去），要么修改 `check_contract_test()` 指向本分支的测试文件 ——
**但 CODE 级不能退化为「只有静态检查」**：范式要求必须有「真的跑得起来」的证据，光有文件不够。

---

## §7 自持性清单（提交前自查）

- [ ] `python benchmark/selfcheck.py` CODE 级 ✅ 通过
- [ ] `ENTRYPOINTS` 覆盖本分支所有对外承诺可跑的入口
- [ ] `ASSETS` 与本文件 §4 表格**一致**（无遗漏、无多余）
- [ ] 新增的包内模块都已在其被引用处可解析（无「本地能跑、换机器 import 失败」）
- [ ] 无 0 字节 / 网盘错误页残留文件
- [ ] 若声明可支持真机验收：已跑 `--strict-runtime` 且 RUNTIME 级 ✅
- [ ] 文档中**没有**任何指向「开发机本地路径」的隐藏依赖

---

## §8 与 E1 其他交付物的关系

| 交付物 | 关系 |
|---|---|
| `benchmark/selfcheck.py` | 本范式的**执行体**；本文件为其**文档出处** |
| `docs/E1_改动说明.md` §1 | 该表定义了「本分支包含什么」，本范式保证「这些内容足以独立跑起来」 |
| `docs/E1_readiness接口文档.md` | 运行时就绪信号（`probe().ready`）；本范式是**交付时**的静态就绪信号，两者互补 |
| `benchmark/benchmark_e1.py` | 真机压测入口；其可执行性由 RUNTIME 级自持保证 |

> 三者构成一条链：**自持（能否独立跑）→ 就绪（预热是否完成）→ 压测（性能是否达标）**。
> 本范式是这条链的第一环，也是最基础的一环 —— 跑不起来，后两环无从谈起。
