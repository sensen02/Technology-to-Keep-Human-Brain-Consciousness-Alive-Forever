# 腿特异前运动池严格图谱审计 — 预登记（先于分析写入）

- 目录（源码）：`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-descending-learning/20261107T203049Z/leg-pools/`
- 目录（桌面镜像）：`/home/sensen/Desktop/engine修复验收/banc-descending-learning/20261107T203049Z/leg-pools/`
- 本轮性质：**离线静态图结构审计**。仿真步 0、神经动力学步 0、学习更新 0、新突触 0、物理步 0。
- 上游阴性证据（不改写、不推翻）：`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-descending-learning/20261107T203049Z/intrinsic/REPORT.md`（无左右交替，PC1 49.5% 前/后模态）、`.../adaptation-screen/REPORT.md`（只放大运动细胞适应不能造出交替）。

---

## 0. 目的与三条不变式

目的：只用**已存在的 435,126 条严格边**，把"哪一类结构在原理上能打破主导的双侧对称模态"这件事变成可核对的数字；对做不到的部分明确写成**缺失数据**，而不是补边。

不变式（任何结果都不得违反）：

1. **I-E**：使用的边集恒等于冻结模块加载的 435,126 条边；不新增、不删除、不改符号、不改递质分类。
2. **I-W**：权重恒等于冻结模块里实际使用的 `_We` / `_Wi` 数值；本轮只**读取**，不重写。
3. **I-Z**：冻结文件（192931Z `module.py`、185910Z `module.py`、`engine/neural_cond.py`、`neural-interface/` 下全部输入）一个字节不写。仓库 tracked 文件（`AGENTS.md`、`GOAL_ROADMAP.zh-CN.md`、`GOAL_TRACKER.zh-CN.md`）本轮不修改。

---

## 1. 输入与只读哈希（分析开始前实测）

| 角色 | 完整路径 | SHA256 |
|---|---|---|
| 冻结模块（本轮权威 `_We`/`_Wi` 与命名空间来源） | `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T192931Z/dynamics/module.py` | `c5106d46a62bedd31faccdbe981764106c23c4dc491890e6e4d21ef531718250` |
| 冻结原始加载器（`LocalCircuit` 构造体） | `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T185910Z/dynamics/module.py` | `3443853d8ebe725f8fedcb2e022b53ca4ed470d87f06a8d3ed71fa049f150676` |
| 引擎电导网络 | `/run/media/sensen/Data2/cell_wound_prototype/engine/neural_cond.py` | `27e46d0a9d2b3014ead4c07c4ae64a6b46a069c35ff772c7533210b053164a4e` |
| 20k 候选/边/递质 | `/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/neural-interface/LOCAL_MODULE_CANDIDATE.npz` | `7a7ce0d7d2166f81d0f9c8c1acd73342b854c7f33915d6d6cbf68ef89738395e` |
| 323 单腿运动读出端口 | `/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/neural-interface/LOCAL_MOTOR_READOUT_PORTS.csv` | `c78650b4a8b705512523674f4c47dd7e7718261db5b51d37519f910b558df39a` |
| 组索引（sensory/descending/ascending 边界） | `/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/neural-interface/GROUP_CACHE_INDICES.npz` | `557cfc944270a34053d74d3eebe39af453eefe530c22ae5ce656f92fbff3b09a` |
| 感觉通道歧义剔除表 | `/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/neural-interface/SENSORY_CHANNEL_REVIEW.json` | `906ccfca7b27698d59666d0d9ce15f37ff43b23a013d44168902467757bb4b64` |

任何哈希不符即**中止并保留 trace**，不继续分析、不写结果。

## 1.1 冻结权重契约（读数，不推导）

冻结加载器按
`weight[e] = 0.02 uS * synapse_counts[e] / incoming_raw[post[e]]`，
`incoming_raw[q] = Σ_{e: post=q} synapse_counts[e]`（两种受体类型一起归一），
`_We` = ACh 边，`_Wi` = GABA/glutamate 边，均为 `shape=(20000,20000)` 的 CSR，索引为 `[post, pre]`。
因此单位是 **µS**，并且**每个有入边神经元的入总和恒为 0.02 µS**（`.02 uS 约定`）。本轮的"块电导"一律是这些 µS 数值的直接求和，不换成无量纲增益。

## 1.2 预登记前已做的检查（完整披露）

为避免定义踩空，写本预登记**之前**只做了下列结构性健全性检查，**未**计算任何池大小、块计数、对称性指数或三足耦合量：

1. `LOCAL_MODULE_CANDIDATE.npz` 键名/shape/dtype 清点；`source_verified_nt` 的取值与计数。
2. 严格子集 `eligible = nt ∈ {acetylcholine, gaba, glutamate}` 的边数 = 435,126；其中 ACh 207,166、GABA/glutamate 227,960。
3. 严格子集内有序对 `(pre_local, post_local)` 无重复（435,126 个唯一有序对），无自环（`pre==post` 计数 0），`synapse_counts` 为 3…805 的正整数。
4. 冻结模块可实例化；`_We.nnz=207,166`、`_Wi.nnz=227,960`、`_We.nnz+_Wi.nnz=435,126`；`ΣWe=126.82030644091996 µS`、`ΣWi=254.01969355908005 µS`；有入边的神经元 19,042 个，每个入总 = 0.02 µS，故模块总电导 `T = 19,042 × 0.02 = 380.84 µS`。
5. `LOCAL_MOTOR_READOUT_PORTS.csv` 中 `readout_eligible_single_leg == 'True'` 的行 = 323，`nerve_matched_legs` 全为单个小写腿名，分布 lf 43 / lh 59 / lm 60 / rf 42 / rh 59 / rm 60。

以上 5 项是**输入身份与权重契约**，不是本轮的结论。本轮所有池、块、对称性、耦合结论均在写入本文件之后才计算。

---

## 2. 精确定义

### 2.1 符号

- `V` = 局部索引 0…19999（20,000 个真实细胞）。
- `E` = 435,126 条严格边，每条 `(pre, post, nt, syn)`；`ACh` → 兴奋矩阵 `_We`，`GABA/glutamate` → 抑制矩阵 `_Wi`。
- `M` = 323 个 `mn_local`（运动读出细胞）；`leg(q) ∈ {lf,lh,lm,rf,rh,rm}` 直接取自 CSV，不由图推断。
- `L = {lf, lh, lm, rf, rh, rm}`。

### 2.2 直接腿可达集（DirectLegReach）

对每个 `p ∈ V`：
`D(p) = { leg(q) : ∃ e=(pre=p, post=q) ∈ E, q ∈ M }`。
即"p 通过**单条已存在边**进入的运动腿集合"。允许 `p ∈ M`（此时它可能把同腿或别的腿的 MN 当目标）。

### 2.3 2 跳腿可达集（TwoHopLegReach）

`H(p) = { leg(q) : ∃ x ∈ V \ M, (p→x) ∈ E, (x→q) ∈ E, q ∈ M }`。
中间细胞**限定为非 MN**（题目要求"通过非 MN 细胞"）。`H` 只统计长度恰为 2 的路径，不与 `D` 合并；`D ∪ H` 另行作为补充视图报告。

### 2.4 桶划分（主划分，覆盖全部 20,000 个细胞，互斥且穷尽）

对每个 `p ∈ V`，按下述**顺序**判定，得到唯一 `bucket(p)`：

1. `p ∈ M` → `pool_mn`
2. `D(p) = ∅` → `pool_nonmotor`
3. `|D(p)| = 1` → `pool_<leg>`（`pool_lf`/`pool_lh`/`pool_lm`/`pool_rf`/`pool_rh`/`pool_rm`）
4. `|D(p)| ≥ 2` → `pool_shared`

即 9 个桶：`pool_lf, pool_lh, pool_lm, pool_rf, pool_rh, pool_rm, pool_shared, pool_nonmotor, pool_mn`。
`pool_mn` 单列的理由：323 个细胞的腿标签来自读出端口 CSV 的实测神经匹配，不是从边推断出来的；把它并进前运动腿池会把"读出靶"和"前运动来源"混为一谈，并且会让"每腿池大小"变成 43/59/60 这类读出端口数而不是前运动结构量。为满足"每个神经元恰好在一个桶"，`pool_mn` 仍计数在划分里。

**补充划分**（次级，不影响主划分判定）：把 2.2 换成 `D(p) ∪ H(p)` 重复同一顺序判定，得到 `pool2_*`，只用于回答"只按直接边判定的 nonmotor 池里，有多少细胞其实 2 跳就能到 MN"。

### 2.5 块（block）

对 `X ∈ {_We, _Wi}` 与有序桶对 `(b_src, b_dst)`：
`block_X(b_src→b_dst) = { X[post, pre] : pre ∈ b_src, post ∈ b_dst }`。
每个块报告：`edges`（nnz，整数）、`synapses`（源 `synapse_counts` 之和，整数）、`conductance_uS`（权重和）、`mean/median/min/max` 与分位 `q05,q25,q75,q95`（µS/边）。

### 2.6 聚合视图（全部由 2.5 的 9×9 块加总得到，不重新定义）

- **池内**：6 个腿池的对角块 `pool_k→pool_k`。
- **左→右 / 右→左**：`{lf,lh,lm}` 与 `{rf,rh,rm}` 之间的 4 组加总（源侧定侧 × 目标侧定侧）。
- **前→后 / 后→前**：`{lf,rf}` 与 `{lh,rh}` 之间的 4 组加总。
- **6 腿两两交叉矩阵**：6×6 有向矩阵 `C[leg_src][leg_dst]`（只看 6 个腿池），并给出全部 15 个无序腿对的双向加总，按预登记分组命名：
  - 同侧（ipsilateral，6 对）：LF-LM, LM-LH, LF-LH, RF-RM, RM-RH, RF-RH
  - 对侧同行（contralateral same-row，3 对）：LF-RF, LM-RM, LH-RH
  - 对侧跨行（contralateral cross-row，6 对）：LF-RM, LF-RH, LH-RM, LH-RF, LM-RF, LM-RH
  - 其中**三足搭档**：LF-RH、RF-LH（属于跨行子集，但单独列出）
- **前运动驱动矩阵**：9 个源桶 × 6 个目标 MN 腿（`leg(q)`，`q ∈ M`）的 `edges`/`conductance_uS`/`synapses`，两种受体分别给。这是"哪一类前运动细胞以多大真实电导压向哪条腿的 MN"的直接读数。

### 2.7 双侧对称性指数（精确量化，不四舍五入）

镜像映射 `mir(k)`：`lf↔rf`、`lh↔rh`、`lm↔rm`，`shared↔shared`、`nonmotor↔nonmotor`、`mn↔mn`。

1. **细胞计数不对称**：对每个镜像桶对，`asym_count = (a−b)/(a+b)`；同时给整数 `a`、`b` 与 `|a−b|`。
2. **块电导镜像不对称**：对 9×9 电导矩阵 `G[b_src][b_dst]`，定义镜像矩阵 `G'[b_src][b_dst] = G[mir(b_dst)][mir(b_src)]`（同时换侧并反向）。
   `asym_index = Σ|G − G'| / Σ(G + G')`（L1 相对差），并给 `Σ|G−G'|`、`Σ(G+G')` 的 µS 原值。
3. **腿对级不对称**：对每个无序腿对的 `{A→B, B→A}` 加总电导与其镜像腿对的同一加总，给绝对 µS 差与相对差。
4. **行内总和守恒检查（结构事实，不是偏好）**：每个有入边神经元的 `We+Wi` 入总 = 0.02 µS；按 2.4 的划分可把它拆成"来自哪个源桶"，报告每个目标桶的入总在 6 源腿桶 + shared + nonmotor + mn 上的**精确分解**（9 个分量的和必须逐位等于 0.02×该桶有入边细胞数）。

## 3. 预登记判定规则（阈值在分析前固定）

- `ASYM_TOL = 0.05`。`|asym| < 0.05` → 该量记为"在 5% 带内（未检出不对称）"；`|asym| ≥ 0.05` → 记为"检出不对称"。无论落在哪边，**原始整数与 µS 原值都如实列出**，标签只是给读者的分类。
- 块强弱标签，以模块总电导 `T = 380.84 µS` 为基准（`T` 见 1.2 第 4 条，是 `.02 uS 约定` × 19,042 的直接结果）：
  - `absent`：`edges = 0`（精确整数判定）
  - `trace`：`0 < G ≤ 1e-4·T`
  - `very_weak`：`1e-4·T < G ≤ 1e-3·T`
  - `weak`：`1e-3·T < G ≤ 1e-2·T`
  - `moderate`：`1e-2·T < G ≤ 1e-1·T`
  - `strong`：`G > 1e-1·T`
- **"能否耦合"的判定用精确路径计数，不用电导阈值**：路径长度 ≤ 2 时，只要计数 ≥ 1 即记为"存在结构耦合"；计数 = 0 记为"该长度上无耦合（缺失数据）"。长度 ≥ 3 的路径本轮**不枚举**（20k 规模上几乎处处连通，枚举无信息量），并在报告中明确写"未分析"。
- 统计层面：本轮全部是**描述性计数与求和**，不做假设检验、不给 p 值、不做多重比较修正；不因为某个数字"看起来显著"就追加分析。若分析中发现定义遗漏，只允许在 `DEVIATIONS.md` 里追加记录并重跑全量，不允许静默改判据。

## 4. 缺失耦合的分类法（对着 5 个步态相关腿对做）

对每个 `(A,B) ∈ {LF-RF, LM-RM, LH-RH, LF-RH, RF-LH}`，按下列**有序层级**给出结论，每层都必须给出精确整数：

- **L0 直接前运动互连**：`pool_A → pool_B` 与 `pool_B → pool_A` 的边数（两种受体分开，且给 `_We`/`_Wi` 分别的电导）。
  - 由 2.4 的定义，`pool_A` 的细胞"直接 MN 目标全部属于 A"，因此它对 B 的 MN **不可能**有直接边；这不是遗漏而是定义推论，必须在报告里明确写出，避免读者误以为"没有直接边"是数据缺失。
- **L1 共享池桥**：同时以 `A` 与 `B` 的 MN 为直接目标的细胞数（即 `shared` 中 `D(p) = {A,B}` 或 `D(p) ⊇ {A,B}` 的子集），以及这些细胞接收 / 发出的边数与电导。
- **L2 两跳通路**：`#paths_2(A→B)` 与 `#paths_2(B→A)`，中间细胞限定非 MN；给中间细胞的去重计数与通路条数。同时给 `#paths_2` 中**经过 `shared` 池中间细胞**的条数占比。
- **L3 判定**：若 L0、L1、L2 全部为 0，则写"严格子图在该腿对之间**没有**可用于打破该模态的耦合结构 —— 缺失数据发现"。若只有 L2 非零，则写"仅两跳、无直接/单桥"，并给出通路条数量级。

## 5. 参数化提案的预登记约束（仅设计文档，本轮不跑）

提案文件：`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-descending-learning/20261107T203049Z/leg-pools/PROPOSED_SCALING_PARAMS.json`。硬约束：

1. **只重标定已存在边**的权重：`w'[e] = s[class(e)] · w[e]`，`s > 0`。不新增边、不删除边、不改递质类别、不改符号。
2. 类别函数 `class(·)` 必须是 435,126 条边上的**互斥穷尽划分**：每条边恰好落在一个类别里，JSON 里给出每类的边数与冻结电导合计（µS），使 `Σ class edges = 435,126` 可被独立核对。
3. **自由标量数 ≤ 5**；每个标量 `s ∈ [0.25, 4.0]`（含端点），并给出建议的 5 档候选网格 `{0.25, 0.5, 1, 2, 4}`。
4. **`.02 uS 约定` 不被破坏**：给出两个变体——
   - `V1 row_renormalized`：施加 `w'` 后再把每个目标神经元的入边整体归一到**恰好 0.02 µS**（逐位与冻结一致的守恒），逐神经元入总 = 0.02 µS；
   - `V2 bounded_unrenormalized`：不归一，但要求每个目标神经元入总落在 `[0.5, 2.0] × 0.02 µS`，超出即投影回边界。
   两个变体都要在 JSON 里以机器可读的方式写明，下一轮必须**择一**预登记。
5. 本提案**没有**被本轮的任何一个数字证实有效；`s ≠ 1` 的任何取值都只是待检验假设。不得在后续报告中把"能重标定"写成"能产生步态"。
6. 下一轮若真做实验，必须是新时间戳目录 + 新预登记 + 开环筛查先于物理；不得以本轮结构审计充当阳性结果。

## 6. 输出文件（本轮要写的）

| 文件 | 内容 |
|---|---|
| `PREREGISTRATION.md` / `PREREGISTRATION.json` | 本文件（先写） |
| `leg_pools.py` | 全部分析代码（单入口，可复跑） |
| `POOLS.json` | 池大小、桶成员计数、补充划分、每细胞可达掩码统计 |
| `POOLS_AND_REACH.npz` | 每细胞 `bucket_id`、`leg_mask_direct`、`leg_mask_2hop`、计数数组（供独立核验） |
| `BLOCKS.json` | 9×9 块（两种受体）：边数、突触数、电导、分布分位；聚合视图；前运动驱动矩阵 |
| `SYMMETRY.json` | 2.7 的全部不对称指数与原始量 |
| `COUPLING.json` | 第 4 节的 L0–L3 精确计数与判定 |
| `PROPOSED_SCALING_PARAMS.json` | 第 5 节的设计文档（机器可读） |
| `RESULT.json` | 本轮汇总（含预登记 sha256、全部输入哈希、分析代码 sha256、关键数字） |
| `verify_leg_pools.py` | 独立核验脚本（不 import 分析代码） |
| `INDEPENDENT_VERIFICATION.json` | 独立核验结果（逐条 PASS/FAIL） |
| `REPORT.md` | 人类可读报告：完整路径、精确数字、缺失耦合、明确界限 |
| `MIRROR_MANIFEST.json` | 桌面镜像清单（只放桌面目录，不参与自身镜像集） |

## 7. 本轮明确不做的事

- 不跑神经动力学（0 神经步）、不跑物理（0 物理步）、不训练、不优化。
- 不新增/删除/翻转任何边；不改冻结文件的任何字节；不改 tracked 文件。
- 不把任何一个桶叫成生物学细胞类型，不宣称果蝇解剖结论、不宣称功能、不宣称任何行为可行性。这是**严格过滤的 20,000 细胞子图上的离线结构审计**，受体符号/增益/延迟均未标定。
- 不改变任何验收判据，不触碰 53 个物理候选的失败记录，不改写 `intrinsic/` 与 `adaptation-screen/` 的历史结论。
- 不给出"能产生三足步态"的结论；本轮的阳性上限是"某类耦合结构在严格子图里存在 / 不存在，及其精确规模"。

---

预登记写入时间以本文件 mtime 为准；`RESULT.json` 会记录本文件的 sha256，任何后续改动都会使核验失败。
