# 腿特异前运动池的严格图谱审计（离线结构分析，0 仿真步）

- 源码目录：`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-descending-learning/20261107T203049Z/leg-pools/`
- 桌面镜像：`/home/sensen/Desktop/engine修复验收/banc-descending-learning/20261107T203049Z/leg-pools/`
- 本轮实际开销：神经步 **0**、物理步 **0**、学习更新 **0**、新增边 **0**、渲染 **0**、forward **0**；只有 1 次离线数组分析 + 1 次独立核验。
- 目标层级：S3 局部运动（结构前提诊断）。本报告**不是**行为证据、**不是**解剖结论、**不是**功能结论。

## 英文摘要（English abstract）

Using only the frozen 435,126 strict edges (no added edges, no dynamics), this audit shows:
(i) 4,723 of the 20,000 cells have a direct edge onto at least one of the 323 motor neurons and
9,785 reach at least one motor leg within two hops; the leg-exclusive premotor pools are
LF 548 / LH 771 / LM 756 / RF 639 / RH 776 / RM 796, shared 437, no-leg 14,954, plus the 323
motor cells themselves.
(ii) Every one of the five gait-relevant leg pairs — including both tripod partners LF–RH and
RF–LH — **does** have direct excitatory and inhibitory edges, shared-cell bridges and two-hop
paths. The needed coupling is **not missing**; it is **weak**: the two tripod pairs carry
266 edges / 0.1348225758151581 µS, i.e. 11.08 % of the contralateral same-row class and
0.035 % of the 380.84 µS module total.
(iii) All 323 motor neurons have **zero outgoing edges**: they are strict sinks, so no
motor-neuron dynamics can feed back into the network through chemical synapses.
(iv) Pooled bilateral asymmetry, corrected rule: 0.0236 (`_We`), 0.0481 (`_Wi`), 0.0382 (both)
over the full 9×9 block matrix, and 0.0401 / 0.0930 / 0.0661 restricted to the six leg pools.
Structural asymmetry exists at the block level, yet the frozen model's leg-rate dynamics were
bilaterally symmetric — this audit cannot and does not claim structure implies function.
(v) A three-scalar bounded rescaling proposal is provided as design only; even at its upper
bound the tripod class moves by +0.4044677274454743 µS (0.106 % of the module total).

---

## 1. 输入身份与只读哈希（分析开始前实测，逐位复核通过）

| 角色 | 完整路径 | SHA256 |
|---|---|---|
| 冻结模块（`_We`/`_Wi` 权威来源） | `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T192931Z/dynamics/module.py` | `c5106d46a62bedd31faccdbe981764106c23c4dc491890e6e4d21ef531718250` |
| 冻结原始加载器 | `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T185910Z/dynamics/module.py` | `3443853d8ebe725f8fedcb2e022b53ca4ed470d87f06a8d3ed71fa049f150676` |
| 引擎电导网络 | `/run/media/sensen/Data2/cell_wound_prototype/engine/neural_cond.py` | `27e46d0a9d2b3014ead4c07c4ae64a6b46a069c35ff772c7533210b053164a4e` |
| 20k 候选/边/递质 | `/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/neural-interface/LOCAL_MODULE_CANDIDATE.npz` | `7a7ce0d7d2166f81d0f9c8c1acd73342b854c7f33915d6d6cbf68ef89738395e` |
| 323 单腿运动端口 CSV | `/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/neural-interface/LOCAL_MOTOR_READOUT_PORTS.csv` | `c78650b4a8b705512523674f4c47dd7e7718261db5b51d37519f910b558df39a` |
| 组索引 | `/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/neural-interface/GROUP_CACHE_INDICES.npz` | `557cfc944270a34053d74d3eebe39af453eefe530c22ae5ce656f92fbff3b09a` |
| 感觉通道歧义表 | `/run/media/sensen/Data2/cell_wound_prototype/experiments/flybody-retarget/20261107T182940Z/neural-interface/SENSORY_CHANNEL_REVIEW.json` | `906ccfca7b27698d59666d0d9ce15f37ff43b23a013d44168902467757bb4b64` |

上游阴性证据（不改写）：`.../20261107T203049Z/intrinsic/REPORT.md`
`e356f1a8001cb5f4ed376514414bd6447514f2a9dbf7249907b25fb6fd1ff24d`、
`.../adaptation-screen/REPORT.md` `6cfb421049aa7c58e669b28237939b0f7a4ddbeb0a519d37f6328e33d7b10349`。

**权重契约（读数，不推导）**：`weight[e] = 0.02 µS × synapse_counts[e] / Σ(两受体 synapse_counts into post[e])`，
行索引为突触后神经元。实测：

- 边 435,126 条（ACh **207,166** → `_We`；GABA/glutamate **227,960** → `_Wi`），`_We.nnz + _Wi.nnz = 435,126`；
- `Σ_We = 126.82030644091996 µS`、`Σ_Wi = 254.01969355908005 µS`，合计 **380.840000000000 µS**（= 19,042 个有入边神经元 × 0.02 µS）；
- 逐神经元的 `We+Wi` 入总 = **0.02 µS**（19,042 行，全部行逐位通过 1e-12 检查）；
- 单边电导范围 8.3904349042092e-06 … 0.02 µS，中位 0.0003243243243243243 µS，q05 3.614457831325301e-05、q95 0.003448275862068966 µS；
- 源突触计数合计 4,199,733（`_We` 1,899,978 / `_Wi` 2,299,755）；
- 严格边集内**无重复有序对、无自环**；`0.02·syn/incoming` 重算与冻结矩阵的最大差 = 0（逐位相同）。

## 2. 定义（预登记原文，未改动）

`D(p)` = p 通过**单条已存在边**进入的运动腿集合；`H(p)` = p 经**非 MN 中间细胞**两跳到 MN 的腿集合。
桶判定顺序：`p ∈ M → pool_mn`；`D = ∅ → pool_nonmotor`；`|D| = 1 → pool_<leg>`；`|D| ≥ 2 → pool_shared`。
9 个桶互斥穷尽，覆盖全部 20,000 个细胞。`pool_mn` 单列，因为 323 个腿标签来自 CSV 的实测神经匹配，
不是从边推断的。块 = `X[post ∈ 目标桶, pre ∈ 源桶]`，两种受体分开，单位 µS。详见
`PREREGISTRATION.md` / `PREREGISTRATION.json`。

## 3. 结果

### 3.1 池大小（主划分，`POOLS.json`）

| 桶 | 细胞数 | 占 20,000 | 说明 |
|---|---:|---:|---|
| `pool_lf` | 548 | 2.740 % | 直接 MN 目标**全部**属 LF |
| `pool_lh` | 771 | 3.855 % | 全部属 LH |
| `pool_lm` | 756 | 3.780 % | 全部属 LM |
| `pool_rf` | 639 | 3.195 % | 全部属 RF |
| `pool_rh` | 776 | 3.880 % | 全部属 RH |
| `pool_rm` | 796 | 3.980 % | 全部属 RM |
| `pool_shared` | 437 | 2.185 % | 直接目标跨**多于一条**腿 |
| `pool_nonmotor` | 14,954 | 74.770 % | 无直接 MN 目标 |
| `pool_mn` | 323 | 1.615 % | 运动读出细胞本身 |
| 合计 | **20,000** | 100 % | 互斥穷尽 |

- 6 个腿专属池合计 **4,286** 个细胞（21.43 %）；`shared` 仅 **437**（2.185 %）。
- **4,723** 个细胞有直接 MN 目标；**9,785** 个细胞存在"经非 MN 中间细胞"的两跳通路；
  取"直接 ∪ 两跳"口径，**9,835** 个非 MN 细胞在两跳内可达至少一条腿的 MN，**10,165** 个到不了。
- 补充划分（`D ∪ H`）：LF 317 / LH 446 / LM 403 / RF 418 / RH 562 / RM 327 / shared **7,362** / no-leg **9,842** / mn 323。
  即"只按直接边判成 nonmotor 的 14,954 个细胞里，有 **5,112** 个其实两跳就能到 MN"。
  这说明**直接边口径会系统性低估**腿特异结构，也是后面耦合分析必须做两跳的原因。
- 组合分布中最大族是"无腿"15,277（= 14,954 + 323，即**没有任何细胞以 MN 为直接目标之外的**；
  其中 323 个 MN 自身属于这一族），其次是单腿池；跨腿共享的 437 个细胞里最大族是 `lh+rh` 62、`lh+lm` 59、`rh+rm` 50。
- 端口身份交叠（仅身份，不含功能推断）：`allowed_sensory_local` **5,103** 个中，落在腿专属池的 24+79+69+52+91+55 = **370**，
  落在 `shared` **2**，其余 4,731 在 `pool_nonmotor`；`descending_boundary` 1,327 个中仅 **32** 个落在腿专属/共享池（腿专属 8 + shared 24），
  **1,295** 在 `pool_nonmotor`。下行入口基本不走腿专属前运动池，这是本轮一个具体结构事实。
- 出/入度：**8,934** 个细胞无出边，**958** 个细胞无入边。

### 3.2 块统计（9×9，两种受体分开，`BLOCKS.json`）

边数（`_We + _Wi`），行 = 源桶，列 = 目标桶：

| 源 \ 目标 | lf | lh | lm | rf | rh | rm | shared | nonmotor | mn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **lf** | 10,692 | 92 | 202 | 498 | 34 | 40 | 1,141 | 12,245 | 1,815 |
| **lh** | 157 | 14,888 | 542 | 105 | 498 | 98 | 1,387 | 10,766 | 5,354 |
| **lm** | 305 | 441 | 14,935 | 103 | 104 | 299 | 1,865 | 11,976 | 4,653 |
| **rf** | 569 | 43 | 77 | 9,655 | 78 | 343 | 1,171 | 11,790 | 2,443 |
| **rh** | 84 | 397 | 61 | 118 | 13,295 | 500 | 1,155 | 11,190 | 4,880 |
| **rm** | 162 | 159 | 407 | 506 | 712 | 17,453 | 2,260 | 14,742 | 5,381 |
| **shared** | 2,408 | 2,457 | 3,100 | 2,159 | 2,140 | 3,415 | 2,771 | 17,095 | 5,488 |
| **nonmotor** | 7,567 | 4,950 | 7,214 | 6,000 | 5,569 | 7,814 | 9,994 | 150,119 | **0** |
| **mn** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** |

总电导（µS，`_We + _Wi`）：

| 源 \ 目标 | lf | lh | lm | rf | rh | rm | shared | nonmotor | mn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **lf** | 5.174604 | 0.022648 | 0.083596 | 0.169530 | 0.010758 | 0.024963 | 0.409 | 8.649907 | 0.736 |
| **lh** | 0.109535 | 9.506278 | 0.292341 | 0.055285 | 0.252493 | 0.073017 | 0.778 | 10.258935 | 0.932 |
| **lm** | 0.172915 | 0.207558 | 7.895709 | 0.089658 | 0.060894 | 0.163350 | 0.649 | 9.076525 | 1.065 |
| **rf** | 0.196026 | 0.016768 | 0.041466 | 5.914449 | 0.047869 | 0.126566 | 0.373 | 9.446627 | 0.747 |
| **rh** | 0.052012 | 0.197781 | 0.057720 | 0.090892 | 8.672336 | 0.250234 | 0.638 | 14.822072 | 1.019 |
| **rm** | 0.121305 | 0.142508 | 0.237384 | 0.313209 | 0.378964 | 8.884195 | 0.778 | 12.126356 | 1.041 |
| **shared** | 1.175 | 1.502 | 1.645 | 1.302 | 1.393 | 1.808 | 1.338 | 13.583 | 0.920 |
| **nonmotor** | 3.919 | 3.764 | 4.806 | 4.765 | 4.664 | 4.570 | 3.717 | **202.317** | **0.000** |
| **mn** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** |

聚合视图（`both` 受体，占比以 T = 380.84 µS 为基准）：

| 块组 | 边数 | 电导 µS | 占 T |
|---|---:|---:|---:|
| 池内（6 个腿池对角） | 80,918 | 46.0475711140 | 12.091054 % |
| 前→后（LF,RF → LH,RH） | 247 | 0.098043 | 0.0257 % |
| 后→前（LH,RH → LF,RF） | 464 | 0.307725 | 0.0808 % |
| 左→右（LF,LH,LM → RF,RH,RM） | 1,779 | 0.899950 | 0.2363 % |
| 右→左（RF,RH,RM → LF,LH,LM） | 1,959 | 1.062970 | 0.2791 % |
| 腿池 → MN 桶 | 24,526 | 5.540311 | 1.4548 % |
| shared → MN 桶 | 5,488 | 0.919689 | 0.2415 % |
| nonmotor → MN 桶 | **0** | **0.000000** | 0 % |
| shared → 腿池 | 15,679 | 8.825721 | 2.3174 % |
| 腿池 → shared | 8,979 | 3.625603 | 0.9520 % |
| nonmotor → 腿池 | 39,114 | 26.487461 | 6.9550 % |
| 腿池 → nonmotor | 72,709 | 64.380423 | 16.9048 % |
| shared 内部 | 2,771 | 1.337822 | 0.3513 % |
| **nonmotor 内部** | **150,119** | **202.316505** | **53.1238 %** |
| MN ↔ MN | **0** | **0.000000** | 0 % |

6 腿两两交叉（`both` 受体，腿池之间）：腿池↔腿池全部 88,652 条边 / 50.1068188303 µS（13.156921 % of T），
其中对角（同腿池内）80,918 条 / 46.0475711140 µS，**非对角只有 7,734 条 / 4.0592477163 µS（1.065867 % of T）**。
非对角按预登记分组：

| 组 | 腿对 | 边数（双向） | 电导 µS（双向） | 占 T |
|---|---|---:|---:|---:|
| 同侧 ipsilateral | LF-LM, LM-LH, LF-LH, RF-RM, RM-RH, RF-RH | 3,996 | 2.0963276325 | 0.550448 % |
| 对侧同行 | LF-RF, LM-RM, LH-RH | 2,668 | 1.2165645270 | 0.319442 % |
| 对侧跨行 | LF-RM, LF-RH, LH-RM, LH-RF, LM-RF, LM-RH | 1,070 | 0.7463555567 | 0.195976 % |
| **其中三足搭档** | LF-RH, RF-LH | **266** | **0.1348225758** | **0.035401 %** |

`_We` / `_Wi` 分开的 6×6 有向矩阵在 `BLOCKS.json -> leg_pair_matrices_6x6`（例如 `_We` LF→LF 5,171 条 / 2.2663 µS，
`_Wi` LF→LF 5,521 条 / 2.9083 µS；`_We` LF→RH 17 条 / 0.0042750 µS，`_Wi` LF→RH 17 条 / 0.0064832 µS）。

每块的单边电导分布（示例，完整在 `BLOCKS.json -> exc/inh -> blocks`）：

| 块 | 边 | 中位 µS | q05 | q95 | 最大 |
|---|---:|---:|---:|---:|---:|
| `_We` LF→LF | 5,171 | 0.00020802377414 | 2.2091310751e-05 | 0.00157909695511 | 0.01478260869565 |
| `_We` shared→LH | 1,824 | 0.00031621047369 | 3.6060293873e-05 | 0.00203786129837 | 0.01151515151515 |
| `_Wi` LF→RH | 17 | 0.00029268292683 | 9.5197132616e-05 | 0.00088864111498 | 0.00102857142857 |

**这些就是任何后续重标定必须使用的真实电导单位**；不存在"任意增益"。

### 3.3 前运动驱动矩阵（9 源桶 × 6 目标 MN 腿，`BLOCKS.json -> premotor_drive_to_mn_legs`）

边数（`both`）：腿专属池**只**驱动同腿 MN（例如 `pool_lf → LF MN` 1,815 条，其余 5 列全 0），
这是桶定义的直接推论；`shared` 池驱动六腿全部（LF 353 / LH 1,338 / LM 999 / RF 475 / RH 1,113 / RM 1,210，合计 5,488）；
`nonmotor` 与 `mn` 两行全 0。

电导（µS）：`pool_lf → LF MN` 0.7361、`pool_lh → LH` 0.9320、`pool_lm → LM` 1.0651、
`pool_rf → RF` 0.7468、`pool_rh → RH` 1.0189、`pool_rm → RM` 1.0414；
`shared` 对各腿 0.1239 / 0.2480 / 0.1349 / 0.0932 / 0.1611 / 0.1586。
每条腿的 MN 入总严格等于 `0.02 × 该腿 MN 数`：LF 0.86、LH 1.18、LM 1.20、RF 0.84、RH 1.18、RM 1.20 µS（合计 **6.46 µS**），
即**同腿专属前运动池供 78.99 %–88.90 %，`shared` 池供 11.10 %–21.01 %**（逐腿：LF 85.59 %、LH 78.99 %、LM 88.76 %、RF 88.90 %、RH 86.35 %、RM 86.78 %）。

### 3.4 双侧对称性（精确量化，`SYMMETRY.json`）

镜像映射 `lf↔rf, lh↔rh, lm↔rm`，`shared/nonmotor/mn` 自映射。

**细胞计数**（主划分）：

| 镜像对 | a | b | \|a−b\| | (a−b)/(a+b) | 5 % 带 |
|---|---:|---:|---:|---:|---|
| `pool_lf` vs `pool_rf` | 548 | 639 | 91 | **−0.07666385846672283** | **检出不对称** |
| `pool_lh` vs `pool_rh` | 771 | 776 | 5 | −0.003232062055591467 | 带内 |
| `pool_lm` vs `pool_rm` | 756 | 796 | 40 | −0.02577319587628866 | 带内 |
| MN 数 lf vs rf | 43 | 42 | 1 | +0.011764705882352941 | 带内 |
| MN 数 lh vs rh / lm vs rm | 59 / 60 | 59 / 60 | 0 / 0 | 0 / 0 | 带内 |

补充划分（`D ∪ H`）里三条镜像对**全部检出不对称**：−0.13741496598639455、−0.11507936507936507、+0.10410958904109589。

**块电导镜像不对称 `asym_index = Σ|G−G'| / Σ(G+G')`**（两种规则都报，见 §6 偏离 1）：

| 矩阵 | 预登记规则 | **修正规则（主判据）** | 修正规则、仅 6 个腿池 | 5 % 带（修正、仅腿池） |
|---|---:|---:|---:|---|
| `_We` | 0.2408103999248847 | **0.023604915746** | **0.04010018871843802** | 带内 |
| `_Wi` | 0.12874122498797627 | **0.048075720690** | **0.09301764887819959** | **检出不对称** |
| `both` | 0.16499904459721387 | **0.038152191481** | **0.06610872716171905** | **检出不对称** |

绝对量：修正规则下 `Σ|G−G'|` = 5.987165 µS（`_We`）、24.424360 µS（`_Wi`）、29.059761 µS（both），
分母 `Σ(G+G')` = 253.640613 / 508.039387 / 761.680000 µS；
仅腿池：1.754294 / 5.252321 / 6.624996 µS，分母 43.747779 / 56.465859 / 100.213638 µS。

**最大的单块不对称**（修正规则，both，逐块 `|G−G'|`）：

| 块 | 与镜像块 | 差 µS | 相对差 |
|---|---|---:|---:|
| `rh → nonmotor` vs `lh → nonmotor` | 14.822072 vs 10.258935 | **4.563137** | +18.19 % |
| `rm → nonmotor` vs `lm → nonmotor` | 12.126356 vs 9.076525 | **3.049831** | +14.38 % |
| `rm → rm` vs `lm → lm` | 8.884195 vs 7.895709 | 0.988486 | +5.89 % |
| `nonmotor → rh` vs `nonmotor → lh` | 4.663666 vs 3.764189 | 0.899477 | +10.67 % |
| `nonmotor → rf` vs `nonmotor → lf` | 4.764702 vs 3.918653 | 0.846049 | +9.74 % |
| `rh → rh` vs `lh → lh` | 8.672336 vs 9.506278 | 0.833941 | −4.59 % |

**方向不对称**（同一腿对内部，`G(A→B) − G(B→A)`，both）：

| 腿对 | A→B µS | B→A µS | 差 µS | 相对差 |
|---|---:|---:|---:|---:|
| LF–RF | 0.169530 | 0.196026 | −0.026496 | −7.25 % |
| LM–RM | 0.163350 | 0.237384 | −0.074033 | **−18.47 %** |
| LH–RH | 0.252493 | 0.197781 | +0.054713 | +12.15 % |
| LF–RH（三足） | 0.010758 | 0.052012 | −0.041254 | −65.72 % |
| RF–LH（三足） | 0.016768 | 0.055285 | +0.038517 | +53.46 % |

结论（只做结构陈述，不外推功能）：**冻结严格子图在腿块层面并非严格双侧对称**——
`_Wi` 与仅腿池口径下超过 5 % 带；逐块最大差达 4.56 µS；同侧镜像腿对 `lh-rm` 与 `rh-lm` 的双向电导差 0.096910 µS（相对 **29.00 %**）。
但上游动力学观测到的却是**双侧对称的前/后模态**。这两件事不矛盾也不互相证明：
结构上的不对称量级（腿池之间全部非对角只有 4.059 µS，占 T 的 1.07 %）远小于未被腿标记的主体
（nonmotor 内部 202.317 µS，占 T 的 53.12 %），而本审计没有任何动力学，不能判断这部分不对称在
20,000 细胞闭环里会不会被放大或抹平。**这是本轮最重要的边界，不得跨过。**

### 3.5 哪些既有块在原理上能打破主导对称模态（`COUPLING.json`）

对五对步态相关腿对 `(LF,RF)`、`(LM,RM)`、`(LH,RH)`（同行对侧）与 `(LF,RH)`、`(RF,LH)`（三足搭档）逐层给精确计数：

| 腿对 | 类别 | L0 直接边 A→B / B→A（both） | L1 同时投射两腿的细胞 | L2 两跳路径 A→B / B→A | 直接电导双向 µS | 判定 |
|---|---|---:|---:|---:|---:|---|
| LF–RF | 对侧同行 | 498 / 569 | 62 | 4,498 / 3,837 | 0.365556 | L0 直接互连存在 |
| LM–RM | 对侧同行 | 299 / 407 | 87 | 8,809 / 9,910 | 0.400734 | L0 直接互连存在 |
| LH–RH | 对侧同行 | 498 / 397 | 110 | 10,154 / 10,127 | 0.450274 | L0 直接互连存在 |
| **LF–RH** | **三足搭档** | 34 / 84 | **27** | 3,704 / 1,248 | **0.062770** | L0 直接互连存在（弱） |
| **RF–LH** | **三足搭档** | 43 / 105 | **32** | 5,278 / 2,383 | **0.072053** | L0 直接互连存在（弱） |

- 全部 15 条无序腿对（不只 5 条）都有双向直接边、共享桥细胞与两跳路径；**没有一条腿对在 1–2 跳口径上为零**。
  因此本轮的结论**不是**"三足耦合缺失"。预登记里准备好的 `ABSENT ... MISSING_DATA` 分支在本数据上**没有被触发**，
  0 条腿对落入该分支。
- 三足搭档的真实缺口是**强度与数量级**：LF–RH + RF–LH 合计 266 条边 / 0.1348225758151581 µS，
  是同行对侧 2,668 条 / 1.2165645270 µS 的 **9.97 %（边）与 11.08 %（电导）**，占模块总电导 **0.0354 %**。
  但两跳通路数量并不稀少：LF–RH 有 3,704 + 1,248 = 4,952 条两跳路径（其中 3,613 + 1,026 = 4,639 条经 `pool_shared` 中间细胞），
  与 LF–RF 的 4,498 + 3,837 同量级。因此三足搭档的结构缺口是**直接块的强度**，不是通路数量。
- 方向性一致偏向"H → F"：`rf→lh` 105 条 vs `lh→rf` 43 条，`rh→lf` 84 条 vs `lf→rh` 34 条。
- L1 桥细胞规模：`cells_targeting_both_legs` 在 27–110 之间；它们对 MN 的总出电导 0.17866183355571555–0.35581348131074897 µS（按腿对）。
- 预登记 §4 的**定义推论必须显式写明**：`pool_LF` 细胞的直接 MN 目标全部属 LF，所以
  "`pool_LF → MN_RH` 的直接边为 0"是**定义必然**，不是数据缺失；真正的互连只能经由
  `pool_LF → pool_RH`（前运动到前运动，存在）、`shared` 桥（存在）或两跳（存在）。

### 3.6 结构发现：323 个运动细胞是纯汇点

- **323 / 323** 个 MN 在严格子图中出边数 = 0（`mn` 行全 0，`mn_to_mn` 块 0 条 / 0.000000 µS）。
- MN 入边 4–265 条，入总 0.02 µS × 323 = **6.46 µS**（占 T 的 1.696 %）。
- 直接后果：化学突触上没有 MN→网络反馈，也没有 MN→MN 耦合；任何只作用于
  MN 自身（如 `adaptation-screen` 的适应增量缩放）的改动都**不可能**改变腿间结构耦合。
  这与 `.../adaptation-screen/REPORT.md` 的阴性结果**在结构上一致**；它是解释，不是那一轮结论的证明，
  也不代表其他参数化一定不行。

## 4. 面向下一轮的有界重标定提案（`PROPOSED_SCALING_PARAMS.json`，**仅设计，未运行**）

- 规则：`w'[e] = s[class(e)]·w[e]`，只在**已存在**的边上乘标量；不新增、不删除、不翻转符号、不改递质类别。
- **自由标量 3 个**（预登记上限 5）：
  `s_contra_same_row`（LF–RF, LM–RM, LH–RH）、`s_front_hind`（LF–LH, RF–RH）、`s_tripod_diagonal`（LF–RH, RF–LH）。
- **边界**：每个 `s ∈ [0.25, 4.0]`（含端点）；建议 5 档网格 `{0.25, 0.5, 1, 2, 4}`，共 125 组。
- **类别划分互斥穷尽**（`Σ = 435,126` 条，独立核验通过）：

| 类别 | 标量 | 边 | 冻结电导 µS |
|---|---|---:|---:|
| `C_contra_same_row` | `s_contra_same_row` | 2,668 | 1.2165645270131350 |
| `C_front_hind` | `s_front_hind` | 445 | 0.27094469578280533 |
| `C_tripod_diagonal` | `s_tripod_diagonal` | 266 | 0.1348225758151581 |
| `C_fixed_everything_else` | 1.0（固定） | 431,747 | 379.21766820138896 |

- **`.02 µS` 约定（必须择一）**：
  - `V1 row_renormalized`：施加后把每个目标神经元的入边整体归一到**恰好 0.02 µS**（逐神经元入总与冻结一致）；
  - `V2 bounded_unrenormalized`：不归一，但每个目标神经元入总必须在 `[0.5, 2.0] × 0.02 µS`，越界按行投影回边界。
  代码里默认推荐 V1，因为它**精确保守**冻结的电导约定，只改变类间相对权重。
- **诚实上限**（同一份 JSON，`scalar_ceiling_if_applied_exactly`）：
  `s_tripod_diagonal = 4.0` 时三足类从 0.1348225758151581 µS 变为 0.5392903032606324 µS，
  增量 **+0.4044677274454743 µS = T 的 0.1062 %**，仍只有同行对侧类冻结值的 **44.33 %**。
  `s_contra_same_row = 4.0` 时变为 4.8662581080525400 µS（+3.649693581039405 µS = T 的 0.958 %）。
- **作用面**（`per_neuron_class_share`）：`C_tripod_diagonal` 只影响 **114** 个靶神经元的入电导，
  其中只有 **3** 个靶的这一类占其入总 > 50 %（最大占比 0.8495934959349591）；
  `C_contra_same_row` 影响 721 个靶，28 个超 50 %（最大 0.9236947791164658）；
  `C_front_hind` 影响 306 个靶，1 个超 50 %。
- 禁止：新增/删除边、改递质、翻符号、外置振荡器/相位钟、改验收判据、把身体或腿状态读进前运动块。
- 下一轮若真做：**新时间戳目录 + 新预登记 + 先开环神经筛查再物理**，且标量组必须**事先指定**
  而不是事后搜索。本轮没有任何数字证实它有效；不得把"能重标定"写成"能产生步态"。

## 5. 独立核验（`verify_leg_pools.py` → `INDEPENDENT_VERIFICATION.json`）

`verify_leg_pools.py` 不 import `leg_pools.py`，用完全不同的实现重算：池用纯 Python dict/set 逐边构建
（无 numpy 位掩码），块电导用源突触计数 + 递质列重算（不读 CSR 切片），两跳路径用中间细胞乘积枚举。

**74 / 74 项检查全部 PASS**，覆盖：

1. 7 个输入文件 sha256 逐一相符；
2. 435,126 条边；ACh 207,166 / 抑制 227,960；`0.02·syn/incoming` 与冻结矩阵最大差 0；`Σ=380.84 µS`；
   19,042 行入总 0.02 µS；
3. 池划分：每个神经元**恰好**一个桶、9 桶合计 20,000；主划分与补充划分逐桶相符；
   `POOLS_AND_REACH.npz` 的 `bucket_id` / `bucket_id_union` / `leg_mask_direct` / `leg_mask_2hop` 逐神经元与独立重建相符；
4. **边守恒**：无重复有序对、无自环；81 个块的边数逐个精确相等、电导在 1e-12 相对容差内相等；
   两受体拆分后的 81×4 个量与递质列重算相符；块边数合计 435,126、电导合计 380.84 µS；
5. 聚合块（L→R、R→L、F→H、H→F、腿池→MN、shared→MN、nonmotor→MN、nonmotor 内部、MN↔MN）全部复现；
6. 三种镜像指数（预登记规则、修正规则、仅腿池修正规则）、逐块差、15 条腿对方向不对称全部复现；
   并且**显式验证了预登记规则在 LF↔RF / LM↔RM / LH↔RH 上恒为 0 的结构性失明**（偏离 1 的证据）；
7. 五对步态腿对的 L0/L1/L2 计数与全部 15 条腿对的桥细胞数、直接边数复现；
8. 提案的三类边数/电导、`Σ = 435,126`、`Σ = 380.84 µS`、3 个标量、边界、上限算术复现；
9. 323 个 MN 出边为 0；`RESULT.json` 中的预登记/代码/输入哈希与磁盘一致；记录的仿真步全为 0。

浮点说明：整数逐位相等；块电导因求和顺序不同，在 **1e-12 相对容差**内相等（预登记外的说明性补充，
本报告所有 µS 数字均直接取自 JSON 的原始双精度值，未四舍五入后再回写）。

## 6. 偏离（`DEVIATIONS.md`，预登记文件未改）

1. **镜像规则定义缺口**：预登记规则 `G'[s][d] = G[mir(d)][mir(s)]` 把"换侧"与"反向"复合，
   对自镜像块（LF→RF、LM→RM、LH→RH 及其反向）与自身比较，恒为 0，**在本轮最关心的三条同行对侧耦合上结构性失明**。
   处理：旧规则结果保留并显式命名（0.2408 / 0.1287 / 0.1650），新增修正规则
   `G'[s][d] = G[mir(s)][mir(d)]` 作为主判据，并给出腿对级 `direction_asym_uS` 与 `self_mirror_pair` 标记。
   两个规则都报告，不存在"改规则把结论改好"的空间；`ASYM_TOL=0.05` 与分档未变。
2. **交接文本的数字不可复现**：`allowed_sensory_local` 实测 **5,103**（不是 556）；
   `.../20261107T185910Z/dynamics/PHYSICAL_QUALIFIED_PORTS.npz` 的 `local_indices` 长度为 **559**，
   其中落在 4 条歧义记录内的为 0，故 559−4 也得不到 556。本轮不用该数字做判断，只如实记录，不推测原因、不改输入。
3. **路径长度上限**：长度 ≥ 3 的路径本轮未做任何可达性判定（预登记原样执行）。

## 7. 明确界限（必须随任何引用一起传播）

- 这是**未标定的严格过滤 20,000 细胞子图上的离线结构审计**。它**不是果蝇解剖**：
  严格纯 verified 递质规则只保留 435,126/705,068 条候选边，缺失的边在本审计里就是缺失，
  不等于生物上没有；`pool_*` 名字只是**本审计按边定义的桶名**，不是细胞类型、不是身份声明。
- **不是功能、不是生理、不是行为证据**。本轮 0 神经步、0 物理步，没有产生任何活动或运动。
- 纯递质标签 ≠ 逐突触受体/符号/电生理标定；`_We`/`_Wi` 的兴奋/抑制符号是沿用的工程假设。
- `nonmotor` 桶占 74.77 %，其中有 5,112 个细胞两跳内可达 MN；"leg-specific" 在直接边口径上是少数派，
  这不代表它们在体内没有腿特异作用。
- 结构不对称存在 ≠ 功能不对称；同理，三足耦合"存在但弱" ≠ "无法产生步态"，也**不**证明能产生。
  本轮**没有**任何"能走"的结论。
- 不改任何验收判据，不推翻 `intrinsic/`、`adaptation-screen/` 的阴性历史，不改写 53 个物理候选的失败记录。
- 绝对力值与物理单位：本轮不涉及，**NOT_ESTABLISHED**。

## 8. 记账与下一最小动作

- 本目录文件：`PREREGISTRATION.md`、`PREREGISTRATION.json`、`DEVIATIONS.md`、`leg_pools.py`、`POOLS.json`、
  `POOLS_AND_REACH.npz`、`BLOCKS.json`、`SYMMETRY.json`、`COUPLING.json`、`PROPOSED_SCALING_PARAMS.json`、
  `RESULT.json`、`verify_leg_pools.py`、`INDEPENDENT_VERIFICATION.json`、`REPORT.md`，
  桌面另加 `MIRROR_MANIFEST.json`（不参与自身镜像集）。
- **镜像核验**：源码目录 14 个文件全部逐字节复制到桌面目录并用 sha256 双向复核（14/14 相同）；
  桌面目录额外有一个 `MIRROR_MANIFEST.json`（只放桌面、不参与自身镜像集），其中逐个文件记录
  `bytes` 与 source/desktop 两侧 sha256，可独立复核"桌面 = 源码"。完整重跑一次
  （`leg_pools.py` + `verify_leg_pools.py`）后 14 个文件的 sha256 **完全不变**，即本轮产物可确定性复现。
- 运行命令（两处逐字节等价复跑）：
  ```
  cd /tmp && PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python -B \
    /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-descending-learning/20261107T203049Z/leg-pools/leg_pools.py
  cd /tmp && PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python -B \
    /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-descending-learning/20261107T203049Z/leg-pools/verify_leg_pools.py
  ```
- 未执行项：任何神经动力学、任何物理、任何学习、任何边改动、任何渲染；长度 ≥ 3 的路径可达性。
- **下一最小动作**（需新预登记，不在本轮）：在冻结 core 上实现 `PROPOSED_SCALING_PARAMS.json` 的 V1
  （行归一化）版本，只跑**开环神经筛查**（同一 100 箱感觉回放、零下行、2,000 ms），
  用 `intrinsic/` 的同一套腿间相关/交替分数口径，检查 `s_tripod_diagonal` 与 `s_contra_same_row`
  是否能把 LF–RH / RF–LH 的相关从 .036 / .082 推向反相；无论阳性阴性都不得升级为行为结论，
  且物理试验必须在其后另立预登记。
