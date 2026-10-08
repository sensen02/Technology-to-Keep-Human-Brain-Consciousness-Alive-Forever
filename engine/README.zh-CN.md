# 引擎（engine/）架构与使用

> 本文件描述**引擎本身**。它不产生生物学结论；它负责让项目的既有规则
> **无法被遗忘**、让错误**无法静默通过**。

## 1. 为什么需要它

这个项目已经踩过的坑，全部是"静默失败"：

| 事故 | 后果 | 现在由谁挡住 |
|---|---|---|
| `m²→µm²` 用 1e6 而不是 1e12 | 扩散长度差 1000 倍，趋化**静默失效** | `units.py` |
| 趋化源项没打开，场恒为零 | 两组参数**逐位相同**，看起来像"否定结果"其实是 bug | `params.py` + `checks.py` |
| `divide()` 未继承 `λ_V` | 子细胞无体积约束，组织生长冻结 | 层契约 + 检查 |
| 坐标还原写错，`clamp` 冻住全部 400 细胞 | 整个实验无效 | 不变量检查 |
| "缺的是 ECM"——**假设本身是错的** | 浪费一轮工作 | 生物档案 + 出处强制 |
| 把毛细血管 Krogh 模型用于果蝇 | 类别错误 | `profile.py` 守卫 |
| 阈值写死在 Python 里 | 示例值被当成结论 | `rules.py` |

## 2. 分层架构

`ORGANISM_SCALE_PLAN.md` 早已写下架构与硬约束——**"耦合规则必须是单向可追踪的"**。
引擎把这条从"约定"变成"机制"。

```
L0 环境场        （温度、氧分压、pH、激素血浆浓度）
L1 血管与输运     transport.py      →  已接入（演示已验证）
L2 细胞状态       cellstate.py      →  已接入（演示已验证）
L3 生长与分裂     growth.py         →  待接入
L4 组织力学       cpm.py / mechanics.py / merks.py  →  待接入
L5 胞内信号       model.py / cellspace.py / vasculogenesis.py → 待接入
L6 激素作用       hormone.py / ligand.py → 待接入
```

**单向性是强制的，不是建议**：`Pipeline` 只把某层**自己声明**的端口喂给它，
其余一律看不到；若某层要求一个"后面那层"才提供的端口，直接
`FeedforwardViolation`。

### 每层自带时间步

钙信号以秒计、组织生长以小时计，差 3–4 个数量级。每层声明自己的 `target_dt`，
管线在外层步内做子步（显式一阶算子分裂——这是**数值陈述**，不是生物学近似）。

## 3. 四道守卫

| 守卫 | 机制 | 已由否定测试证明会触发 |
|---|---|---|
| **单位** | `Quantity` 带量纲运算，不兼容即抛错 | ✅ µm+s、µm→s、未知单位 |
| **出处** | `measured` 必须带引用；`derived` 必须列父参数；值必须在自述范围内 | ✅ 无引用、无父参数、超范围 |
| **物种/阶段** | 跨物种实测参数、跨阶段锚点混用会被拦下 | ✅ 果蝇模型里用小鼠参数 |
| **生物档案** | 无血管生物拒绝血管层；阶段警示自动注入报告 | ✅ 果蝇/水螅拒绝血管层 |

**注意**：生物档案守卫最初写成"读已接线的端口表"，导致**未接线时静默放行**——
这是"失败开放"的守卫，比没有守卫更糟。已被自检抓出并修正为直接读层声明。

## 4. 规则即数据（PhysiCell 的工程启发）

规则不再写死在代码里，而是可校验的数据：

```python
rs = RuleSet("notum")
rs.add("cell_o2 < 5 mmHg", "cellstate.death_rate", "set", "0.05 1/h",
       basis="illustrative", note="低氧死亡阈值，由我们选定")
rs.compile(pipeline.ports())          # 编译期校验端口与单位
rs.annotate()                         # 自动生成文档
```

编译期会拒绝：
- 引用了**没有任何层提供**的端口（该规则永远不会触发——正是"静默失效"）；
- 把 mmHg 与 s 相比；
- 声称 `measured` 却没有引用。

## 5. 检查框架与它的边界

六类检查：`analytic / conservation / determinism / convergence / invariant / order`。

**通过检查只说明「代码在数值上做到了它声称的事」，不说明模型在生物学上正确。**
这句话被写进 `summary()` 与生成的 markdown 里，下游读者无法误读。

## 6. 运行记录

每次运行输出自描述记录：环境（python/numpy/numba/CPU）、git commit、
生物档案与**强制注入的警示**、完整参数快照（含出处）、接线报告、全部检查结果、
输出清单、内容哈希。

`claim_boundary` 字段固定写入：

- 这是模拟，不是永生实验；
- 唯一可用的定量实验对照是一条对照伤口钙波半径轨迹；
- 未标注 `measured` 的参数不是实测值；
- 本模拟不验证、也不能验证主观体验。

## 7. 验证状态

`run_engine_selftest.py`：**44 项测试全部通过，其中 36 项是否定测试**
（专门验证守卫会触发、约束不可绕过）。见 `outputs/ENGINE_SELFTEST.zh-CN.md`。

`run_engine_demo.py`：引擎承载**真实模块**的端到端演示
（`transport.py` 的保守 Laplacian + `cellstate.py` 的逐细胞状态机），
5/5 检查通过。见 `outputs/engine_demo_report.zh-CN.md`。

## 7b. 已接入的层（本轮补齐）

| 层 | 包装的模块 | 端口 | 状态 |
|---|---|---|---|
| `TrachealGasLayer` | 新增，**昆虫专属**气管供气 | `spiracle_o2` → `cell_o2` | ✅ 与解析稳态圆柱解**偏差 0** |
| `CellStateLayer` | `cellstate.py` | `cell_o2` → `atp`/`integrity`/`viability` | ✅ |
| `HormoneLayer` | `hormone.py` | `hormone_nM` → `bound_fraction`/`response` | ✅ |
| `GrowthLayer` | `growth.py` | `substrate` → `biomass` | ⛔ **拒绝运行**：`growth.run()` 是整段运行入口，无法增量推进，包装器宁可报错不假装推进 |
| `MechanicsLayer` | `cpm.py` | `cell_o2` → 几何 | ⛔ **拒绝运行**：CPM 的"MCS→秒"映射本项目未标定，给端口就等于说谎 |
| `neural` | `engine/neural.py` | 见下 | ✅ 12/12 |
| `sensorimotor` | 真实 FlyWire 标注 | 感觉模态 → 运动输出 | ✅ 一致性校验通过 |

**"拒绝运行"也是交付内容**：一个只会假装推进的层比没有更糟。

## 7c. 感觉/运动接口（为虚拟环境准备）

端口**来自数据**，不是我们发明的分类：

| 感觉输入 | 神经元数 | | 运动输出 | 神经元数 |
|---|---|---|---|---|
| `visual` 视觉 | 11,010 | | `descending` 下行 | 1,290 |
| `mechanosensory` 触觉 | 2,633 | | `motor` 运动 | 109 |
| `olfactory` 嗅觉 | 2,279 | | `endocrine` 内分泌 | 80 |
| `gustatory` 味觉 | 334 | | | |
| `hygrosensory` 湿度 / `thermosensory` 温度 | 74 / 29 | | | |

闭环已验证：**只注入视觉通道**，真实连接组把信号传到了运动输出
（12 mV → 下行 53.5 Hz；14 mV → 126 Hz），且 0–10 mV 全静默。

## 7d. 完整中枢神经系统与虚拟环境闭环（本轮）

**BANC**（雌成蝇脑+神经索，一体）：**153,962 神经元 / 3,037,361 连接 /
23,556,214 突触**。选它而不是 FAFB+MANC，因为后两者是**不同标本**，拼接需要
解决 FlyWire 的 VNC Matching Challenge；BANC 一体同框，无需跨标本配对。

BANC 还**自带真实受体神经元**（它的 class 标注）：

| 模态 | 受体神经元 | 转导模型 |
|---|---|---|
| `photoreceptor_neuron` 光感受器 | 1,838 | Naka-Rushton + 适应 |
| `bristle`/`chordotonal`/`campaniform` 机械感觉 | 8,616 | 两态 Boltzmann 通道 |
| `olfactory_receptor_neuron` 嗅觉受体 | 2,967 | Langmuir |
| `taste_*` 味觉 | 1,430 | Langmuir |
| `hygrosensory`/`thermosensory` | 90 / 31 | 饱和 S 形 |

闭环：**物理刺激（光强/应变/浓度）→ 受体转导 → 真实受体神经元 → 真实 BANC
连接组 → 运动输出**。没有任何直接往感觉神经元注电流的捷径。
受体 10/10 通过解析验证（光稳态 2.7e-15、Boltzmann 2.2e-16、与 hormone.py
的 Langmuir **逐位一致**）。

### 但闭环的真实结果不理想，且已定位

| 刺激 | 运动输出 |
|---|---|
| 暗/静息、光（0.85、3.0）、气味、温干 | **0 Hz** |
| 触碰、光+触碰 | **245 Hz** |

增益扫描：光 14→20 mV 全 0，30 mV 直接跳到 278 Hz，90 mV 到 314 Hz。
**只有「关」和「发作」两态，没有分级的感觉响应。**
245–320 Hz 比真实果蝇运动神经元高 **1–2 个数量级**。

原因（已定位，非调参问题）：网络**无抑制稳定**、LIF **无适应/无突触抑制**、
权重标度是 illustrative。**这不是靠调一个数能修的。**

## 7e. 物理化：从 LIF 点到电缆神经元（本轮）

**之前不是物理的。** LIF 点神经元没有形态、没有树突、没有离子通道、没有
reversal potential，突触只是带符号的标量——完全符合「参数固定的虚拟对象」。

现在 `engine/cable.py` 实现了真正的电缆方程（空间延展的膜）：

```
Cm·A·dV/dt = -(V - E_leak)/R_m·A + Σ_j (V_j - V)/R_axial(i,j) + I_syn + I_inject
```

**参数是实测/拟合的，不是编的**：Moreno-Sanchez et al. 2024（PMC11071487）
对真实 FlyWire 下行神经元 DNp01/DNp03 做全细胞电流钳拟合得到
`Cm=0.7 µF/cm², g_leak=4.35e-4 S/cm², Ra=212 Ω·cm, E_rev=-66.63 mV`。
他们的流程正是我们要走的：EM 网格 → SWC 骨架 → 带坐标/半径/父子层级的多室模型
→ 突触映射到真实位置。

**采用了 NEURON 9.0.2**（许可已核实：**BSD 3-Clause**）。但关键工程判断：
**NEURON 不适合 15 万个神经元**，它是为少量精细细胞设计的。
所以按 `ORGANISM_SCALE_PLAN.md` 早就写好的原则分层：**少量细细胞用电缆模型，
大量粗细胞用（下一步要升级成电导型的）点神经元**。

### 验证：与电缆理论 + 与 NEURON 双重对拍

| 检查 | 结果 |
|---|---|
| 膜时间常数 vs Rm·Cm | 1.5968 vs 1.6092 ms（**0.77%**） |
| 空间常数 vs √(Rm·d/4Ra) | 164.74 vs 164.65 µm（**0.06%**） |
| 封闭端无通量 | 相对梯度 3e-5 ✓ |
| 分流抑制 | 444.49 → 137.94 mV/nA ✓ |
| **与 NEURON 逐位对拍** | -289.8459 vs -289.8459 mV，**差 0.0000** |

### 这一轮抓出的三个真 bug（值得记下来）

1. **膜电导算出来却没放进矩阵对角**：模型根本没有膜，只有电容和轴向电阻。
2. **漏电流驱动项 `g_mem·E_leak` 缺失**：膜电位朝 0 mV 松弛，而不是朝 E_rev。
3. **电流单位错 1000 倍**：`G` 用 µS、`V` 用 mV 时 RHS 必须是 **nA**
   （µS×mV=nA），我却做了 nA→µA 转换，结果解出来是伏特当毫伏用。

**第 3 条只有 NEURON 能抓出来**——因为我自己的"解析对照"和仿真用的是同一套
（错的）参数，两边自洽地一起错，τ 和 λ 全都"通过"。
**独立实现对拍不是可选项，是唯一能抓到这类错误的手段。** 这一条已经写进
引擎文档。

## 8. 引擎现在**还不能**做什么（诚实清单）

1. **有神经层了，但没有"会工作的脑"**。真连接组 + LIF 只有"死寂"与"饱和"
   两个状态（实测：无驱动处处静默；激活后 110–145 Hz 发作样）。
   **原因是缺身体与环境**——没有感觉输入，循环网络无物可传。
   接口已备好（见 7c），环境尚未实现。
2. **只有 2D**。所有已接入模块都是 2D。
3. **缺 3D 邻域、自适应网格、GPU**。
4. **没有参数辨识**（Sobol/ABC/贝叶斯）与不确定量化。
5. **没有真实数据对拍管线**（全项目只有 1 条钙波轨迹）。
6. 生物档案目前只有 3 个（果蝇 notum / 水螅 / 泛哺乳动物），
   且除果蝇外几乎没有核实过的参数。
7. **不能验证意识或主观体验**——这是概念边界，不是工程量问题。

## 9. 怎么加一层

```python
class MyLayer(Layer):
    name = "my_layer"
    requires = {"cell_o2": "mmHg"}        # 只能读这些
    provides = {"my_out": "uM"}

    def initialize(self, inputs): ...
    def step(self, dt, inputs):           # dt 是 Quantity
        return {"my_out": ...}

pipeline.add(MyLayer())
pipeline.wire()                           # 端口、单位、单向性在这里校验
profile.validate_pipeline(pipeline)       # 生物档案守卫
```

新参数必须先在 `ParamRegistry` 里声明并标注出处——**不声明就不能用**。
