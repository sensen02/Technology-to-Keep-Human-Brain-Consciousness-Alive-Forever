# 全面复核与差距分析（忽略此前结论，按磁盘实物重做）

生成时间：2026-10-03T21:18-07:00
工作区：`/run/media/sensen/Data2/cell_wound_prototype`
性质：**手工研究原型**。本文的每个数字都标了出处文件；凡我自己没验证的，写「未验证」。
方法：先盘点磁盘实物，再派两个独立代理**只读**复核此前的声称，最后由我自己复算关键几条。两个代理都遵守「不修改任何文件」；我自己动手的只有清理与桌面同步——逐项记录在 `/run/media/sensen/Data2/cell_wound_prototype/CLEANUP_MANIFEST_2026-10-03.md`。

---

## 1. 磁盘实况（先看东西有多少）

| 项 | 实测 |
|---|---|
| 总占用 | **17.8 GiB**（19,142,708,463 字节），37,526 个非虚拟环境文件 |
| 最大单件 | `/run/media/sensen/Data2/cell_wound_prototype/data/flywire/fafb_skeletons_swc.zip` **12.9 GiB**（139,273 个 .swc） |
| 目录构成 | data 13.1 G(73.2%)、vendor 3.3 G(18.3%)、venv_body 570 M、venv 519 M、outputs 419 M、viewer 15 M、engine 1.6 M |
| Python 代码 | 根目录 153 个 .py（66,886 行）+ `engine/` 33,280 行 ≈ **10 万行** |
| 脚本形态 | 90 个 `run_*.py`、27 个 `*selftest*.py`、24 个 `launch_*.sh`、9 个 `deliver_*`、8 个一次性探针 |
| 可视化出口 | **389 张 PNG、291 张 SVG、8 个 GIF、2 个 MP4、11 份 HTML**；203 个脚本里有 `savefig` |
| 年龄 | 最早文件 2026-10-01，最晚 2026-10-03 → 全部工作集中在 **3 天**内 |

一个直接后果：**389 张 PNG 里只有极少数进了身体视图**。详见第 5 节。

---

## 2. 独立复核结果（两个代理，只读）

### 2.1 伤口钙 / 力学 / 胞内空间（P1–P6）

| 声称 | 证据文件 | 实测值 | 判定 |
|---|---|---|---|
| P1 钙信号与实验曲线对照 | `/run/media/sensen/Data2/cell_wound_prototype/outputs/metrics.json` | control MAE **5.32771 µm**、RMSE **5.34506 µm**、bias 5.32771、窗口内 11/11 点在 7.399 µm 内 | VERIFIED |
| P2 梯度校核 | 无存盘工件 | 复跑 `mechanics.test_1_gradient()` → **1.0312745e-09** | 数值真，**工件不存在** |
| P2 能量单调 | `/run/media/sensen/Data2/cell_wound_prototype/outputs/mechanics_cut_n24.npz` | 31 点 154.5937→96.5557，30 步无上升 | VERIFIED（但元数据写明「能量上升就拒步并减半 dt」，偏弱） |
| P2 闭合 42%/60 s | `/run/media/sensen/Data2/cell_wound_prototype/outputs/metrics_mechanics_lit.json` | **0.4194030660171568** | VERIFIED |
| P2 T1 | `/run/media/sensen/Data2/cell_wound_prototype/outputs/metrics_mechanics.json` | `{"t1":0,"t2":0,"injury":1}` — 严格 **0**，已发表 8.5e-4/(min·junction) | VERIFIED |
| P3 索引平移 / 残差 / 跨度 | `/run/media/sensen/Data2/cell_wound_prototype/outputs/metrics_mechanics.json` | 平移 **1.7616065669991545 µm**、残差 **1.481e-13**、跨度 **26.0967 µm**（可用 4、删失 3） | VERIFIED |
| P3b 形态指数 / 松弛时间 | `/run/media/sensen/Data2/cell_wound_prototype/outputs/metrics_mechanics_lit.json` | **3.7224194364**、τ **8.2707 s**（R²=0.8892；已发表 5–20 s） | VERIFIED |
| P4 二阶收敛 / D→∞ | **无存盘工件** | 复跑 → order 1.92364 / 1.96366；lumped 残差 0.503% | 数值真，**存档链断裂** |
| P5 迟滞 0/576 | `/run/media/sensen/Data2/cell_wound_prototype/outputs/metrics_cellstate.json` | `cells_returning_to_healthy=0`、`latched_not_lysed_cells=576` | VERIFIED |
| P6 迟相失配 | `/run/media/sensen/Data2/cell_wound_prototype/outputs/metrics_late.json` | 49.22 s：**78.6923 vs 51.4124 µm**（差 27.28 µm） | VERIFIED |

### 2.2 具身 / 神经 / 电极

| 声称 | 证据文件 | 实测值 | 判定 |
|---|---|---|---|
| 10/10 种子行走 | `/run/media/sensen/Data2/cell_wound_prototype/outputs/embodied_body/body_walk_report.json` | n_seeds=10、n_passed=10 | VERIFIED |
| 行走位移/速度/胸高 | 同上 | 所引三数是**种子 0**；10 种子实际胸高 **0.6937–1.1184 mm**（文档写 0.771–1.102，**低估**） | **PARTIAL** |
| 三时钟 1e-4/5e-4/5e-3 | `/run/media/sensen/Data2/cell_wound_prototype/outputs/embodied_body/loop_report.json` | 完全一致 | VERIFIED |
| 速度比 0.731±0.020、朝向 −21.39±7.59° | 同上 | 0.7306170492901051 ± 0.0196050673；−21.389255993211133 ± 7.594191049322832；**仅 3 个种子** | VERIFIED（样本小） |
| 「47/47 测试」 | `/run/media/sensen/Data2/cell_wound_prototype/outputs/embodied_body/loop_selftest.json` | **49/49** | **数字过期** |
| 52/52、36,989 禁用边、104 次拒绝 | `/run/media/sensen/Data2/cell_wound_prototype/outputs/embodied_body/plasticity_selftest.json`、`plasticity_report.json` | 全部逐字对上 | VERIFIED |
| P1 否证 55.7 Hz / CV 0.03 | 同上 | 55.68892443022791 / 0.0336764656 | VERIFIED |
| P2 成立 | 同上 | verdict=reproduced，但工件自带「该驱动下**无判别力**」 | VERIFIED（限定未随文引用） |
| P3 否证 2.4× | 同上 | 2.4451133488798176（仅 stp_homeostasis/all 两臂）；homeostasis_only 臂是 **4263.6×** | VERIFIED（限定未随文引用） |
| 氧账本残差 +2.27e-13 nmol | `/run/media/sensen/Data2/cell_wound_prototype/outputs/embodied_body/slowphys_report.json` | 逐项精确 | VERIFIED |
| Pearson r=0.8721、np.clip=0、46/46 | 同上 + `slowphys_selftest.json` | 0.8720598704825623、0 次、46/46 | VERIFIED |
| A/B/C PASS、D NOT RUN | `/run/media/sensen/Data2/cell_wound_prototype/outputs/embodied_body/behaviour_acceptance.json` | 字面一致 | VERIFIED |
| FAFB ≥5 阈值 100% vs BANC 50.3% | `/run/media/sensen/Data2/cell_wound_prototype/outputs/embodied_body/banc_fafb_reconciliation.json` | 1.0 vs 0.5032608899633596，代理 mmap 独立复算**逐位一致** | VERIFIED（BANC 在线接口那条 404 失败，结论只由缓存 npz 支撑） |
| bench 硬件 NOT_RUN、0 次测量 | `/run/media/sensen/Data2/cell_wound_prototype/outputs/workbench/bench_latest.json` | 9/9 NOT_RUN、`measurement_count=0`、`readiness_claim=false` | VERIFIED |
| 睡眠未实现 | `/run/media/sensen/Data2/cell_wound_prototype/outputs/embodied_body/slowphys_report.json` | `sleep.implemented=false`、无 SLEEP 标签 | VERIFIED |
| 记录链 77 项测试、`result_validated=true` | `/run/media/sensen/Data2/cell_wound_prototype/outputs/workbench/runs/1c8337e5854e4adaaf479724ccbd3f9b/result.json` | 我自己复核：`result_validated=true`，文件齐全 | VERIFIED |

**总体：实质数值绝大多数属实。问题不在「造假」，在「证据链与引用」。**

---

## 3. 十条具体差距（按严重度排序，全部可复核）

### G1 ★★★ 电极几何在项目里有两套，且工作台显示的那一套没有配套归属数据

| 位置 | pitch | diameter | 说明 |
|---|---|---|---|
| `/run/media/sensen/Data2/cell_wound_prototype/run_access_map.py` 第 44–45 行 | **100 µm** | **10 µm** | 写成 `FIXED input`，但 `access_map.json` 自己标 pitch 为 **ASSUMED**、diameter 为 **ENGINEERING_DEFAULT** |
| `/run/media/sensen/Data2/cell_wound_prototype/viewer/data.js` + `viewer/modules/data.js` 第 3 行 | **20 µm** | **7 µm** | 前端强制校验「必须是 7/20」，否则抛错 |
| `/run/media/sensen/Data2/cell_wound_prototype/export_integrated_viewer.py` 第 109–111 行 | **20 µm** | 7 µm | **用 100 µm 阵列的中心点，重新铺了一张 20 µm 网格** |

后果（这是全项目最严重的一处不一致）：
- `/run/media/sensen/Data2/cell_wound_prototype/outputs/embodied_body/access_map.json`（生成 09:34）里 256 个通道的坐标间距是 100 µm，**221/256 是空通道**、捕获率 44/256；
- 工作台把这 256 个通道重铺成 20 µm 间距（间距缩了 5 倍，面积缩小 25 倍）后显示；
- **每个通道「碰到哪些神经元」的归属数据仍来自 100 µm 阵列**（`primary_build.channel_neurons`）；20 µm 阵列的捕获、串扰与重叠**从未重算**。
- 该文件自 2026-10-03 09:34 未再生成；几何在 `ROUND26_NEW_GEOMETRY.md`（09:55）才改为 7/20。

### G2 ★★★ 由几何引出的「源分离必需」没有落地

`/run/media/sensen/Data2/cell_wound_prototype/outputs/embodied_body/ROUND26_NEW_GEOMETRY.md` 第 32–42 行已实测：20 µm pitch < 2×50 µm 捕获半径 ⇒ 重叠成为主态，**单根神经元最多被 21 个通道同时拾取**，3186/7737（41.2%）被多通道捕获。ROUND26 明确写「源分离从可选变必需，**我没有擅自实现**」。
现状核对：`access_map.py` 第 1417 行与 `electrode_frontend.py` 第 1740 行都写着 `"no_spike_sorting": True`；`engine/electrode_sorting.py` 有 2,082 行，但它是**排序/聚类诊断**，不是源分离。

### G3 ★★★ `capture_radius_um = 50 µm` 是声明值，而结论完全由它支配

`ROUND26` 第 46–60 行自己给出敏感性：10 µm→可及 0.0000；25 µm→0.0208；**50 µm→0.4167**；100 µm→0.8438；200 µm→1.0000。这个数既非实测也非器件参数推出，**几何重叠与捕获率的全部结论都挂在这一个自由数上**。

### G4 ★★★ 工作台的通道级检查依据的是旧几何

`WORKBENCH_P123_STATUS.zh-CN.md` 称已被验证的部分包括「电极检查器按准确通道ID联动台架检查」。
但旧几何（100/10）下 `ROUND25` 实测**通道碰撞 = 0**，所以「哪个通道对应哪个神经元」在旧几何里是唯一可辨的；换成 7/20 后同一实测给出 3186 个多通道神经元。**「通道↔神经元唯一对应」在 7/20 下不成立**，而这正是电极检查器的前提。

### G5 ★★ 存档工件早于源码，结果不能声称由当前代码复现

| 模块 | 源码 mtime | 存档工件 mtime | 差 |
|---|---|---|---|
| `model.py` 07:38、`run_validation.py` 07:40 | 2026-10-01 | `outputs/metrics.json` 00:35 | 工件早 **7 小时** |
| `mechanics.py` 08:46 | | `metrics_mechanics.json` 08:39 / `metrics_mechanics_lit.json` 08:44 | 早 2–7 分钟 |
| `cellstate.py` 11:59 | | `metrics_cellstate.json` 08:16 | 早 **3.7 小时** |

含义：P1/P2/P3/P5 的存档数字出自**当前模块之前的版本**。要么重跑留新工件，要么在报告里注明不可由当前代码复现。注意 `cellspace.py`(09:01→工件 09:07)、`model.py` 之后的 late 层(08:20→工件 08:21)时序正常。

### G6 ★★ 我上一轮报告有 7 处数字/引用错误（已核实）

1. **「47/47 测试」→ 实际 49/49**（`loop_selftest.json`）。
2. **行走三数是种子 0，被当成 10 种子结论**；跨种子胸高 0.6937–1.1184 mm。
3. **「电极阵列未开始」**与 2026-10-03 的 `electrode_frontend.json`（256 通道）冲突。
4. **「P2 成立」**未带工件自带的「该驱动无判别力」限定。
5. **「冻结臂 2.4×」**只对两个臂成立；`homeostasis_only` 是 4263.6×。
6. **`REPORT_P1P4.zh-CN.md` 第 9 行引 `vertex_division_probe.json`「0/5817」**——**该文件里根本没有 5817**，实为 n_side=5/10 各 3 次尝试、0 成功（共 6 次）。我亲自打开核过。
7. **我用 `python3 -m unittest test_body_export_regression` 跑回归，输出「Ran 0 tests / NO TESTS RAN」却当成通过**——该文件不是 unittest 模块，只有直接运行才有效。我自己复跑：`python3 test_body_export_regression.py` → PASS（exit 0）。**这是我上一轮把「没跑」当「跑过」的报告错误。**

### G7 ★★ 报告的溯源自洽性不足（不是数值错）

- PROTOCOL.md 第 14 行要求报告 MAE/**RMSE**/**bias**/fraction 四项，`metrics.json` 里四项齐全，但**两份中文报告一个 RMSE、一个 bias 都没有**（我 grep 确认：`REPORT_FULL.zh-CN.md`、`REPORT_P1P4.zh-CN.md` 中 rmse/bias 零命中）。
- `metrics_late.json` 同一量两个值：`pattern.radius_at["49.22"]=78.6923` vs `radius_49_22=79.1565`（差 0.46 µm），报告引的是后者。
- `slowphys_report.json` 里 `ledgers.n_time_jumps=0` 与 `time.n_time_jumps=1` 口径不同，报告未说明。

正面结果（也应记录）：`/run/media/sensen/Data2/cell_wound_prototype/outputs/experimental_control.csv` 与 `/run/media/sensen/Data2/cell_wound_prototype/upstream/controlCaRadData.m` **222 点逐点完全一致（最大绝对差 0.0）**，且 upstream commit `f0be2fa8e2da5ce59166866b72c819e3df2f874d` 与 `metrics.json` 记录的一致。

### G8 ★★ 可视化整合只覆盖了很小一部分（这条直接对应你的原始要求）

**进了身体内部视图的（`/run/media/sensen/Data2/cell_wound_prototype/viewer/data.js` 顶层键只有 6 个）**：
`meta`、`body`（69 节点/100 帧/69 网格）、`neural`（6000 点/100 帧）、`electrodes`（256 点）、`calcium`（400 细胞/100 帧）、`bench`。

**没有进身体视图的**（仅存在于 CSV/PNG/NPZ）：力学顶点能量与 T1/T2、细胞存活/ATP/膜完整性、胞内空间、血管输运与氧场、慢生理氧账本与气管、视觉与场景、行为 epoch 与可塑性消融、脑区与调质、电极阻抗/串扰/刺激伪迹。共 **389 张 PNG / 291 张 SVG / 11 份 HTML 没有一个是身体视图的一部分**。

其中**本来可以挂到身体上的**（有空间坐标或逐节点时间序列）：顶点力学（有网格）、氧/气管（有隔室但无空间）、可塑性（有图节点）、电极阻抗（有 256 个通道坐标）、场景光照（有相机）。**只有后面两类真正有空间落点。**

### G9 ★ 全保真规格的分层完成度（按 `SPEC_FULL_FIDELITY.md` 第 146–159 行的 P1–P9 与第 35–126 行的 L0–L8 逐条数）

| 状态 | 数量（约） | 内容 |
|---|---|---|
| 有实现 + 可复跑自检 | ~25 项 | L1 顶点力学、L2 钙/状态、L3 胞内空间(单细胞)、L4 受配体、L5 间隙连接、L6 具身闭环、L7 慢生理、L8 生长/谱系 |
| 只有数值验证、无生物校准 | ~10 项 | 力学参数、受体转导、速率→行为映射 |
| 只有假设/演示 | ~12 项 | 场景、视觉桩、电极电弧、刺激 |
| 缺数据 | ~8 项 | innexin 单通道电导、τ_p、放电率分布、气管电导、果蝇代谢率、T1 实测率(有文献值但未接入) |
| **未实现** | **12+ 项** | **P7 三维多层组织**、**P8 免疫/增殖/再生**、**P9 灌注闭环**、飞行、嗅觉、肌肉、睡眠、细胞体积/渗透/ATP/pH 动力学、ECM、脑区/细胞类型动力学、源分离、ADC 非线性 |
| 规格自己列为不可行 | 4 项 | 全原子 MD、全脑逐分子、意识、长期轨迹预测 |

**对照最原始的目标（离体脑保留 / 灌注 / 与主体闭环）：全部落在上面最后两行，即未实现或原理不可行。**

### G10 ★ 交付物自身不一致

- 桌面 `/home/sensen/Desktop/cell_wound_prototype/` 里 `index.html`(10-01 00:36)、`index_full.html`(10-01 15:31) 是**被工作台取代**的旧 HTML 报告，仍在根目录显眼处。
- 桌面 `engine/` 本轮之前缺 30 个文件（全部 electrode*.py、embodied/、da_protocol.py 等）。**已按你的选择同步**，见第 7 节。
- `/run/media/sensen/Data2/cell_wound_prototype/outputs/PHYSICELL_NOTES.md` 与报表自称「Illustrative (not measured, not cited)… *all* rate constants」——PhysiCell 血管生成那一层**参数全为手选**，不能当结果。

---

## 4. 距离目标还差什么（按「能不能验证」分档）

**这不是算力问题，`SPEC_FULL_FIDELITY.md` 第 5–28 行自己列的四条硬上限依然成立**，我复核后同意其中三条的表述，第四条（观测通道）需要补一句：本项目还多了一道自加的墙——**没有任何实验数据**（第 2 节里所有「实测」都是本机数值自检，不是生物测量）。

| 档 | 内容 | 差的量 |
|---|---|---|
| A 已可验证 | 钙信号对照一条曲线、力学自检、具身行走/闭环、慢生理账本、电极记录链数值自检 | 与目标无关，是地基 |
| B 可做但要新数据 | 三维多层组织（notum 厚度 7.5–10 µm 已有，三维几何没有）、真实脑区动力学（BANC 有注释但无实测活动）、电极源分离（需要算法选择 + 冒烟验证） | 缺数据 + 缺算法 |
| C 只能演示 | 灌注、代谢、pH、体积调节、免疫、再生、飞行/嗅觉 | 缺参数，且无对照实验 |
| D 原理不可行 | 全脑逐分子、意识/身份连续、长期个体轨迹、离体脑长期认知保留 | 不因投入而改变 |

**最实际的下一步（若继续）**：
1. 把电极几何收敛成单一来源（`ElectrodeArraySpec` 一份），重算 20/7 下的访问映射与重叠，再决定是否需要源分离模块（G1–G4）；
2. 用当前代码**重跑并留新工件**，让 P1–P6 的存档与源码对齐（G5），并补 RMSE/bias/fraction 进报告（G7）；
3. 修正 G6 的 7 处引用错误（已完成 6 条的定位，只差改文字）；
4. 把「本来有空间落点」的四类数据接进身体视图（G8），并给每类加实测/假设标记；
5. 桌面旧 HTML 报告归档、只留工作台一个入口（G10）。

---

## 5. 我这次做了什么、没做什么

**做了**：全盘盘点（17.8 GiB / 37,526 文件）；派两个独立只读代理复核 24 条声称；自己复算并打开核对了 G6 第 6、7 条与 G8 的载荷结构；清理缓存/空文件；同步桌面 `engine/`；写报告索引。

**没做**：没有重跑任何会覆盖 `outputs/` 的仿真入口（`run_validation.py`、`run_mechanics.py`、`run_cellstate.py`、`run_cellspace.py`、`run_late.py`、行走/闭环/可塑性/慢生理套件）——它们会写回既有工件；没有核对原始文献（PMC10208100 / Bonnet 2012 / Curran 2017），τ 5–20 s 与 T1 8.5e-4 只追到 `/run/media/sensen/Data2/cell_wound_prototype/MEASURED_ANCHORS.md` 的二手摘录；没有播放两个 MP4；没有重新生成 `viewer/data.js`。

**风险声明**：本文所有结论都是本机数值与文件的核对，**没有一项构成生物学验证、器件验证或安全声明**。电极直径 7 µm、间距 20 µm 未改动，也未被优化。
