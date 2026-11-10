# BANC 离线信息诊断（20261107T203049Z / information）

预登记与全部数字均来自本目录已保存的产物，本条**没有执行任何物理步或神经步**（physics_steps=0，neural_steps=0，training_updates=0）。这不是成品，也不是行为成功证据。

## 1. 目的与范围

固定物理学习研究 20261107T195435Z 的结论是：168 系数同腿 MN 读出在零下行驱动下未超过零输出基线（selected 仍为 trial_000 零参数）。本轮只做离线诊断，分离两个假设：

- **假设 A：冻结感觉编码器过粗 / 自由度不足**（556 细胞、每腿 4 通道：爪位置、钩方向、毛板角、钟形感器应变）。
- **假设 B：神经/读出阶段压缩或无法暴露可用状态**。

诊断边界：不新增物理/神经仿真步、不拟合可部署策略、不改验收阈值、不从相关数据给出生物学结论。

## 2. 路径与预登记

| 项 | 完整路径 |
|---|---|
| 预登记（绑定） | `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-descending-learning/20261107T203049Z/information/PREREG.json` |
| 工作区产物 | `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-descending-learning/20261107T203049Z/information` |
| 桌面镜像 | `/home/sensen/Desktop/engine修复验收/banc-descending-learning/20261107T203049Z/information` |
| 冻结编码器 | `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T185910Z/encoder/encoder.py` |
| 48 个非零 trial 源数据 | `/home/sensen/Desktop/engine修复验收/banc-physical-learning/20261107T195435Z/learning/training/trial_001..012,014..025,027..038,040..051` |
| 旧失败源（外部诊断，仅计数与转用） | `/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T185910Z/readout/TRAIN_TRACE.npz` + `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T192931Z/dynamics/SOURCE_TRACE.npz` |
| 基线 trial_000 与 4 个对照 | `/home/sensen/Desktop/engine修复验收/banc-physical-learning/20261107T195435Z/learning/training/trial_000,013,026,039,052`（全部排除） |

预登记 SHA256：`a5a90d8723287f9a9448616da937e0bec28da5e147a33b133ac64153cff102c9`（`VERIFICATION.json` 已核对该文件自审计开始后未被改动）。

## 3. 步骤一：编码器灵敏度/混叠审计

独立重写四条声明公式（不调用 encoder 的 helper 特征），并在**全部** 9600 个训练 bin 与 1817 个旧失败源 bin 上与真实编码器逐 bin 比对：

| 检查 | 结果 |
|---|---|
| 记录感觉电流最大误差（真实 encoder vs 保存的 556 电流） | **0.0 nA** |
| 独立公式 vs 保存的 24 个通道均值最大误差 | **6.94e-18 nA** |
| 解析 Jacobian vs 中心差分 1e-6（24x90 通道图） | **2.47e-12** |
| 解析 Jacobian vs 中心差分（556x90 细胞图） | **2.54e-12** |
| 编码器调用数 | 11711 |

参考输入：42 个关节范围中点、膝速度 +2 rad/s、每腿载荷 W/6=4.0744（W=24.446354755878005）。输出钳位 [0, 0.06] nA 在参考点未激活（通道最大值 0.038080 nA）。

**敏感度与秩（90 输入 -> 24 通道）**

| 块 | 输入列数 | 非零列 | 零列 | 独立方向（数值秩） |
|---|---:|---:|---:|---:|
| q（关节角） | 42 | 24 | 18 | **12** |
| qvel（关节速度） | 42 | 6 | 36 | **6** |
| foot_load（足载） | 6 | 6 | 0 | **6** |
| 合计 | 90 | 36 | **54** | **24** |

- 66 个零列更正为 **54 个**：18 个 q（6 腿 x 3 个 trochanter 角）+ 36 个 qvel（全部非膝关节速度）。此前一条进度消息写作 66，属口误，已在 `FAILED_DIAGNOSTIC_ATTEMPTS.json` 记录并更正。
- 24 个非零 q 列只给出 **12 个独立方向**：每腿 coxa yaw/pitch/roll 只进入同一个 hairplate 均值通道（每腿秩 1）。
- 90 维输入到 24 维输出的映射秩仅 **24**，其余 66 维输入空间完全不可见。

**工程化别名固件（18 个 = 6 腿 x 3 类；不是真实身体轨迹）**

| 固件 | 构造 | 编码器电流 | 独立公式 |
|---|---|---|---|
| coxa yaw/pitch 中点互换 | 同腿 yaw=mid+d、pitch=mid-d 与互换后 | 逐位相同 | 逐位相同 |
| trochanter 角改变 | 同腿 Trochanter_yaw/roll 改到实际范围内 10% 位 | 逐位相同（**无输入端口**） | 逐位相同 |
| 膝速度 +2/-2 | 同腿膝速度反号 | 逐位相同（精确绝对值混叠） | 逐位相同 |

全部固件满足：关节范围合法、q 或 qvel 确实不同（`distinct_q`/`distinct_qvel`），且 `physical_trajectory=false`、`collision_validity_not_tested=true`——它们是人为构造输入，绝不代表真实身体姿态。

另有 **6 个非预登记的可见性对照**：把膝角改动 10% 行程后电流变化 4e-3 nA（编码器确实能分辨），证明秩 24 不是代码假象。

**接口缺口声明（原文记录）**："A zero input column is an uncalibrated interface gap in this proxy encoder, not evidence that source sensory biology lacks that information. No cell tuning direction is inferred."

## 4. 步骤二：72 个固定拟合

固定协议：train=gen0+gen1（24 episodes，4800 bin），test=gen2+gen3（24 episodes，4800 bin）；ridge lambda=1；只用训练集均值/标准差做标准化（下限 1e-8）；诊断模型含未惩罚截距；分类用平衡类权重 N/(2 n_class)；无调参、无按测试集选择。

特征维度（每腿）：encoded4=4；MNinstant=42–60；MNhistory=168–240（滞后 0/1/4/10 bin = 0/5/20/50 ms，lag-major，逐 episode 零前缀）；rawphysical=15（7 q + 7 qvel + 1 足载，特权，永不部署）。

分类标签：stance/air 用足载 > 0.05W 为 +1、< 0.005W 为 -1（中间值排除）；膝速度符号用同腿 Tibia_pitch 速度 |>1e-3| rad/s。**encoded4 的 stance/air 任务是同义反复**：campaniform_strain 按设计就是足载的函数。

**stance / air（测试集平衡准确率，chance=0.5）**

| leg | encoded4 | MNinstant | MNhistory | rawphysical |
|---|---|---|---|---|
| LF | 0.9843 | 0.5618 | 0.5432 | 0.9370 |
| LM | 0.9929 | 0.4897 | 0.4787 | 0.9685 |
| LH | 0.9785 | 0.4548 | 0.4357 | 0.9615 |
| RF | 0.9809 | 0.7629 | 0.8218 | 0.9517 |
| RM | 0.9923 | 0.5319 | 0.5256 | 0.9788 |
| RH | 0.9872 | 0.5558 | 0.5669 | 0.9341 |


**膝速度符号（测试集平衡准确率，chance=0.5）**

| leg | encoded4 | MNinstant | MNhistory | rawphysical |
|---|---|---|---|---|
| LF | 0.5821 | 0.5814 | 0.5776 | 0.8901 |
| LM | 0.5560 | 0.5191 | 0.5252 | 0.8196 |
| LH | 0.4421 | 0.4906 | 0.5116 | 0.8233 |
| RF | 0.5528 | 0.5005 | 0.5199 | 0.8224 |
| RM | 0.5373 | 0.5523 | 0.5486 | 0.9580 |
| RH | 0.6466 | 0.5541 | 0.5284 | 0.6930 |


**带符号膝速度回归（测试集 R²，0=不优于训练均值）**

| leg | encoded4 | MNinstant | MNhistory | rawphysical |
|---|---|---|---|---|
| LF | 0.1735 | 0.1283 | 0.3187 | 1.0000 |
| LM | 0.1100 | -0.0156 | -0.0254 | 1.0000 |
| LH | 0.3477 | 0.4088 | 0.5215 | 1.0000 |
| RF | 0.6532 | 0.0168 | 0.0711 | 1.0000 |
| RM | -0.0955 | -0.0159 | -0.0857 | 1.0000 |
| RH | 0.5999 | 0.1215 | 0.3161 | 1.0000 |


**带符号膝速度回归（测试集 MSE）**

| leg | encoded4 | MNinstant | MNhistory | rawphysical |
|---|---|---|---|---|
| LF | 0.03486 | 0.03677 | 0.02874 | 0.00000 |
| LM | 0.02738 | 0.03125 | 0.03155 | 0.00000 |
| LH | 0.07670 | 0.06952 | 0.05626 | 0.00000 |
| RF | 0.03182 | 0.09021 | 0.08522 | 0.00000 |
| RM | 0.02629 | 0.02439 | 0.02606 | 0.00000 |
| RH | 0.04779 | 0.10493 | 0.08168 | 0.00000 |


**旧失败源转用（不重拟合、仅用已训练模型；外部诊断）——膝速度符号平衡准确率**

| leg | encoded4 | MNinstant | MNhistory | rawphysical |
|---|---|---|---|---|
| LF | 0.4814 | 0.4771 | 0.4628 | 0.8944 |
| LM | 0.5022 | 0.4951 | 0.4706 | 0.8591 |
| LH | 0.4956 | 0.5025 | 0.5085 | 0.8921 |
| RF | 0.5655 | 0.3914 | 0.3911 | 0.6388 |
| RM | 0.5268 | 0.4822 | 0.4981 | 0.9611 |
| RH | 0.5103 | 0.5111 | 0.4944 | 0.8850 |


**旧失败源转用——stance/air 平衡准确率**

| leg | encoded4 | MNinstant | MNhistory | rawphysical |
|---|---|---|---|---|
| LF | 0.9132 | 0.4594 | 0.5045 | 0.8301 |
| LM | 0.8998 | 0.4920 | 0.4782 | 0.8716 |
| LH | 0.9381 | 0.5437 | 0.5189 | 0.8255 |
| RF | 0.9160 | 0.7164 | 0.7019 | 0.9232 |
| RM | 0.9869 | 0.4505 | 0.4743 | 0.9559 |
| RH | 0.9789 | 0.4583 | 0.4867 | 0.9056 |


NOT_ESTIMABLE 数量：**0**（无缺失类别，条件一个都没有被静默丢弃）。

## 5. 步骤三：图

- 图文件：`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-descending-learning/20261107T203049Z/information/FIT_SUMMARY.png`（已实际打开渲染确认）
- 四面板：stance/air、膝速度符号、带符号膝速度 R² 的 腿x表示 热图，以及编码器秩/别名摘要柱状图。
- 桌面镜像同名文件在 `/home/sensen/Desktop/engine修复验收/banc-descending-learning/20261107T203049Z/information/FIT_SUMMARY.png`。

## 6. 步骤四：独立验证

`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-descending-learning/20261107T203049Z/information/verify_information.py` **不 import** `fit_information.py` 与 `audit_encoder.py`，从冻结编码器源码和 195435Z 的 trial 产物重建标签、滞后堆叠、训练统计、ridge 正规方程、预测与指标。

| 项 | 结果 |
|---|---|
| 状态 | **PASS** |
| 检查数 / 通过 / 失败 | 1914 / 1914 / 0 |
| 重算拟合数 | 72 |
| 独立读取产物次数 | 246 |
| 系数最大相对偏差 / 预测最大绝对偏差 / 指标最大相对偏差 | 2.32e-14 / 8.61e-09 / 2.39e-16 |
| 保存的正规方程残差上界 | 3.01e-12 |
| 独立重算秩 / 零列 | {'channel_map_24x90': 24, 'q_block_24x42': 12, 'qvel_block_24x42': 6, 'foot_load_block_24x6': 6} / 54 |

## 7. 失败与更正记录（全部保留）

见 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-descending-learning/20261107T203049Z/information/FAILED_DIAGNOSTIC_ATTEMPTS.json`，要点：

1. **拟合首轮（已废弃）**：中间阈值行以 NaN 目标进入 ridge 求解，全部系数变 NaN、所有预测塌到 -1，72 个平衡准确率恰好 0.500。因 encoded4 stance/air 的同义反复性质本应接近 1，判定该轮无效，未用于任何报告。产物保存为 `FAILED_NAN_TARGET_FIT_RESULTS.json` / `FAILED_NAN_TARGET_FIT_MODELS.npz`。修正：无标签行权重置 0 且目标用有限占位值，同时从拟合与评分中排除；lambda、切分、表示、滞后、阈值一律未改。
2. **独立验证第 1 轮**：1842 检查中 10 项失败——其中 2 项因独立公式与 `np.tanh` 存在约 1e-18 的浮点差异却用了精确相等；1 项对 9.4e8 量级的源转用 MSE 用了绝对容差；6 项对特权回归目标期望"精确恒等"。
3. **独立验证第 2 轮**：1914 检查中 6 项失败，全部是 `rawphysical` 带符号膝速度的"精确恒等"期望——实际是 lambda=1 的正常收缩（1-R² = 4.84e-08–2.62e-07，残差 3.0e-4–2.4e-3 rad/s），不是数据缺陷。
4. **验证第 3 轮**：改为断言预登记的 ridge 解（R² > 1-1e-6）并记录剩余收缩后 **1914 项全部通过**。lambda、数据与切分未改。

## 8. 计数语义（严格）

- **拟合数 = 72**：等于"腿 x 表示 x 目标"的 ridge 解个数；特征构造、标准化、指标计算不计入拟合；上限 72 未被突破。
- **数据读取**：拟合脚本 2 次 npz 读取；审计脚本 290 次文件读取（含 288 个 trial 产物）；独立验证 246 次 npz 读取。
- **编码器调用**：11711 次（纯数组运算）。
- **物理步 / 神经步 / 训练更新**：**0 / 0 / 0**。
- 训练 bin 9600（24+24 episodes x 200 bin）+ 旧失败源 1817 bin。

## 9. 两个假设的读法（严格限定）

- **假设 A（编码器过粗）**：方向信息上有支持。`hook_direction` 是 `0.05*tanh(|膝速度|/10)`，是精确的绝对值混叠；除特权 rawphysical 外没有任何表示在膝速度符号上明显超过 0.65 平衡准确率；90 维输入中 54 列完全无端口，映射秩 24。
- **假设 B（神经/读出压缩）**：状态信息上有支持。encoded4 在 stance/air 上达到 0.978–0.993 测试平衡准确率，而同一批 bin 的同腿 MN 表示只有 0.436–0.822（见 `aggregates_by_task_and_representation`），即 MN 观测压缩了编码器已经暴露的信息。
- **尚未确立**：以上都不是因果神经表征、知觉或运动成功的证据；48 个 episode 共享同一 canonical checkpoint 与确定性策略变体，样本高度相关；本轮没有任何物理或神经步。

## 10. 明确边界

- 粗编码器是**故意未标定**的工程代理；它的信息损失**不能**用来断言源感觉生物学缺少该信息。
- 零输入列是**未标定的接口缺口**，不是生物缺失证据。
- 未编造细胞身份，未推断调谐方向；通道顺序与端口全部来自已冻结的身份记录。
- `rawphysical` 是特权诊断量，**永不可部署**；本轮没有任何策略被拟合用于部署，selected 政策仍为 trial_000 零参数（历史事实未变）。
- 未修改任何冻结/历史文件；未执行任何物理或神经仿真。

## 11. 下一步（已准备、未执行）

`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-descending-learning/20261107T203049Z/information/run_dn_pair.py`（只写不跑）：确定性选取历史中**最大训练奖励**的非零支持可行 Actor（trial_017，reward -1.0144429918；tie-break 用 trial 序号；永不选零 incumbent），在同一个完整 canonical 1s checkpoint 上跑 2 秒释放，唯一差别是下行电流 0 vs 0.05 nA（`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-descending-learning/20261107T203049Z/descending/command.py`，sha256 `ba84bca2b7e3ee710c3412467db2a4eadcd839dd38d027d44241b7b9363b455c`）。必须显式 `--execute` 才会真正运行；由父代理执行（输出到 `/home/sensen/Desktop/engine修复验收/banc-descending-learning/20261107T203049Z/physical/dn_zero` 与 `dn_on`）。

**本轮对该 runner 只做了 plan 模式验证（无物理、无神经步）**：确定性选择复现 12 个合规候选中的 trial_017（reward -1.0144429918，theta 最大绝对值 0.022107），命令模块哈希 `ba84bca2b7e3ee710c3412467db2a4eadcd839dd38d027d44241b7b9363b455c` 与预期一致，DN root ids 为 [720575941433155799, 720575941566493282]。

- 默认（无 `--execute`）：只打印计划，**不 import mujoco、不建 Plant、不建神经核、不写任何文件**。
- `--execute`：两次 2 s 释放，每次 160000 个 12.5 µs 原生步、2000 个神经步、160002 次 forward；合计 320000 物理步 / 4000 神经步；每个条件 13 个文件（含 `DN_INPUT.npz`），并在父级物理目录写 `DN_PAIR_PLAN.json` 与 `DN_PAIR_MATCH.json`（配对一致性：初始物理态、执行器目标、神经初态 v/spike、Actor theta/weights 逐位相同，唯一差异是 DN 向量）。
- 若确定性选择结果与已记录的最佳非零候选不一致，runner 直接报错退出，**不会静默换 Actor**。
