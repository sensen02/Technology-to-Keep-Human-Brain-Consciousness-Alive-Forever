# 偏离记录（DEVIATIONS）

本文件按预登记 `PREREGISTRATION.md` / `PREREGISTRATION.json` 第 3 节"deviation_policy"追加：
凡发现定义缺口，只允许在本文件记录并重跑全量分析，不允许静默改判据。
预登记文件本身**不改写**（其 sha256 已写入 `RESULT.json`，改动即会使核验失败）。

## 偏离 1：镜像矩阵规则（预登记规则在"自镜像"块上恒等于自身）

- **预登记原文**（`PREREGISTRATION.md` §2.7 第 2 条）：
  `G'[b_src][b_dst] = G[mir(b_dst)][mir(b_src)]`，括号说明为"同时换侧并反向"。
- **发现的缺口**：该规则把"换侧"与"反向"复合在一起。对**目标侧本来就是源侧镜像**的块——
  即 `LF→RF`、`LM→RM`、`LH→RH` 及其反向 `RF→LF`、`RM→LM`、`RH→LH`——规则给出
  `G'[LF][RF] = G[mir(RF)][mir(LF)] = G[LF][RF]`，即该块与自身比较，差值恒为 0。
  于是这套规则**在本轮最关心的三条同行对侧耦合上结构性失明**，无法回答
  "从左到右与从右到左的电导是否相等"。这是定义错误，不是数据结论。
- **处理**（不删旧结果、不改预登记）：
  1. 预登记规则的结果**照原样保留**并继续报告，键名显式改为
     `SYMMETRY.json -> conductance_matrix_mirror_preregistered_rule`；
  2. 增加**修正规则** `G'[b_src][b_dst] = G[mir(b_src)][mir(b_dst)]`（只换侧，不反向；一条
     `A→B` 边镜像为 `mir(A)→mir(B)`），写在
     `SYMMETRY.json -> conductance_matrix_mirror_corrected_rule`，并作为**主判据**；
  3. 腿对级另行给出 `direction_asym_uS = G(A→B) − G(B→A)`，这是自镜像腿对唯一有意义的
     左右不对称量，并加 `self_mirror_pair` 布尔标记，避免读者把恒为 0 的腿对级镜像差
     误读成"完全对称"。
- **判定阈值未变**：`ASYM_TOL = 0.05`；`absent/trace/...` 分档未变；耦合判定仍用精确整数路径计数。
- **影响**：两个规则都报告，因此不存在"因为改了规则而让结论变好"的余地；两者在本数据上
  给出不同但同向的结论（见 `REPORT.md` 第 5 节）。

## 偏离 2：两个数字与交接文本不一致（只报告，不改动任何输入）

- 交接文本称 `allowed_sensory_local` 为"556 个合格感觉输入"。冻结模块实测为 **5,103**
  （`LocalCircuit.allowed_sensory_local`，与 `leg_sensory_nerve_union` 减去
  `SENSORY_CHANNEL_REVIEW.json` 的 4 条歧义记录后逐位一致，本轮独立核验亦复算通过）。
- `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T185910Z/dynamics/PHYSICAL_QUALIFIED_PORTS.npz`
  的 `local_indices` 长度为 **559**；其中落在 4 条歧义记录里的为 **0** 个，
  因此 559 − 4 也不能得到 556。
- 结论：**556 无法由本轮可读的冻结产物复现**。本轮不使用这个数字做任何判断，
  腿池/块/耦合结论与它无关。此差异如实记录，不推测原因，也不修改任何输入。

## 偏离 3：路径长度上限

预登记写"长度 ≥ 3 的路径本轮不枚举"。实际执行时同样没有对长度 3 及以上的路径做任何
可达性判定；`COUPLING.json` 的 `path_lengths_not_analyzed` 字段记录的就是这条限制。
这不是事后改判据，而是预登记规则的原样执行。

---

未发生其他偏离。分析代码 `leg_pools.py` 的 sha256 记录在 `RESULT.json`，
独立核验脚本 `verify_leg_pools.py` 的 sha256 记录在 `INDEPENDENT_VERIFICATION.json`。
