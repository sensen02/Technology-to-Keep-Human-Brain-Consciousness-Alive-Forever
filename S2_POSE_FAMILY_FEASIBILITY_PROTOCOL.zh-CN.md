# 声明姿态族的必要性可行性判定协议（执行前登记）

手动研究原型，非成品。目的：把"这个姿态支撑不可行"到底是**姿态属性**还是**模型属性**分开。只做**必要性可行性**（静态线性松弛是否存在解）与**精确证书**；**不设任何表现目标**、不随机搜、不放宽激活上下界、不调增益、不用下沉/画面/回放当站立证据。

## 判据（执行前冻结）

对每个声明姿态：构造与 A 门完全相同的 `A=[R, C, limit]`、`b`（bias−passive−base 口径），检查静态松弛可行性

> 存在 `x`，使 `A x = b`，`lo ≤ x_muscle ≤ hi`，`x_contact ≥ 0`

- **主口径（唯一用于结论）**：**native 激活上下界 lo=0.0001、hi=1**（规范模型自带，不放松）；接触锥 diamond（4向）与 outer（4斜向）各报一次。
- **判定语义（必要性）**：若上述松弛**不可行**，则该声明姿态**不存在**静态支撑（必要条件下失败）；若**可行**，仅说明必要性条件通过——**不等于**可支撑、**不等于**稳定、**不等于**站立（松弛解可能物理不可实现）。
- **次要参考（不用于结论）**：`lo=0` 的 zero-floor 变体，仅作对照并列，明确标注"非主口径"。
- 每姿态每口径只跑一次；不因结果重跑、不调阈值、不放宽边界。
- 若某姿态未通过既有**标量关节范围门**（越界 >1e-10），判 `REJECTED_BY_RANGE_CHECK`，**不裁剪、不代入**——该结果本身是事实，不是失败。

## 姿态族（预先固定；取自既有已记录取值，无随机、无新搜索）

| 成员 | 构造（全部使用既有代码路径） |
|---|---|
| M1 `keyframe_seated` | `model.key_qpos[0]` 原样 + `engine.canonical_physics.seat_root`（既有就座约定） |
| M2 `round30_named_seated` | keyframe，把 `outputs/stance_joint_feasible_round30/stance_for_runner.json` 的 `joints_rad`（18个具名关节）写入 + seat_root |
| M3 `round37_best_seated` | keyframe + `outputs/stance_margin_round37/stance_margin.json` 的 `best.x`（42偏移，参数序 `r16.LEGS × ('Coxa_yaw','Coxa_pitch','Coxa_roll','Trochanter_yaw','Trochanter_pitch','Trochanter_roll','Tibia_pitch')`，**按该搜索自身 `make_qpos` 的 clip 语义**写入）+ seat_root |
| M4 `round37_best_scale0.5_seated` | 同上，但偏移整体 ×0.5（**确定性**，非随机） |
| M5 `round37_best_scale1.5_seated` | 同上，但偏移整体 ×1.5（**确定性**，非随机） |
| M6 `saved_projected_pose` | 既有存档 `outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz` 的 `projected_seated_qpos`，**不再坐座**（作为已知不可行的锚点，用于一致性核对） |

## 交付（三件）

1. 姿态族中**不可行/可行的比例**（主口径：native 边界；diamond/outer 各报）。
2. 若存在**可行**样例：给出该样例的**可独立验证的证书**（LP 原始解 `x`，已由既有 gate 保存在 `native_diamond.npz` 的 `x/residual` 中）；若某姿态**不可行**：给出**精确有理分离证书**（复用 `tools_s2_support_certificate.py` 的精确机制）。
3. **结论**：在**参数未改**的前提下，该声明模型**是否存在任何静态支撑**——仅限本姿态族内可判定；族内全部不可行只能说"本族内未见可行姿态"，不等于全姿态空间不可能（该界限必须写在结论里）。

## 明确不做

不改 canonical XML、不改接触参数、不改质量/力标度、不搜索姿态、不引入粘附/支架/PD/CPG、不跑神经闭环、不把下沉或回放当支撑证据、不把可行解当站立。

## v1.1 勘误（2026-10-15）：判定语义的力量标度限定

本协议"不可行 ⇒ 该声明姿态不存在静态支撑"的表述，必须加上限定：该不可行是在 **s=1（当前未标定力标度）** 下的结果，且裕度仅约 **4.5%**（首个真阈值≈1.0450258）；由于 mass/force scale 为 **NOT_ESTABLISHED**，该结论只等价于"当力参数取当前未标定值时不可行"，**不得**用作换姿态或否定站姿的依据。可行/不可行均限于**线性松弛**，不是可支撑姿态、不是合规动力学平衡、不是生物结论。
