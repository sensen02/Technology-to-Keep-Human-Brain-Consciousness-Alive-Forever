# 神经侧接口重建（第 5 部分）：通道模式接入真实受体库并通过端到端验收

**手动实施的实现，不是成品。** 规范模型未被修改
（SHA256 仍为 `65d3904dce9802f46997a92cb3f80cd13c032f847c1e1f6710694d65cfdd73b6`，程序内断言校验）。

---

## 0. 一句话

**通道模式真的接进了 `MechanoReceptorBank`**，四条端到端判据 B1–B4 全过，
默认路径（`channel_mode=False`）行为不变、每腿感觉组大小与独立勘测逐位一致。
接的过程中撞出并修掉了 **3 个真实的接口 bug**——都是"名字/键不一致"这一类，与前几轮同族。

---

## 1. 接线方式

| 位置 | 改动 |
|---|---|
| `engine/embodied/adapters.py::ReceptorConfig` | 新增 `channel_mode: bool = False`、`channel_current_nA_per_unit: float = 3e-1`（并进 `validate()`） |
| 同文件 `MechanoReceptorBank.__init__` | 新增可选参数 `built_roots`；`channel_mode` 为真时构造 `ChannelSensoryEncoder` |
| 同文件 `MechanoReceptorBank.stimuli` | 新增 `_channel_stimuli()`：返回**同样形状**的两个 `(6,)` 数组，但 `angle_strain` 现在是**带符号**的聚合量 |
| 同文件 `MechanoReceptorBank.transduce` | 通道模式下走编码器，并在 drive 里带上 `channel_record` / `channel_stats` |
| `engine/embodied/loop.py` | 把 `built["tier"]["root_ids"]` 传给受体库（**调用点其余一行未改**） |

**关键设计**：`angle_strain` / `force_strain` 的**键与形状完全不变**，所以
`loop.py` 的 `observed["force_strain_per_leg"]` 等下游消费者不用改；
但 `angle_strain` 从 `|q−q_ref|` 的均值变成**带符号**量——**符号盲缺陷就在源头被修掉了**。

## 2. 端到端验收（第 14 步）

`run_neural_rebuild_14_channel_mode_acceptance.py` → `..._14_channel_mode_acceptance.json`
（**解释器 `venv_body/bin/python`**，用同一个合成观测同时驱动**新旧两条路径**）

| 判据 | 结果 |
|---|---|
| **B1** 通道模式可用真实 tier 与真实感觉组构造，并报告放置了多少细胞 | **PASS** |
| **B2** 旧路径对屈/伸**完全相同**（其定义性缺陷），通道模式**不同** | **PASS** |
| **B3** 通道模式电流的支撑**只在通道细胞上**，非通道细胞电流恰好为 0 | **PASS** |
| **B4** 两条路径对同一扰动的输入向量**不同**（开关非装饰性） | **PASS** |

```
legacy  flex/ext angle_strain identical : True    ← 缺陷被复现
channel flex/ext angle_strain identical : False   ← 修复生效
```

## 3. 接线过程中修掉的 3 个真实接口 bug（都属于"名字/键不一致"）

1. **`leg_rows_from_dof_order` 对规范模型不可用**：它从 FlyGym DOF 序的 child 名解析腿
   （小写 `lf_coxa-yaw`），而编译模型里是 `joint_LFCoxa_yaw` → 抛
   `"DOF 0 (LFCoxa_yaw) does not belong to exactly one leg"`。
   **改为直接从关节名建角度映射**（每腿 7 行：coxa yaw/pitch/roll、trochanter yaw/pitch/roll、tibia pitch）。
   期间还试错两次命名约定（无前缀、前缀大小写），已写进代码注释。
2. **`stimuli()` 要求 ≥66 维关节角，而规范模型只有 42 个铰链** → 补零到 66，
   并**显式声明**腿行都在前 42 槽内（`[0..6] [7..13] [14..20] [21..27] [28..34] [35..41]`），不隐藏这个补零。
3. **腿键大小写不一致**：`adapters.LEGS` 是小写 `("lf",...)`，而 `sensory_channels.LEGS`
   我写成了大写 `("LF",...)` → 银行查 `groups["lf"]` 时 **KeyError**。
   统一为小写（模块、导出器、查找表三处），这是**跨模块契约**问题。

**另外**：`sensory_channels` 原本用 pandas 读 feather，但**能跑 MuJoCo 的解释器没有 pandas**
（实测：`venv_body` 有 mujoco 无 pandas；`venv` 有 pandas 无 mujoco）。
改为**导出一次查找表** `outputs/neural_rebuild/sensory_channel_lookup.json`
（`tools_export_sensory_channel_lookup.py`，记录源文件 sha256
`86ccf5df…0098` 与规则），运行时只读 JSON。

## 4. 回归（确认没破坏既有路径）

| 检查 | 结果 |
|---|---|
| `ReceptorConfig.channel_mode` 默认 | **False**（既有行为不变） |
| `ReceptorConfig.validate()` | 通过 |
| `build_neural_tier` 仍可运行 | n=12000，6 个感觉组 |
| **每腿感觉组大小** | **665 / 658 / 736 / 623 / 651 / 808** —— 与独立勘测**逐位一致** |

## 5. 仍应记录的边界

- 验收用的是**合成关节角向量 + 合成接触力**，直接调 `stimuli()`/`transduce()`：
  **没有跑身体、没有跑 CPG、没有完整 episode**，所以**不证明果蝇被驱动或站得住**。
- 通道的**符号约定是人写的**（数据集标了 `direction` 但不带符号）。
- 角度向量补零到 66 是本测试的适配，**不是**模型真有 66 个铰链。
- `club`（振动）与 `multidendritic`（伤害）**声明为 0**，未用代理量。

## 6. 本目标内仍未达成

1. **闭环可控性**：时序绑定未解决（5 ms 窗口下每细胞每窗口 **0.00495** 个脉冲，需要 3 个）。
2. **姿态重搜**：把 16 个无本腿池执行器如实置零后（占第37轮激活 **43.2%**），可行姿态必须重搜。
3. 未跑**完整 episode**（`loop.py` 的端到端）验证通道模式。

## 7. 产物与代码

- 修改：`engine/embodied/adapters.py`（`ReceptorConfig` + `MechanoReceptorBank`）、`engine/embodied/loop.py`
- 新增：`engine/embodied/sensory_channels.py`（第 4 轮）、`tools_export_sensory_channel_lookup.py`
- 产物：`outputs/neural_rebuild/neural_rebuild_14_channel_mode_acceptance.json`、
  `sensory_channel_lookup.json`
