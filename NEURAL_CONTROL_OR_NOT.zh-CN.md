# 是不是"真实神经元控制"？—— 三条不同的链，必须先分开

**结论先讲：不能确保，因为按你问的那个意思，答案是否定的。**

而且这里有一个真实存在的混淆（MIXUP）：项目里有**三只不同的"果蝇"**，它们由完全不同的东西驱动，跑在不同的模型上。之前的对话里这三者的数字被混在一起引用过。本文把它们彻底拆开。

---

## 链 ①「走路录像 / 肢体角」——**零个神经元**

**产出**：`arena_bare_seed0–3` 四个 episode，也就是全部逐帧关节角数据（173 个角度、中位误差 1.730°）和触觉/视觉记录。

| 项 | 实测值 |
|---|---|
| 模型 | NeuroMechFly 型（FlyGym） |
| body / geom | 70 / 69 |
| 执行器 | **42 个位置执行器**（= 6 腿 × 7 DOF），**无肌肉** |
| 铰链自由度 | 66 |
| 总质量 | **0.00102431 模型单位 = 1.02431 mg**（长度单位 mm，质量单位 g，力单位 1 µN） |

**腿由什么驱动**：`run_arena_record.py:275` 调的是

```python
be = be.attach_cpg_baseline()
```

而 `engine/embodied/body_backend.py` 里这个函数的 docstring 是我自己写的：

> ``"""Attach FlyGym's OWN tripod CPG.  ENGINEERING BASELINE, not the brain."""``

往下一层，这个 CPG 由两个部件构成：

1. **`PreprogrammedSteps`** ← `assets/single_steps_untethered.pkl`。这个文件自己的 meta 写着
   `source: data/3D_pose_alfie/clean_3d_best_ventral_best_side.csv`、`interpolated: True`、
   `smoothed: True`、`window_size: 10`。
   也就是：**从一只真实果蝇腹视 3D 姿态轨迹里抠出来的一个步态周期，插值 + 平滑后固定循环播放。** 这是"动作捕捉回放"，不是神经活动。
2. **`CPGNetwork`** —— 它自己的 docstring：
   > "Euler-integrated network of coupled phase-amplitude oscillators."

   每条腿的状态只有 **(相位 θ, 幅值 r)**。耦合权重 `(phase_biases > 0) * 10.0` 是**手写的三角步态矩阵**，频率 12 Hz、收敛系数 20.0 都是**选的**。没有膜电位、没有脉冲、没有突触。

**所以：这批录像里，腿的动作 = 动作捕捉回放 + 6 个非脉冲振子。一个神经元都没有，连我写的指令都没有（用的是纯 baseline，`grep neural|tier|CommandDecoder|arm run_arena_record.py` 无任何命中）。**

---

## 链 ②「环路里的 `neural_modulated` 臂」——**有真连接组，但只输出 2 个数字**

**这是唯一一条把连接组接进运动的地方**，但它改的是节奏的**参数**，不是节奏本身。

链路（`engine/embodied/loop.py`）：

```
身体接触力 + 关节角 → 腿机械感受器 → 注入电流 nA
      → BANC 连接组电导型点神经元网络（12,000 个）
      → 下行读出群放电率 → 解码 → (speed_scale, turn) 两个旋钮
      → 写进同一个 CPG 的 intrinsic_freqs
```

| 项 | 实测值 |
|---|---|
| tier 神经元数 | **12,000**（BANC 数据集共 153,962；tier 内占数据集突触行 14.58%） |
| tier 内下行神经元 | **196**（数据集 1,327 的 14.8%） |
| tier 内脑侧神经元 | **464 / 110,297 = 0.42%** —— 这个 tier 基本是 VNC + 腿机械感受 |
| 下行总放电率 | 28.63 Hz（参考 38.42 Hz） |
| 解出的指令 | speed_scale 0.778、turn 0.216 |
| 造成的差异（3 seed 配对） | 走慢 **27%**（speed_ratio 0.731）、偏航 **−21.4°**、位移少 31.2 mm |

配对定义是"同 seed、同初始态、同一次 CPG 工厂调用，**唯一差别是两个旋钮的来源**"。

**我自己报告里写死的边界**（`outputs/embodied_body/loop_report.json`）：

- `fingerprint_note`: "**the CPG is FlyGym's demo tripod CPG in BOTH arms; the neural tier modulates the gait, it does not generate it**"
- `rationale`: "**The connectome does NOT generate the gait and no result may be described that way.**"
- `not_measured`: "nothing here shows that the connectome generates the gait, **that the rate->knob map is correct**, or that the fly perceives anything"
- `tier.is_a_selection_not_a_brain: True`

**所以：神经元确实在**影响**身体，但它决定的是"快慢和转向"；步态的节奏来自链 ① 的回放 + 振子。**

---

## 链 ③「站立 / 肌肉」——**唯一真正的"神经 → 肌肉 → 运动"，但它站不住**

**这是另一个模型**，和链 ①② 不是同一个东西：

| 项 | 实测值（本次重新只读加载 `outputs/muscles_six_legs/fruitfly_six_leg_standing.xml` 验证） |
|---|---|
| 模型 | **FlyMimic 肌肉骨骼模型** |
| body / geom | 73 / 72 |
| 执行器 | **90 个 `class="muscle"` 的 Hill 型肌肉**，`ctrlrange [1e-4, 1]` = 肌肉激活度 |
| 关节 / nq | 43 / 49 |
| **自由关节** | **有**（`<freejoint name="thorax_free" />`） |
| 总质量 | **0.002494 模型单位 = 2.49427 mg** |
| 一个体重 | **24.447 力单位** |

驱动链（`run_standing_neural_loop.py`）：

```
关节角 → 该腿机械感受神经元 → 连接组自身的边 → 运动神经元脉冲
      → 每条肌肉自己的运动神经元放电率 → 激活度 → MuJoCo 肌肉 → 身体运动
```

**急性消融实测（阳性、逐腿特异）**：静默某一条腿的全部运动神经元，
LF 姿态 −60%、LM −41%、LH −16%，而**其他五条腿只变 1–3%**。
且受试后的 LF = 1.2317，正好等于单独测的**纯被动平衡值 1.2317** ——
**去掉神经元，腿就掉回被动平衡点；完整时是肌肉把它撑在外面的。** 这是"神经元驱动"该有的结构。

**但它没有站住。`stood = False`，五种配置全部为假**（`outputs/standing_load_reflex.json`）：

| 配置 | stood | 身体承担的体重 | 腿承担的体重 | 着地的腿 |
|---|---|---|---|---|
| 仅角度通道 | False | **0.6018** | 0.3991 | RF, RH, RM |
| LOAD 阻力反射 | False | **0.6783** | 0.3214 | RF, RH, RM |
| LOAD + 角度 | False | 0.5539 | **0.4459**（最好） | RF, RH, RM |
| LOAD，全部腿运动神经元静默 | False | 0.6010 | 0.3990 | 全部六条（瘫下去） |
| LOAD，静默一条腿（LH） | False | 0.5651 | 0.4349 | LF, RF, RH, RM |

**即：55–68% 的体重压在胸/腹/喙上，不在腿上；最好的一次腿也只承担 44.6%。且这个模型不是在走路。**

---

## 所以对"你确保是真实神经元控制的吗"的完整回答

| 你看到的东西 | 驱动它的 | 是真实神经元吗 |
|---|---|---|
| 走路录像、肢体关节角、触觉记录 | 动作捕捉回放 + 6 个非脉冲振子 | **不是。零神经元。** |
| 环路里"神经调制"的行走速度/转向 | BANC 连接组 12,000 神经元 → 2 个旋钮 | **是神经元，但只调参数、不产生步态** |
| 消融后腿姿态的逐腿变化 | 连接组 → 运动神经元 → 肌肉激活度 | **是神经元，且直接驱动肌肉；但站不住、不走** |

**并且"真实神经元"这个说法本身还要再收一档**：即使用 BANC，那也是
**连接组注释数据（组学推断的连接）+ 我写的电导型点神经元方程**。
受体动力学在本项目里自己标为 `ILLUSTRATIVE`，**没有任何一个参数是果蝇实测标定的**；
`nt_pair` 在缓存里**每一行都是 −1**，所以突触计数只有"带宽"意义，不等于兴奋性。
准确说法是 **"连接组结构约束下的、参数未经实测标定的点神经元网络"**，不是"真实神经元"。

---

## 附：我自己犯过并已更正的一处 MIXUP（单位/体重张冠李戴）

两个模型的质量不同：**1.02431 mg（链①②）vs 2.49427 mg（链③）**，差 2.44 倍。
我先前把链③测到的自身接触力 403 力单位写成"= 体重 40 倍"，用的是**另一个模型**的体重常量。
按链③自己的体重 24.447 力单位，**正确倍数是 16.5 倍**。
数字方向不变，倍数已在该文档中更正（`ABLATION_FIXED.zh-CN.md` 第 1 节）。

---

## 交付与复现

```bash
cd /run/media/sensen/Data2/cell_wound_prototype
# 链①：确认录像用的是 baseline、无神经层
grep -n "attach_cpg_baseline\|neural\|tier" run_arena_record.py
# 链②：实测的神经贡献
venv/bin/python -c "import json;d=json.load(open('outputs/embodied_body/loop_report.json'));print(d['contribution'])"
# 链③：站立结果（全部 stood=False）
venv/bin/python -c "import json;[print(r['tag'],r['stood'],r['final']['body_bw']) for r in json.load(open('outputs/standing_load_reflex.json'))['runs']]"
```

绝对路径：

- `/run/media/sensen/Data2/cell_wound_prototype/run_arena_record.py`（链①：第 275 行）
- `/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/body_backend.py`（`attach_cpg_baseline` 与 `attach_action_source`）
- `/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/loop.py`（链②：第 426–445 行）
- `/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/adapters.py`（`CommandDecoder` 第 902 行、`CommandedTripodCPG` 第 1040 行）
- `/run/media/sensen/Data2/cell_wound_prototype/run_standing_neural_loop.py`（链③）
- `/run/media/sensen/Data2/cell_wound_prototype/outputs/muscles_six_legs/fruitfly_six_leg_standing.xml`（链③模型）
- `/run/media/sensen/Data2/cell_wound_prototype/outputs/embodied_body/loop_report.json`（链②数字）
- `/run/media/sensen/Data2/cell_wound_prototype/outputs/standing_load_reflex.json`（链③数字）
- `/run/media/sensen/Data2/cell_wound_prototype/outputs/standing_shots/`（链③站立渲染图）
- `/run/media/sensen/Data2/cell_wound_prototype/venv_body/lib/python3.12/site-packages/flygym_demo/complex_terrain/preprogrammed.py`（链①回放数据）

**这是手工搭建的研究原型，不是成品；全部"真值"均来自模拟器，任何一条链都没有对真实果蝇做过测量。**
