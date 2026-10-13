# 有来源的感觉参数：数据集 + 文献两条链，以及"有来源"之后究竟改变了什么

**手工研究原型，不是成品。** 本轮走你选的路（为感觉前端找**来源**），分两段：
(A) 数据集自身的官方标注；(B) 已发表文献的量化调谐。全部参数经**闸门**
`tools_apply_sourced_sensory_config.py`——**没有 citation + URL 的参数会被拒绝，不会被填上**。

---

## 1. 一句话结论

- **有来源的归一化器把站姿显著变好**：六足六腿全部落地、末段高度 1.1043 mm（发布中性高度 1.1611 mm 的 95%）、
  漂移 0.0098 mm、无躯干触地——**只差 `quiet` 一条**（末段抖动 0.0236 mm > 0.02 阈值）。这是本项目迄今最好的站姿。
- **但"没有姿态参考"这条性质没有被来源化改变**：所有构型的中位相关仍是 **+0.74**、恢复侧 **0/36**。
  这与我事先做的**可达性扫描**一致（把主导通道权重改 16 倍也去不掉它）。
- **文献反而推翻了上一轮的"时间尺度修复"**：用有来源归一化器后**最快的 5 ms 反而最好**（6 爪），
  文献窗口内的 13.42/30 ms 更差（4/5 爪）⇒ 那个"放慢就站住"是**未来源归一化器的产物**。
- **力归一化器仍无来源**（闸门拒绝，见 §5）；第四个代理仍在查钟形感受器的载荷范围。

---

## 2. 数据集来源：通道里是哪些细胞，以及三个硬事实

数据集（`data/banc888/banc_888_meta.feather`，188,508 行）对每个腿部感觉细胞都有官方功能标注。
逐通道对照（详见 `outputs/model_branches/banc_sensory_annotation/`）：

| 通道 | 细胞数 | 数据集功能标注 | 本项目喂的量 | 判定 |
|---|---|---|---|---|
| claw_position | 162 | **position** 162/162 | 胫节角偏差（有符号） | 一致 |
| hook_direction | 142 | **direction** 142/142 | 胫节角速度（有符号） | 一致 |
| club_vibration | 342 | **vibro_tactile** 342/342 | 恒为 0 | 一致 |
| hairplate_angle | 234 | **joint_angle** 234/234 | 6 铰链平均偏差 | 量一致，**关节错**（见 §3） |
| campaniform_strain | 53 | **mechanical_strain** 52 + 1 | 爪-地接触力（按单腿承重份额） | **部分**：应变 ≠ 地面接触力 |
| bristle_touch | 2,979 | **仅 12 个有标签** | 足接触 0/1 | 数据集沉默 |
| multidendritic_noxious | 36 | **putative_nociception** 36 | 恒为 0 | 一致 |

另外三条实测硬事实：
1. **覆盖缺口**：1,115 个带本体感觉标注的腿部细胞里 **182 个（16%）**落在本前端从不读取的亚型
   （orphan 124、泛称 chordotonal 46、bilateral campaniform 9），其中 **60 个标为 joint_angle**。
2. **负载通道其实只存在于前腿**：tier 每腿 campaniform 细胞数 lf=28 rf=24 lm=2 rm=2 lh=2 rh=2，
   而消融实验证明**正是它单独维持了全部运动驱动**。
3. **毛板带编号基部组**（`CoHP8` 16、`CoHP3` 6、`CoHP4` 4，前腿），是将来按附着关节拆通道的把手；
   三个弦音器官组的**胞体位置沿腿轴分不开**（与"同一器官的三个组"相符）。

---

## 3. 文献来源：三条映射判定 + 一条纠正 + 一条证伪

| 代理 | 关键结论 | 判定 |
|---|---|---|
| FeCO（claw/hook/club） | 三者是**同一个器官（股骨近端 femoral chordotonal organ）的三个亚群**，都转导**股骨-胫节**关节；名字指 VNC 内轴突分枝形状 | — |
| ↳ claw ← 胫节角 | 强直（tonic）、非适应、按角度线性；屈/伸由**不同细胞**携带 | **支持** |
| ↳ hook ← 胫节角速度 | 相位性、方向选择（屈曲 DSI 0.811±0.028）；方向必须带符号 | **支持（量）**，线性未证实（只测 100–800°/s） |
| ↳ club ← 跨关节均值 | club = 双向股骨-胫节运动 **+ 胫节振动**（100–2000 Hz，400/800 Hz 峰值）；Lee 2025 认为以外感受为主 | **证伪**（所以我们把它留 0 是对的） |
| 毛板 | 唯一定量果蝇毛板是 **CxHP8**：编码**基部旋转/内收**，响应跨度 **149°=2.601 rad**，是**极限检测器**；"多关节积分"没有任何已发表受体支持 | **纠正**：`hairplate_angle` 原来喂错了关节 |
| 增益 | **没有任何果蝇 FeCO 或毛板的发放增益/阈值被发表**（全部是钙成像；唯一关系是"claw 阈值 >10× club"） | 归一化器只能由**已发表的刺激范围**派生 |
| 时间尺度 | 果蝇感觉中继 ~3–6 ms（Karashchuk 2024；Tuthill & Wilson 2016）、运动神经元→力峰值 **~30 ms**（Azevedo 2020）；无果蝇 FTiR 环路潜伏期 | 驱动滤波的**文献窗口 6–30 ms** |

**模型几何实测**（`tools_flybody_joint_geometry.py`，纯前向运动学）：

| 量 | 实测 | 文献 |
|---|---|---|
| 坐姿股骨-胫节解剖角 | **89.95°**（84.8–93.6°） | 静息/最小活动角 ~90°（Mamiya 2018）/ ~80°（Dallmann 2025） |
| 模型解剖角范围 | 10.5–171.0° | 实验扫过 18–180°（Mamiya 2018） |
| 铰链→解剖角梯度 | −57.2…−57.3 °/rad | —（几何量，非文献） |
| 基部摆动（coxa_abduct）范围 | LF 40–75°、LM 69–85°、LH 105–161° | CxHP8 基部旋转跨度 149°（Pratt 2026） |

⇒ **项目一直用的参考（坐姿）与文献的静息参考角一致**（差 0.04 rad），所以 claw 通道的**参考不需要改**；
但**毛板的关节确实错了**，已按文献改为**基部摆动**。

---

## 4. 有来源之后：跑了什么，结果如何

闸门产出的有来源参数（`outputs/model_branches/flybody_loop/sourced_sensory_config.json`）：

| 参数 | 有来源取值 | 来源 |
|---|---|---|
| `reference_angle_rad` | **1.4137 rad**（= 18–180° 跨度的一半） | Mamiya 2018 / Dallmann 2025 |
| `reference_velocity_rad_s` | **6.1087 rad/s**（= 100–800°/s 跨度的一半） | Mamiya 2018 |
| 毛板量 | **基部摆动偏差**（角色映射，声明） | Pratt 2026 |
| `reference_force` | **拒绝**（无引用） | — |
| `club_quantity` | `zero`（有来源：振动通道） | Mamiya 2018 / Lee 2025 |
| `filter_tau_s` | 13.42 ms（6–30 ms 窗口的几何均值，规则声明） | Karashchuk 2024 / Azevedo 2020 |

闭环结果（同一 tier、连接组、通道划分、肌肉池、解码器、时钟、同一躯干）：

| 配置 | 驱动 | 中位 r | 恢复侧 | 末段高度 | 爪/腿 | 漂移 | 触地体 | 判据 |
|---|---|---|---|---|---|---|---|---|
| 发布配置（历史角色） | 0.1014 | +0.761 | **0/36** | 1.0148±0.0301 | 5 / 3 | 0.0064 | 无 | ✗ `feet_down`,`quiet` |
| **有来源归一化器** | 0.0908 | +0.744 | **0/36** | **1.1043±0.0236** | **6 / 6** | 0.0098 | 无 | ✗ **仅 `quiet`** |
| 有来源 + 文献角色（毛板→基部） | 0.0908 | +0.744 | **0/36** | 1.1043±0.0236 | 6 / 6 | 0.0098 | 无 | ✗ 仅 `quiet` |
| 发布配置 + 同一扰动 | 0.1028 | +0.745 | **0/36** | 0.9882±0.0376 | 4 / 4 | 0.0106 | 无 | ✗ |
| 有来源 + 同一扰动 | 0.0824 | +0.742 | **0/36** | 1.1167±0.0264 | 4 / 3 | 0.0095 | 无 | ✗ |

（扰动 = 世界系力加在胸部、本躯干自重的 50%、5 ms、t=150 ms，所有臂完全相同。）

**读法**：
1. 有来源归一化器把**六足六腿全部落地**、高度提到发布中性的 95%、无躯干触地、几乎不漂移，
   **只差"末段要安静"这一条**（0.0236 vs 0.02，差 18%）。
2. **文献角色测得为零效应**——毛板通道只占总电流 **1.0%**（有来源配置下），改它的关节改变不了回路。
   这与"纠正本身是对的、但该通道太小"一致。
3. **所有构型耦合仍是 +0.74、恢复侧 0/36** ⇒ **参数来源化不装姿态参考**（与 §2 的可达性扫描一致）。
4. **τ 的结论被反转**：有来源归一化器下，5 ms（低于文献窗口下界）给出 **6 爪**，
   13.42 ms（有来源规则值）给 4 爪，30 ms（窗口上界）给 5 爪。
   ⇒ 上一轮"放慢驱动就站住"是**未来源归一化器**的产物；**文献窗口并不能决定 τ**。

---

## 5. 仍然缺的来源

**力归一化器**：闸门明确拒绝（`refused_without_source: ['force_normaliser_channel']`），
因为还没有拿到钟形感受器的已发表载荷范围。而 §2 已证明**这个通道单独维持了全部运动驱动**、
且可达性扫描显示它的权重是唯一显著影响站姿的量。第四个研究代理仍在查这一项
（钟形感受器的载荷-发放关系、阈值、以及以体重分数表达的动态范围）。

---

## 6. 不支持的主张

- 不支持"有来源参数=标定"：文献给的是**实验用过的刺激范围**，不是调谐增益；
  规则"取跨度的一半"是**本项目的**，已写进产物。特别是**没有任何果蝇 FeCO/毛板的发放增益被发表**。
- 不支持把 20 ms 说成"文献规定的"：文献里**不存在**"滤波不得快于反射环路"这条规则，
  也没有果蝇 FTiR 环路潜伏期；窗口 6–30 ms 是本项目对已发表分量潜伏期的读法。
- 不支持"六足落地=站住"：五条判据里 `quiet` 仍不通过（差 18%）。
- 不支持"毛板通道的关节纠正改变了回路"：实测零效应（该通道占 1.0% 电流）。
- 不支持用胞体位置推断通道的附着关节（三组投影不可分）。
- 不支持把数据集的功能标注当实测调谐（那是数据集自己的标签）。
- 不支持任何行为主张。

---

## 7. 产物、复现与图

```bash
cd /run/media/sensen/Data2/cell_wound_prototype
./venv/bin/python tools_banc_sensory_annotation_audit.py          # 数据集标注审计（pandas 解释器）
./venv_body/bin/python tools_flybody_joint_geometry.py            # 解剖角/基部摆动的模型实测
./venv_body/bin/python tools_apply_sourced_sensory_config.py      # 闸门：没引用就拒绝
./venv_body/bin/python run_flybody_sourced_sensory.py            # 有来源配置的闭环 + τ 窗口
./venv_body/bin/python run_flybody_normaliser_sensitivity.py      # 可达性扫描（16× 力权重）
./venv_body/bin/python run_branch_regressions.py                  # 回归套件（170 项）
```

- 文献存档：`sources/literature/FeCO_claw_hook_club_literature_report.md`、
  `sources/literature/hairplate_literature_report.md`、`sources/literature/lit_report_motor_filter_timescale.md`
- 引用参数：`data/literature_sensory_parameters.json`（每个数字一条 citation + URL）
- 派生参数：`outputs/model_branches/flybody_loop/sourced_sensory_config.json`
- 结果：`outputs/model_branches/flybody_loop/flybody_sourced_sensory_run.json`、
  `outputs/model_branches/flybody_loop/flybody_joint_geometry.json`、
  `outputs/model_branches/banc_sensory_annotation/banc_sensory_annotation.json` / `.png`
- 本文件：`FLYBODY_SOURCED_SENSORY.zh-CN.md`
