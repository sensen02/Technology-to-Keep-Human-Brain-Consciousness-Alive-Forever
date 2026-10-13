# 单位语义更正：长度单位是**毫米**，我上一阶段的 D1「修复」方向反了

**手工实施并测试的更正，不是成品。** 规范模型未被修改
（SHA256 仍为 `65d3904dce9802f46997a92cb3f80cd13c032f847c1e1f6710694d65cfdd73b6`，程序内断言校验）。

---

## 优先勘误：本报告也有推断过界（长度 mm 结论保留）

- 后文“等效加速度差是重力的 0.075%”用了含角加速度的广义向量范数，该比较无效；广义力范数也混合力/力矩，只能作为给定坐标下的数值量，不能当量纲统一的力比例。
- 0.05 s 对照只支持该试验中下限影响较小、90% 高度穿越时间相同，不证明长期闭环无影响或下限永远不是障碍。
- act=1 的总肌肉力包含被动力，不等同于纯主动力或跨所有长度速度的最大力。“12%”只是在指定构型下零/满激活总肌肉力范数之比。
- 质量设为克只是显式标注的推断，不能排除所有其他质量标度。质量单位从 mg 改为 g 相差 1000 倍，并非后文所写 1e6；混合长度变化时必须逐项换算。
- 旧 cm/s² 标签中的 9801 应改标签为 mm/s²，而非再把原始模型数值除以10。已换算的旧 `*_mm` 字段才需除以10。
- 受控 A/B 只覆盖测试过的闭环，不证明所有消费者或审计的物理不变。外部生物尺寸范围未经原文核验，不作为精确校准证据。

## 0. 一句话

我（第39轮）曾断言"模型长度单位是**厘米**"，并据此把项目历史报告"更正"了 10 倍，
例如声称站立胸高 2.04535 "其实是 20.45 mm"。**这个断言是错的，现予撤回。**
长度单位是**毫米**：胸高 2.04535 = **2.04535 mm**，体长 2.6 mm，mesh 跨度 5.78 mm——
全都符合果蝇；而厘米读法会让这只果蝇**大 8–10 倍**，物理上不可能是 *Drosophila*。

**原来的标签（`PENETRATION_MM = -0.05`）是对的，我第39轮的"更正"（改成 -0.5）才是错的。**

```diff
- MODEL_LENGTH_UNIT_NAME = "cm"      # 第39轮，错
- MM_PER_MODEL_LENGTH   = 10.0
+ MODEL_LENGTH_UNIT_NAME = "mm"      # 本更正
+ MM_PER_MODEL_LENGTH   = 1.0
```

---

## 1. 五条独立证据（我自己逐条复核过，不只是采信审计）

| # | 证据 | 位置 |
|---|---|---|
| 1 | **71 个 `<mesh ... scale="1000 1000 1000"/>`**，而缓存 STL 原始坐标是**米**（`Thorax.stl` 跨度 1.069e-3 m）→ 1.0692 模型单位 = 1.0692 **mm** | 规范 XML 第 46–116 行 |
| 2 | **几何交叉验证**：70 对 mesh 跨度 / body 偏移的比值中位 **0.9408**（必须 O(1)）；cm/mm 错配会给出 9.41 或 0.094 | 编译后资产 |
| 3 | **上游自己写 mm**：`flygym/assets/model/flybody/mujoco_globals.yaml:9` = `gravity: ['0','0','-9810']  # mm/s^2 (source XML is cm: -981 cm/s^2 x10 -> mm)` | 上游包内 |
| 4 | **上游代码注释**：`compose/fly/base_fly.py:166-168`（"we simulate length in mm, not m…SCALE = 1000"）、`simulation.py:188`、`compose/fly/flybody.py:106` | 上游包内 |
| 5 | **本项目生成器自己写的**：规范 XML 头部注释 "the keyframe places the thorax at **z=2.0453 mm**"；`tools_stand_and_ground.py:256` | 本项目 |

**重力自洽性**：`9801 mm/s² = 9.801 m/s²` ✅；若按 cm 则是 `98.01 m/s²` ❌。
旁证：同包 `flybody/fruitfly.xml` 用 `-981`（厘米族），`musculoskeletal/best_combined_arm_damping_stiff_cvt3.xml` 用 `-9801`（毫米族），倍数正好 10。

**生物学交叉检验**：体长 2.60 mm、胸 mesh 1.069×0.947×1.085 mm、extent 5.78 mm，
对照真实 *D. melanogaster*（体长 2.5–3 mm、胸长 1.0–1.2 mm、腿展 5–6 mm）**全部落在范围内**。

---

## 2. 关键约束：修标签**不允许**移动物理——已用受控 A/B 证明

这是我这次最小心的地方。第39轮的做法是：声明 cm（因子 10），同时把
`SEATING_PENETRATION_MM` 从 -0.05 改成 -0.5，使送进 `seat_root` 的乘积仍是 -0.05。
如果我只是把因子改回 1.0 而保留 -0.5，**坐姿就会下沉 10 倍，物理随之改变**——那是第二个更严重的 bug。

因此本更正把**申报值改回实际施加值 -0.05**，使
`SEATING_PENETRATION_MODEL = -0.05` **两个约定下完全相同**。

**受控 A/B 证据**（`run_unit_fix_ab.py`，同一份闭环代码、同一初态，只改这两个常数）：

| 约定 | 因子 | 申报 mm | `seat_root` 目标 |
|---|---:|---:|---:|
| 旧（第39轮） | 10.0 | -0.5 | **-0.05** |
| 新（本更正） | 1.0 | -0.05 | **-0.05** |

- **22 个轨迹/控制/脉冲数组全部逐位相同** → 物理没有移动；
- 报告的毫米标签比值 **恰好 10.0**（17.236 "mm" → 1.7236 mm）→ 标签被修正。

产物：`/run/media/sensen/Data2/cell_wound_prototype/outputs/unit_semantics/ab_physics_identity.json`

**回归自测 12 项全通过**（`test_canonical_units.py`），其中包含把这次教训钉死的锚点：

- 长度单位是 mm 且因子**不得回到 10.0**；
- `SEATING_PENETRATION_MODEL` **必须等于 -0.05**（物理锚点）；
- 门控坐姿仍然通过，坐姿高度 **2.0449**（ttarsus-only、无身体穿地）；
- 71 个 mesh 缩放与 XML 头部 "z=2.0453 mm" 作为**证据本身**被测试固定，
  将来资产若被重新生成会失败而不是让结论悄悄漂移；
- `engine/embodied/body_backend.py` 的常数**必须与 canonical 一致**。

---

## 3. 质量单位：**未建立**，我不替项目选

`<inertial mass=...>` 共 71 个，和 = `0.002494271478`，**`density` 出现 0 次** → 资产没有声明单位。

项目内三处**互相矛盾**的声明：

| 声明 | 位置 | 总质量 |
|---|---|---|
| kg | 第39轮我的 docstring（该句自身 `0.002494271478 kg = 2.494 mg` 就差 1000 倍） | 2.494 g |
| g | `tools_stand_and_ground.py:704`、`diag_leg_load_test.py:38-45`、`LEG_TUNING.zh-CN.md:117` | 2.494 mg |
| 1 单位 = 1 mg | 第39轮的 `MODEL_MASS_UNIT_KG = 1e-6` | 2.494e-9 kg |

**几何约束**：全部 mesh 体积和 = 1.4679 mm³（**上界**）⇒ kg 给 1.7e6 kg/m³（排除）、
**g 给 1699 kg/m³（唯一不荒谬）**、mg 给 1.70 kg/m³（仅比空气重 0.5，排除）。
诚实标注：体积是上界 ⇒ 真实密度 >1699，所以"克"只是**唯一不荒谬**，不是恰好正确。

**因此**：`MODEL_MASS_UNIT_KG` 改为 1e-3（克），但新增
`MODEL_MASS_UNIT_STATUS = "INFERRED_FROM_MESH_VOLUME_DENSITY_ARGUMENT_NOT_ESTABLISHED"`，
并且在 `write_result` 的默认 `units` 里明确写 **INFERRED (NOT established)**。
**不要再把 2.494 mg 当作已确立的 SI 质量引用。**

---

## 4. 改动波及面（也检查了修复代价）

| 位置 | 改动 | 是否影响物理 |
|---|---|---|
| `engine/canonical_physics.py` | 单位常数 + 撤回 docstring + 坐姿申报值 | **否**（A/B 证明逐位相同） |
| `engine/embodied/body_backend.py` | 常数改为 mm/1.0 | **是，且未重测**（见下） |
| `run_fix_round39_checks.py:145` 的门控 | 未改 | **否**——已证明**尺度不变**：旧路径 `(L×10)/10`，新路径 `(L×1)/1`，两者同为 `L` |

`body_backend.py` 值得单独说：**它自己就自相矛盾**——第 42–48 行说厘米，
而同一文件的 `GEOMETRY_NOTE`（第 57–59 行）说"模型单位是毫米（gravity [0,0,-9810] mm/s²）"。
同一文件两种说法，其中一种与全部证据一致。已统一为毫米。

**并且我明确不做的事**：`body_backend.summarise_walk()` 的工程门限
（`gate_upright` 0.3–3.0、`gate_moved` ≥20）在旧因子下是被 10 倍膨胀后的量喂进去的，
改因子**会改变某段记录命中哪个门限**。这个**行为后果我没有重测**，
已在函数 docstring 里写明"这些门限量测结论必须重新推导后才能与任何东西比较"。

---

## 5. 连带影响：哪些数字需要重新解读

**只是标签错、物理没变**的部分（可直接换算，无需重跑）：

| 旧标签（cm 约定） | 正确值 |
|---|---|
| 站立胸高 "20.449 mm" | **2.0449 mm** |
| 历史 "2.045 mm" | **正确**（原报告本来就是对的） |
| 穿透 "0.5 mm" | **0.05 mm** |
| 身体穿地 "4.3 mm / 179 接触" | **0.43 mm** |
| 肌肉力单位 | mm + g ⇒ **1 µN**，与项目自己的 `diag_leg_load_test.py:56`（µN·mm/rad）、`LEG_TUNING.zh-CN.md:82,86`（0.4 µN·mm/rad、4.07 µN）**吻合**；第39轮的 mg 读法**差 1e6 倍** |

**必须重新推导、不能只换算的部分**：

1. `body_backend.summarise_walk()` 的所有门限判据（见 §4）；
2. 任何**真的做了换算**的路径——纯标签错不改物理，但旧 `qpos[2] = root_z_mm/10.0`
   那类路径改的是物理（该路径早已被门控拒绝）；
3. **并行审计 `muscle_floor_audit.json` 里的 `*_mm` 标签**：它在我修正**之前**运行，
   因此其 `root_z_mm: 20.4488`、`worst_floor_penetration_mm: -0.4999`、
   `gravity_for_scale_cm_per_s2: 9801` 标签都偏 10 倍。
   **但它的模型单位值 `2.044880` 与我的坐姿测试 `2.0449` 完全一致**——
   这是对本更正的一次独立交叉验证。

---

## 6. 并行独立审计的附带结论：最低肌肉控制是**数值细节，不是障碍**

来自 `/run/media/sensen/Data2/cell_wound_prototype/outputs/body_physics_audit/muscle_floor_audit.json`
（模型单位值有效；`*_mm` 标签见 §5 第 3 条）：

- **90 个执行器全部**是 `mjDYN_MUSCLE`/`mjGAIN_MUSCLE`/`mjBIAS_MUSCLE`；
  `dynprm` 唯一行给出 **`tau_act = 0.0001`、`tau_deact = 0.0004`**，
  即**激活动力学是 0.1–0.4 ms 量级**，相对 5 ms 命令时钟**近似瞬时**。
  （顺带说明：先前被引用的"肌肉 50% 阶跃响应 88–110 ms"**不可能是肌肉激活**。）
- 那个 `act_dot = 2.0` 的**来源已被隔离**：native（下限 1e-4）下 90 个全为 2.0，
  内存派生模型 zero_floor（下限 0）下 90 个**全为 0.0** → **完全由控制下限产生**。
- **后果可忽略**：下限造成的稳态激活恰为 `1e-4`（满量程的 0.01%）；
  广义力差是全部执行器广义力的 **0.42%**；等效自由加速度差是重力的 **0.075%**；
  自由积分瘫倒时间 native 与 zero_floor **完全相同（0.0149 s）**。
- 被动力 ~ 主动力的 **12%**（L2 比 0.1236），最大被动力 -57.15（`LHF_trochanter_flexor_a`）。
- 派生模型**仅在内存中构造**，规范 XML 未被写回。

**判定**：XML 的 1e-4 下限应**保留**（它是模型的声明设定），
它使"神经命令为零"不完全等于"零激活"，这一点继续如实报告；
但它**不构成机械或控制上的障碍**，因此**不值得**为它做更多派生模型实验。

---

## 7. 精确运行命令

工作目录：`/run/media/sensen/Data2/cell_wound_prototype`

```bash
# 单位语义回归自测（12 项）
/run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python \
  /run/media/sensen/Data2/cell_wound_prototype/test_canonical_units.py

# 受控 A/B：旧约定 vs 新约定，证明物理逐位不变
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python \
  /run/media/sensen/Data2/cell_wound_prototype/run_unit_fix_ab.py --convention legacy    --out /tmp/ab_legacy
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python \
  /run/media/sensen/Data2/cell_wound_prototype/run_unit_fix_ab.py --convention corrected --out /tmp/ab_corrected
```

---

## 8. 诚实交代：本轮我修掉的自己的三个错误

1. **D1 方向反了 10 倍**（本文件主题）。我把"厘米"当成结论，还据此写了 11 份历史文档的更正脚注——
   那些脚注本身也需要按本文件重新解读。
2. **闭环脚本里的 `UnboundLocalError`**：`total_weight_model` 写在了 `prepare_body()` **之前**，
   只有真跑才暴露（`ast.parse` 查不出来）。已修。
3. **一次假通过**：我拿"运行失败后未被重写的旧 npz"和它自己比较，输出了 `BIT-IDENTICAL`。
   形状全不一致才暴露了比较对象错误（那份旧文件其实被锁演示的 `--duration 0.30` 覆盖过）。
   **已改为受控 A/B**，并断言输出文件存在，不再可能自己和自己比。

## 9. 仍然未解决

- 质量 SI 换算**未建立**（§3）；
- `body_backend` 门限的行为后果**未重测**（§4）；
- 早期 `outputs/electrode_payload/*/vision*.json` 里 `touch_force_uN = 1674.17`
  （约为单腿体重 4.07 µN 的 411 倍）**未追**；
- mesh 缓存版本未校验（项目无上游 mesh hash）；
- 两个外部 PDF 未能打开（工具拒绝 application/pdf），所以真实果蝇质量/胸长的
  外部查证**只到链接级别**，未读全文。
- 单位更正**不会**让果蝇站稳、走路或让闭环变成自主——它只让数字的含义变正确。
