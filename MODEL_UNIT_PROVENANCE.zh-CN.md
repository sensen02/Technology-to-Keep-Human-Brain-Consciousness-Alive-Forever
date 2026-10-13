# 规范果蝇身体模型的物理单位与质量/力量级溯源（MODEL_UNIT_PROVENANCE）

> ## ⚠️ 手工研究，不是成品
>
> 本文是**手工研究**，由我在本机**手工检索、手工核对**完成的一次**只读**溯源调查。
> 不是经过评审、回归或版本化的产品模块，也不是「任务完成」的交付结论。
>
> - **没有修改任何项目文件**（本次只新增本文与 `outputs/neural_muscle_closure/model_unit_sources.json` 两个文件）。
> - **没有编辑模型**，**没有跑 episode**，**没有跑长积分**。唯一一次加载 MuJoCo 只是为了读出编译后的 `nq/nv/nu/na`、`gravity`、`body_mass`（不 step）。
> - 本文**不重算任何项目历史数字**，只回答一个问题：**这个模型的长度/质量/时间/力单位，按上游文档**应该**是什么**。
> - 凡本文明写「未解决」的，就是**没有**解决。

**调查对象**（只读，未修改）

| 项 | 值 |
|---|---|
| 规范资产 | `/run/media/sensen/Data2/cell_wound_prototype/outputs/muscles_six_legs/fruitfly_six_leg_standing.xml` |
| SHA256（本次已核对） | `65d3904dce9802f46997a92cb3f80cd13c032f847c1e1f6710694d65cfdd73b6` |
| 路径来源 | `/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/zero_floor/model_manifest.json` 的 `canonical_xml` / `canonical_sha256` 字段 |
| 分支资产 | `/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/zero_floor/fruitfly_zero_floor.xml`（SHA256 `b4f6b375…`，把 90 个肌肉 actuator 的 `ctrlrange` 下界 0.0001 改成 0） |

---

## 0. 结论速览（先给答案）

| # | 问题 | 结论 | 强度 |
|---|---|---|---|
| 1 | 长度单位？ | **毫米 mm** | **有文档声明**（上游 OpenSim 源 + flygym 自述） |
| 2 | 质量单位？ | **千克 kg** | **有上游文档支撑**（见 §3，本轮新证据） |
| 3 | 时间单位？ | **秒 s** | 有文档（SI 源 + 无任何字段反对） |
| 4 | 力单位？ | **毫牛 mN** = kg·mm/s² | **推导**（由 1+2+3 得出，非声明） |
| 5 | 重力 9801 自洽吗？ | **自洽**：9801 mm/s² = 9.801 m/s² | 已确立 |
| 6 | 总质量 | `0.002494271478` 模型单位 = **2.494 mg** | 已确立（数值）；单位归属见 §3 |
| 7 | 一个体重 | **24.4464 模型力单位 = 24.4464 mN** | 推导 |
| 8 | 存在自洽 SI 读法吗？ | **存在**（mm–kg–s，力 mN） | **是** |
| 9 | 项目一贯说法对吗？ | **不对**：`engine/canonical_physics.py:88` 的 `MODEL_MASS_UNIT_KG = 1e-3`（克）**大了 1000 倍** | 本轮更正 |

**一句话**：这个模型是 **FlyMimic** 的肌肉骨骼 MJCF（OpenSim → MyoConverter → MuJoCo），它的**祖先文件明确声明 `length_units = m`、`force_units = N`**，即 **SI**；转成毫米时**几何 ×1000，而质量和时间数值原封不动**。因此资产的单位是 **毫米 + 千克 + 秒**，力就是 **毫牛**。项目此前从 mesh 体积反推的「克」是**推论，且在 kg 读法成立时不成立**。

---

## 1. 方法与边界

- 本地检索：项目目录、`venv_body` 内 `flygym` 2.1.0 包源码、`/home/sensen/.cache/flygym_assets/`。
- 网络检索：FlyMimic 仓库与其 OpenSim 源文件、arXiv 论文全文、MyoConverter 文档、NeuroMechFly 文档站。
- 只读加载 MuJoCo 一次（`venv_body/bin/python` + `mujoco` 3.9.0，`OMP_NUM_THREADS=1`，单次 < 60 s，**不 step**），用途仅为读出编译期常量与 `body_mass`。
- 两个上游文件只在 `/tmp` 落地（`/tmp/best_combined.osim`、`/tmp/flymimic_atds.xml`），**未写入项目**。

---

## 2. 模型出处（已查实）

规范资产是 **FlyMimic** 的肌肉骨骼模型，不是 flygym 的 FlyBody，也不是 NeuroMechFly 的「行走」模型。证据链每一环都有路径：

```
[1] flygym 2.1.0 内置的上游 MJCF
    /run/media/sensen/Data2/cell_wound_prototype/venv_body/lib/python3.12/site-packages/
      flygym/assets/model/musculoskeletal/best_combined_arm_damping_stiff_cvt3.xml
        flygym/compose/fly/musculoskeletal.py:93-105
          DEFAULT_MUSCULOSKELETAL_XML = (
              MUSCULOSKELETAL_MODEL_DIR / "best_combined_arm_damping_stiff_cvt3.xml")

[2] flygym 自己写的出处（musculoskeletal.py:160-177）
      Ozdil, P. G., et al. (2026). Musculoskeletal simulation of limb movement
      biomechanics in Drosophila melanogaster. ICLR 2026. arXiv:2509.06426
      Source code for the original model: https://github.com/gizemozd/FlyMimic

[3] 关键：flygym 对该模型【不做任何缩放】
      class MusculoskeletalFly(BaseCompositionElement):     ← 直接继承基类，不是 BaseFly
      grep "SCALE" musculoskeletal.py → 无结果
      → 也就是说，这个资产的单位由 FlyMimic 决定，flygym 不会替它换算。

[4] FlyMimic 仓库里的同一个文件（本轮实际下载核对）
      https://github.com/gizemozd/FlyMimic/blob/main/flymimic/assets/models/
        best_combined_arm_damping_stiff_cvt3.xml         （49670 字节）
      与规范资产逐项一致：gravity 9801、timestep 1e-4、
      statistic extent="5.78307"、71 个 mesh scale="1000 1000 1000"。
      → 规范资产就是这份文件，加上项目自己的地板/keyframe 与肌肉修复
        （规范资产自己的头部注释已说明）。

[5] FlyMimic 的 OpenSim 源（决定性文件）
      https://github.com/gizemozd/FlyMimic/blob/main/flymimic/assets/models/opensim/
        best_combined.osim                                （483413 字节）
```

---

## 3. 决定性证据：上游 OpenSim 源**明确声明**了 SI 单位

本轮最重要的发现。`best_combined.osim` 里逐字写着（行号为下载文件的行号）：

```
osim:17: <!--Acceleration due to gravity, expressed in ground.-->
osim:18: <gravity>0 -9806.6499999999996 0</gravity>
osim:23: <!--Units for all lengths.-->
osim:24: <length_units>m</length_units>
osim:25: <!--Units for all forces.-->
osim:26: <force_units>N</force_units>
```

**读法**：

- `length_units = m` → 源模型长度是**米**。
- `force_units = N` → 源模型力的单位是**牛顿**。在「长度 m + 力 N」下，**质量只能是千克 kg**（N = kg·m/s²）。这是**声明**，不是推论。
- `<gravity>0 -9806.6499999999996 0</gravity>` → 9.80665 m/s²，沿 −Y 轴（OpenSim 的 Y 轴是竖直方向）。这也印证了源模型是标准 SI。

### 3.1 质量数值在转换中原封不动

把 71 个 `<mass>` 相加：

```
osim  BodySet 质量之和 = 0.0024942719507646226   （71 个）
MJCF  inertial 质量之和 = 0.002494271478          （71 个）
```

**两者在 6 位有效数字上相同**（差值 4.7e-10 kg，是 MJCF 把质量圆整到 6 位小数造成的，不是换算）。既然 osim 的质量按 SI 是千克，**MJCF 里同一批数字也必须是千克**。

正是这个「质量数值不变」的现象，正是 flygym 自己写下的换算原则：

```
venv_body/.../flygym/flybody/parse_flybody.py:6-11
# Unit-conversion factors for flybody XML attributes, converting the source model
# (authored in cm) to FlyGym's mm convention: length x10, mass and time unchanged.
…
units_mapping = { "pos": 10, "size": 10, … }        ← 里面【没有】mass
```

即 **flygym 的官方约定：改长度，不动质量与时间。**

### 3.2 惯量数值也不变 → 三重自洽

OpenSim `<inertia>` 与 MJCF `diaginertia` 的**数值逐位相同**（只是对角元顺序被重排）：

```
osim inertia[0] = 2.5427345511451568e-05 1.3331186548450451e-05 2.0053536743222668e-05
mjcf diaginertia = 2.54273e-05 2.00535e-05 1.33312e-05      ← 同一组数
```

长度单位 m → mm 时，惯量单位随之从 kg·m² 变成 kg·mm²，**数值可以不变而物理不变**。这是「几何 ×1000、质量不动」这条转换的**指纹**。

### 3.3 独立几何自洽检查

对 71 个 body 算回转半径 √(I/m)，用模型自己的长度单位表达：

```
√(I/m):  min 0.0234   中位数 0.0871   max 0.707     ← 作为「毫米」是合理的果蝇节段回转半径
OpenSim <location> 模长（米）: 中位数 0.15   max 0.6069
```

OpenSim 里 body 位置是 0.15 米量级（换算后 150 mm，正是模型坐标系里的节段距离），而 mesh `scale="1000"` 把米制 STL 放大成毫米几何。

> ✅ **长度 mm、质量 kg、惯量 kg·mm² 三者在同一份文件里同时自洽。** 这是本轮最强的单一证据。

### 3.4 重力 9801 的来源

源文件写 `-9806.65`（m/s²），资产写 `9801`（mm/s²）：

- 9806.65 × 1000 = 9806650 → 显然不是直接 ×1000 得到的 9801；
- 9801 是「9.801 m/s²」的毫米写法，与 9.80665 相差 **0.058%**；
- 无论哪种取整，**9801 mm/s² 都是毫米族的标准重力**；按米读则是 9801 m/s²（荒谬），按厘米读则是 98.01 m/s²（10 倍重力）。

---

## 4. 力的量级：毫牛（推导，不是声明）

由 §3 的三个**已文档化**单位直接推出：

```
1 模型力单位 = 1 kg × 1 mm / s² = 1e-3 N = 1 mN
1 模型力矩单位 = 1 mN·mm
```

**没有任何文件声明「mN」**——这是从 kg + mm + s 推导出来的，本文如实标注为**推导**。

### 4.1 与总质量/体重的对照

本轮只读加载模型得到：

```
body_mass.sum()   = 0.002494271478            （71 个 body）
按 kg 读          → 2.494271478 mg   ← 成年果蝇量级（约 1 mg；本项目自己记录「重约 2.4 倍」）
按 g  读          → 2.494271478 g    ← 约 2500 倍真果蝇
按 mg 读          → 2.49e-9 kg       ← 约轻 400 倍

一个体重 = 0.002494271478 kg × 9801 mm/s² = 24.4464 模型力单位 = 24.4464 mN
```

### 4.2 与「静止/静默时总执行器力可达 ~34 模型单位」的一致性

若 **1 单位 = 1 mN**：34 mN ≈ **1.4 倍体重**。这正是一只靠六条腿站住的果蝇应有的量级（腿必须承担并超过体重）。

若 **1 单位 = 1 µN**（即「克」读法）：34 µN ≈ 体重的 **0.14%**——**根本托不住这只蝇**。

⚠️ **诚实标注**：`~34` 这个数字是任务简报里**给定**的，本轮**没有重测**它，也**没有跑 episode**。这条只是**量级交叉验证**，不是一次新测量。

---

## 5. 更正：哪些项目内既有说法不成立

### 5.1 关于 `dynprm` 的事实更正

简报里说「actuator dynprm [1e-4, 4e-4, 0]」，并暗示这是分支改的。**实际上规范资产本身就已经是 0.0001/0.0004**：

```
fruitfly_six_leg_standing.xml:38   （class 默认，已被覆盖）
      <general … dynprm="0.01 0.04 0 0 0 0 0 0 0 0" … />
fruitfly_six_leg_standing.xml:544  （所有 90 个具名肌肉）
      <general name="LFC_tergopleural_promotor_a" class="muscle" … 
        dynprm="0.0001 0.0004 0 0 0 0 0 0 0 0" … />
```

即 class 默认值 0.01/0.04 **被每一个具名 actuator 覆盖成了 0.0001/0.0004**；`zero_floor` 分支改的**只有 `ctrlrange`**。`gainprm` 的 `0.0001/0.0004` 是 MuJoCo muscle actuator 的 `τ_act/τ_deact`（秒），与单位结论无关但值得记正。

### 5.2 项目内的质量单位说法（三种，互斥）

| 说法 | 出处（行号） | 结论 |
|---|---|---|
| **kg**（数值即 kg） | `engine/canonical_physics.py:9`（`total mass = 0.002494271478 kg`） | 数值对，但同句后半 `= 2.494 mg` 自相矛盾（0.002494 kg 是 2.494 **克**） |
| **g**（1 单位 = 1 克） | `engine/canonical_physics.py:88` `MODEL_MASS_UNIT_KG = 1e-3  # INFERRED (gram)`；`BODY_UNITS_PROVENANCE.zh-CN.md` §4.3 | **本轮判定不成立**（见 §3、§4） |
| **mg**（1 单位 = 1 mg） | `engine/canonical_physics.py:54`（旧版 `1e-6`）、`:419` `"mass_model": "mg"` | 不成立 |

`engine/embodied/body_backend.py:92` 里其实已经写对了链路：

```
#: Actuator FORCE LIMIT, in the model's own force unit (kg*mm/s^2 = 1 mN; the model's
```

即**项目里一处写对了（mN），另一处写成了克**。

### 5.3 为什么此前的「克」推论会错

`BODY_UNITS_PROVENANCE.zh-CN.md:261-281` 用 **mesh 体积和 = 1.4679 mm³（上界）** 反推密度，得出「只有克落在生物组织数量级」。这条路的问题：

1. 它**明确没有查上游**（该文 §7 自述「资产本身从未声明」就停下了），而声明**存在于上游 OpenSim 文件里**；
2. 它依赖「体积是上界 → 真密度更高」这一步，并接受「密度高于几丁质」为代价；
3. 它把「单位」当成必须由**体积/密度**约束的东西，但**单位是由作者声明的东西**——声明一旦找到，密度论证就不再需要。

**必须说清楚**：该文把长度单位 mm 确立下来、并如实拒绝在无声明时编造质量单位，这两点是对的、有价值的；本轮只是**找到了它当时没找到的那份声明**。它自己写的「SI 质量换算未建立」正是本轮要处理的状态。

---

## 6. 自洽的 SI 读法（回答任务第 3 问）

**存在，且唯一自洽的读法是**：

| 量 | 模型单位 | 等价 SI |
|---|---|---|
| 长度 | mm | 1e-3 m |
| 质量 | **kg** | 1 kg |
| 时间 | s | 1 s |
| 重力 | 9801 mm/s² | 9.801 m/s² |
| 力 | **mN** | 1e-3 N |
| 力矩 | mN·mm | 1e-3 N·mm |
| 惯量 | kg·mm² | 1e-6 kg·m² |

**等价表述**：把同一个模型改用米写，就是「几何 ×1e-3、质量仍为 2.494e-3 kg、重力 9.801 m/s²、力 1e-3 N」——**纯单位更名，动力学完全不变**。

**一个重要的实务推论（「改标签不改物理」）**：因为 MuJoCo 在任意自洽单位制下动力学相同，**只要某个数字没有被换算后重新灌回模型**，它的数值和轨迹就不受标签影响。所以本轮**不声称**任何历史运行因此有物理错误；本轮只更正**对外把模型力读成 µN / 把质量读成克或 mg** 的那些标签。

具体方向性提示（**不重算历史**）：

- 若某处把模型力 ×1e-6 报成 µN（「克」读法），应改为 ×1e-3 报成 mN —— 即**放大 1000 倍**。
- 若某处把总质量报成 2.494 mg 且同时用 `MODEL_MASS_UNIT_KG` 做换算，注意 2.494 mg **数值恰好是 kg 读法的总质量**，两者一致；但 `MODEL_MASS_UNIT_KG = 1e-3` 这个**常量本身**是错的（应为 1.0）。

---

## 7. 未解决 / 无法证实（明确列出）

1. **资产内部仍无质量单位声明。** `fruitfly_six_leg_standing.xml` 与 flygym 内置的 musculoskeletal XML **都没有**任何 mass 单位字段；mesh 缓存的 `metadata.yaml` 也**不声明单位**。kg 结论来自**上游 OpenSim 源 + 数值不变**，不是资产自己说的。
2. **MyoConverter 的具体转换代码没有读**（只读了它的两个文档页）。因此「几何 ×1000 是转换器做的还是作者手工做的」**未确立**；`9806.65 → 9801` 的确切取整路径也**未追**。这不影响单位结论，但影响对来源的完整叙述。
3. **总质量 2.494 mg 比真果蝇重约 2.4 倍**，这是**标定/生物学**问题，与单位无关，**本轮未解决**。
4. **`~34` 模型单位**取自任务简报，**本轮未重测**。
5. **规范资产相对 FlyMimic 祖先的改动没有做穷尽 diff**：头部注释自述改了地板、keyframe、40 个 attachment site、75 个 force-length band；其中**重新放置 attachment site 会改变 moment arm**，从而可能改变项目实测到的力量级。本轮只核对了 gravity / statistic / mesh scale / 质量数值一致。
6. **项目其他历史报告里的「模型力单位」数字**没有被逐个重新判定，那需要重读并重新解释大量既有结果，**超出本轮范围**。
7. **两个文档页只部分打开**：`https://neuromechfly.org/api_ref/flygym.html` 返回 **HTTP 404**（路径错误）；`https://neuromechfly.org/api_reference/flygym/compose/fly/base_fly/` **能打开但正文被截断**，因此 mm 约定的权威出处改用了**已安装包的源码**（`base_fly.py:166-168`）。**没有虚构任何页面内容。**
8. **没有做任何动力学测量**：本轮不验证也不推翻任何逃逸时间、接触数、塌陷量、肌肉力。

---

## 8. 什么证据能真正定案

1. **FlyMimic 后续版本或作者出面**，在 MJCF/仓库里写明单位（今天不存在）。
2. **实际使用的 MyoConverter 转换配置**，或作者对「几何 ×1000」这一步的说明。
3. **一次决定性的静力学测量**：把蝇静置，读出接触力之和（模型单位），与 `总质量 × 9801` 比较。
   - kg 读法预测：**24.4464**
   - g 读法预测：同物理力会差 **1000 倍**
   这是一次模拟任务，本轮**刻意没有跑**（只读约束）。
4. **独立文献**给出成年果蝇质量约 1 mg，用于印证 2.494 mg 是正确量级（而非正确数值）。

---

## 9. 来源清单（本地路径 + URL，含片段）

机器可读版本见 `/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/model_unit_sources.json`（24 条，含 `excerpt` / `retrieval_status` / `supports`）。

### 9.1 本地

| 编号 | 路径 | 行 | 原文（片段） |
|---|---|---|---|
| S1 | `/tmp/best_combined.osim`（下载；源 <https://github.com/gizemozd/FlyMimic/blob/main/flymimic/assets/models/opensim/best_combined.osim>） | 24 | `<length_units>m</length_units>` |
| S2 | 同上 | 26 | `<force_units>N</force_units>` |
| S3 | 同上 | 18 | `<gravity>0 -9806.6499999999996 0</gravity>` |
| S4 | 同上 | 64,104,…,434 | `<mass>0.0002139246364324101</mass>`（71 个，和 `0.0024942719507646226`） |
| S5 | `/tmp/flymimic_atds.xml`（下载；源 FlyMimic `best_combined_arm_damping_stiff_cvt3.xml`） | 2 | `This model has been converted from an OpenSim model. Model conversion by MyoConverter …` |
| S6 | 同上 | 4, 12-15 | `<option timestep="1e-04" gravity="0 0 -9801"/>` / `<mesh … scale="1000 1000 1000"/>` |
| S7 | `/run/media/sensen/Data2/cell_wound_prototype/outputs/muscles_six_legs/fruitfly_six_leg_standing.xml` | 26 | `<option timestep="1e-04" gravity="0 0 -9801" />` |
| S8 | 同上 | 46-116（71 处） | `<mesh name="Thorax_geom_Thorax" file="./meshes/stl/Thorax.stl" scale="1000 1000 1000" />` |
| S9 | 同上 | 544 | `<general name="LFC_tergopleural_promotor_a" class="muscle" … dynprm="0.0001 0.0004 0 0 0 0 0 0 0 0" …>` |
| S10 | `/run/media/sensen/Data2/cell_wound_prototype/venv_body/lib/python3.12/site-packages/flygym/flybody/parse_flybody.py` | 6-11 | `# … to FlyGym's mm convention: length x10, mass and time unchanged.` |
| S11 | `…/venv_body/lib/python3.12/site-packages/flygym/compose/fly/base_fly.py` | 166-168 | `# For numerical reasons, we simulate length in mm, not m. This changes the units` / `# of other quantities as well, for example acceleration is now in mm/s^2.` / `SCALE = 1000` |
| S12 | `…/venv_body/lib/python3.12/site-packages/flygym/compose/fly/musculoskeletal.py` | 160-177 | `class MusculoskeletalFly(BaseCompositionElement):` … `Source code for the original model: https://github.com/gizemozd/FlyMimic`（**该模块无 SCALE**） |
| S13 | 同上 | 93-105 | `DEFAULT_MUSCULOSKELETAL_XML = (… / "best_combined_arm_damping_stiff_cvt3.xml")` |
| S14 | `…/venv_body/lib/python3.12/site-packages/flygym/assets/model/flybody/mujoco_globals.yaml` | 9 | `gravity: ['0', '0', '-9810']  # mm/s^2 (source XML is cm: -981 cm/s^2 x10 -> mm)` |
| S15 | `/home/sensen/.cache/flygym_assets/neuromechfly_musculoskeletal_meshes_20260623a/metadata.yaml` | — | `description: > Meshes in musculoskeletal model from FlyMimic …`（**不含任何单位**） |
| S16 | `/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/body_backend.py` | 12, 49, 65, 92 | `…extent: 1  # mm`；`(kg*mm/s^2 = 1 mN; the model's` |
| S17 | `/run/media/sensen/Data2/cell_wound_prototype/engine/canonical_physics.py` | 77, 88 | `MM_PER_MODEL_LENGTH = 1.0` / `MODEL_MASS_UNIT_KG = 1e-3  # INFERRED (gram), not established` |
| S18 | `/run/media/sensen/Data2/cell_wound_prototype/BODY_UNITS_PROVENANCE.zh-CN.md` | 24, 261-291 | `| A-质量 | … 只有「克」通过物理约束 | **SI 质量换算未建立** |`（前一轮审计，本轮更正其质量结论） |

### 9.2 网络

| 编号 | URL | 抓取状态 | 原文（片段） |
|---|---|---|---|
| W1 | <https://github.com/gizemozd/FlyMimic> | ✅ 成功 | `This repository … provides MuJoCo models converted from OpenSim.` |
| W2 | <https://ar5iv.labs.arxiv.org/html/2509.06426> | ✅ 成功（全文） | `…scaling them with a specific tension of 28 mN/mm^2.` / `Using MyoConverter, we converted our OpenSim model … into a MuJoCo compatible format.` |
| W3 | <https://myoconverter.readthedocs.io/en/latest/step1.html> | ✅ 成功 | `In this step, the OpenSim OSIM file is converted to a MuJoCo XML file.` |
| W4 | <https://myoconverter.readthedocs.io/en/latest/limitations.html> | ✅ 成功 | 官方 Limitations 列表**完全未提单位换算**（负面证据） |
| W5 | <https://neuromechfly.org/api_reference/flygym/compose/fly/base_fly/> | ⚠️ 打开但**正文截断** | `Abstract base for all fly body models (e.g. NeuroMechFly, FlyBody).` |
| W6 | <https://neuromechfly.org/api_ref/flygym.html> | ❌ **HTTP 404**（路径错误） | `# 404 - Not found` —— 如实记录，未编造内容 |

---

## 10. 产物清单（绝对路径）

| 交付物 | 绝对路径 | 状态 |
|---|---|---|
| 本报告 | `/run/media/sensen/Data2/cell_wound_prototype/MODEL_UNIT_PROVENANCE.zh-CN.md` | 新增 |
| 机器可读来源 | `/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/model_unit_sources.json` | 新增（24 条来源） |
| 临时下载（**不在项目内**） | `/tmp/best_combined.osim`、`/tmp/flymimic_atds.xml` | 临时 |
| 被修改的项目文件 | **无** | 只读 |

规范资产 SHA256 在本轮读取前后一致：`65d3904dce9802f46997a92cb3f80cd13c032f847c1e1f6710694d65cfdd73b6`。

---

## 11. 本文自身可能出错的地方（自我否证）

1. **我没有读 MyoConverter 的转换代码。** 若它的默认行为其实**包含**一次质量重标定（例如把 kg 改成 g），而「数值相同」是两次相反缩放互相抵消的巧合，则 kg 结论会被推翻。我认为这不成立，理由是**惯量数值也同时相同**，且 flygym 的书面约定是「mass and time unchanged」；但这是**可被下一轮推翻**的点。
2. **我依赖 `best_combined.osim` 与规范资产是同一谱系。** 二者质量数值在 6 位有效数字上相同、FlyMimic 仓库只有一个 `.osim` 对应这组模型，但我**没有**做逐 body 的一一配对（body 顺序在上游转换中被重排）。
3. **`√(I/m)` 的中位数 0.087 只是量级论证。** 它排除「厘米/米级」读法，但 0.087 mm 与 0.087 m 的区别靠的是 mesh scale 与 gravity，不是我这一条。
4. **`~34` 未重测**，§4.2 的一致性论证因此不是独立测量。
5. **规范资产的 attachment-site 重放未被 diff**（§7.5），它可能影响项目**实测**到的力量级——但**不影响单位**。
