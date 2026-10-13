# 规范资产单位溯源审计（BODY_UNITS_PROVENANCE）

> ## ⚠️ 这是手工实施的研究审计，**不是成品**
>
> 本文是我在本机**手工编写并手工跑通**的一次只读溯源审计，不是经过评审、回归或版本化的产品模块。
> - 它**不重新测量任何物理**：没有加载 MuJoCo、没有编译模型、没有解接触、没有积分轨迹。
> - 它**只做两件事**：把每个单位说法追到「文件 + 行号」，以及从**资产实际缩放的 mesh 几何**独立推出长度单位。
> - 它能推翻标签，**不能**替代任何一次真实测量。凡本文标「未建立 / 无法证实」的，就是**没有**建立。
>
> 审计对象（**只读，未修改**）：
> `/run/media/sensen/Data2/cell_wound_prototype/outputs/muscles_six_legs/fruitfly_six_leg_standing.xml`
> SHA256 = `65d3904dce9802f46997a92cb3f80cd13c032f847c1e1f6710694d65cfdd73b6`（审计前后均核对，未变）
>
> 本审计**没有**修改该 XML，**没有**修改任何历史日志，**没有**修改 `engine/canonical_physics.py` 或其他共享模块。

---

## 0. 结论速览（先给答案）

| # | 问题 | 结论 | 强度 |
|---|---|---|---|
| A-长度 | 模型长度单位是什么？ | **毫米（mm）**，不是厘米 | **已确立**（三条独立证据） |
| A-时间 | 模型时间单位？ | **秒（s）**，无争议 | 已确立（无任何字段反对） |
| A-质量 | 模型质量单位？ | **资产本身从未声明**；项目内部存在 kg / g / mg **三种互斥说法**；只有「克」通过物理约束 | **SI 质量换算未建立** |
| B | gravity=9801 是否自洽？ | **只在毫米下自洽**：9801 mm/s² = 9.801 m/s²。按厘米读则是 98.01 m/s² | 已确立 |
| C | 质量数值来源字段 | `<inertial mass=...>`，71 个，和 = `0.002494271478`；**无任何 `density=`** | 已确立 |
| D | 尺寸是否生物学合理？ | 毫米下**合理**（站立胸高 2.045 mm、体长 ~2.6 mm、胸 mesh 1.069 mm）；厘米下**荒谬**（站立胸高 20.45 mm、体长 26 mm） | 已确立 |
| E | 单位读错会连带影响什么？ | 见 §6（只给方向，不重算历史） | 定性 |

**一句话**：`engine/canonical_physics.py` 第 52–56 行声称的「厘米 + 1 模型质量单位 = 1 mg」**与资产本身、与上游 flygym 的约定、与 mesh 几何三者都不符**；长度单位错了 10 倍，而质量单位在项目内根本没有统一说法。

---

## 1. 方法、安全与可复跑证据

### 1.1 工程约束（遵守）

- 入口用 `engine/memory_guard.py` 的 `guarded_run` 包裹：
  `with mg.guarded_run("units_audit", **mg.BUDGETS["units_audit"]): main(...)`
  （`need_mb=1200`，`cap_mb=4608`，与 `MEMORY_GUARD.zh-CN.md` §3 的预算表一致）
- 脚本**纯标准库**（`hashlib/json/math/re` + `struct`），**不 import numpy/pandas/mujoco**，因此 `venv/bin/python` 与 `venv_body/bin/python` 都能跑
- **没有**运行 12,000 神经元网络，**没有**跑 MuJoCo 长积分，**没有**并发重任务

### 1.2 实测占用（守卫自己报告的，不是估计）

| 项 | 实测 |
|---|---:|
| label | `units_audit` |
| 峰值 `VmPeak` | **44.59 MB** |
| 结束 `VmRSS` | 18.86 MB |
| `ru_maxrss` | 19.90 MB |
| wall time | **0.109 s** |
| cap 利用率 | 0.97 %（上限 4608 MB） |
| 结束后 MemAvailable | 5512.8 MB |

即：**这次审计的足迹约为本项目实测峰值（3159 MB）的 1.4%**，与当初造成 OOM 的重任务不是同一量级。

### 1.3 精确复跑命令

```bash
cd /run/media/sensen/Data2/cell_wound_prototype
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python \
  /run/media/sensen/Data2/cell_wound_prototype/run_body_units_audit.py
```

**退出码：0**。原始输出：`/tmp/units_audit_run.log`。

---

## 2. 问题 A：长度 / 质量 / 时间单位，从上游追

### 2.1 资产链条（每一环都有绝对路径）

```
[1] flygym 2.1.0 内置的上游资产
    /run/media/sensen/Data2/cell_wound_prototype/venv_body/lib/python3.12/site-packages/
      flygym/assets/model/musculoskeletal/best_combined_arm_damping_stiff_cvt3.xml
        ├─ <option timestep="1e-04" gravity="0 0 -9801"/>          ← 与规范资产逐字节相同
        ├─ <statistic meansize="0.298677" extent="5.78307" .../>   ← 与规范资产逐字节相同
        └─ 71 个 mass=，和 = 0.002494271478                        ← 与规范资产完全相同

[2] 该资产的模型来源（flygym 自己写的出处）
    .../flygym/compose/fly/musculoskeletal.py:78-98
    .../flygym/compose/fly/musculoskeletal.py:167-177
      → "FlyMimic's musculoskeletal MJCF"
      → Özdil et al., "Musculoskeletal simulation of limb movement biomechanics in
         Drosophila melanogaster", ICLR 2026, arXiv:2509.06426
      → 源码 https://github.com/gizemozd/FlyMimic

[3] 资产真正缩放的 mesh（包内不附带，flygym 首次使用时下载到缓存）
    /home/sensen/.cache/flygym_assets/neuromechfly_musculoskeletal_meshes_20260623a/
      ├─ 71 个 *.stl  +  metadata.yaml
      └─ metadata.yaml 原文：
           description: Meshes in musculoskeletal model from FlyMimic
                        (Özdil et al., 2025, arXiv:2509.06426).
           source: https://github.com/gizemozd/FlyMimic
           version: 20260623a
         （注意：metadata.yaml **没有声明任何单位**）

[4] 规范资产（本次审计对象，未修改）
    /run/media/sensen/Data2/cell_wound_prototype/outputs/muscles_six_legs/
      fruitfly_six_leg_standing.xml
      生成脚本：tools_stand_and_ground.py
```

**为什么 [3] 就是「资产实际用的 mesh」**：规范 XML 里写的是相对路径 `./meshes/stl/Thorax.stl`，
而 `outputs/muscles_six_legs/meshes/stl/` **不存在**。按 `flygym/compose/fly/musculoskeletal.py:142-164`
的 `_resolve_mesh_files()`，本地找不到时**按文件名回落到 S3 缓存目录** `MUSCULOSKELETAL_MESH_DIR`
（即 [3]）。所以资产编译时用的就是这份 mesh。

**关键点 [1] 与 [4] 的质量对照**：上游 71 个 mass 之和 = 规范 71 个 mass 之和，
**比值 = 1.0**。即**本项目没有对质量做过任何重新缩放**——质量的来源完全在上游。

### 2.2 决定性证据 ①：`mesh scale=1000` 作用在**米制** STL 上 → 模型单位 = 毫米

规范 XML 第 46–116 行有 **71 个** `<mesh ... scale="1000 1000 1000"/>`。
MuJoCo 的 `mesh@scale` 把 mesh 文件坐标**乘进模型长度单位**。所以只要知道 STL 原始坐标的量级，就知道模型单位。

我直接读了缓存里的 STL（`struct.iter_unpack` 解析二进制 STL 三角面）：

| mesh | 原始坐标跨度（STL 文件单位） | ×1000 后 = 模型单位 |
|---|---:|---:|
| `Thorax.stl` | 1.069e-3 × 9.470e-4 × 1.085e-3 | **1.0692 × 0.9470 × 1.0850** |
| `Head.stl` | 5.262e-4 × 7.630e-4 × 6.747e-4 | 0.5262 × 0.7630 × 0.6747 |
| `LFFemur.stl` | 1.264e-4 × 1.695e-4 × 6.082e-4 | 0.1264 × 0.1695 × **0.6082** |
| `LFTibia.stl` | 9.523e-5 × 8.712e-5 × 5.679e-4 | 0.0952 × 0.0871 × **0.5679** |
| `LHTibia.stl` | 1.332e-4 × 1.155e-4 × 7.594e-4 | 0.1332 × 0.1155 × **0.7594** |
| `LHTarsus5.stl` | 7.134e-5 × 6.570e-5 × 1.058e-4 | 0.0713 × 0.0657 × 0.1058 |

**解读**：原始 STL 坐标是**米**（胸 mesh 长 1.069 mm = 真实的果蝇胸长量级；这些 mesh 是 FlyMimic 从真实标本
X 射线/μCT 重建的）。乘 1000 后得到的是 **毫米**。

- 按**毫米**读：胸 1.07 mm、头 0.53 mm、前股节 0.61 mm、前胫节 0.57 mm、后胫节 0.76 mm —— 全部是真实果蝇的节长量级。
- 按**厘米**读：胸 10.7 mm、前股节 6.1 mm、后胫节 7.6 mm —— 那是**一只 2 厘米长的「果蝇」**，节长比整个真果蝇还长 3–4 倍。

### 2.3 决定性证据 ②：body 偏移 与 mesh 跨度**必须同量级**，实测中位比 0.941

在 MJCF 里，子体 `<body pos="...">` 的关节偏移与 mesh 坐标**共用同一个模型长度单位**。
两者不要求相等（偏移是关节到关节的距离，mesh 跨度是表面包围盒，关节处有缝隙，所以比值会散），
但它们**必须是同一数量级**——若模型单位比 mesh 单位大/小 10 倍，全部比值会被整体乘/除 10。

我对**全部 70 个有同名 mesh 的 body**算了 `mesh最大跨度 / |body pos|`：

```
n = 70   中位数 = 0.9408   （最小 0.2941, LHCoxa；最大 32.8189, Head）
70 个里 69 个落在 1 的「一个数量级内」（0.1 ~ 10）
若长度单位是厘米而 mesh 是毫米：中位数会变成约 9.41
若长度单位是毫米而 mesh 是厘米：中位数会变成约 0.094
```

**再看两个最干净的个例**：

| 节 | body pos 偏移（模型单位） | mesh 最大跨度（模型单位） | 比值 |
|---|---:|---:|---:|
| `LFTibia` | 0.5661（z 偏移） | 0.5679 | **1.0032** |
| `RFTibia` | 0.5701 | 0.5685 | **0.997** |
| `LHTibia` | 0.8363 | 0.7594 | 0.908（后胫节 mesh 未覆盖整个节，有缝隙） |

即：**关节偏移与 mesh 跨度在同一个毫米量级上，比值 O(1)**。厘米解释会让它们差 10 倍。

### 2.4 决定性证据 ③：flygym **自己**声明的约定是毫米

| 文件 | 行 | 原文 |
|---|---|---|
| `.../flygym/compose/fly/base_fly.py` | **166–167** | `# For numerical reasons, we simulate length in mm, not m. This changes the units` / `# of other quantities as well, for example acceleration is now in mm/s^2.` |
| `.../flygym/compose/fly/base_fly.py` | **168** | `SCALE = 1000` |
| `.../flygym/simulation.py` | **188** | `Body positions in mm, shape ``(n_bodies, 3)``, ordered as in` |
| `.../flygym/compose/fly/flybody.py` | **106** | `# authored in cm; parse_flybody.py converts it to mm -- lengths x10, gravity x10` |

**注意最后一行**：flygym 明确说原始 flybody XML 是**厘米**写的，它会 `×10` 转成毫米。
这解释了上游两个资产为什么一个写 981、一个写 9801（见 §2.5），也说明**毫米是 flygym 的规范约定**。

### 2.5 旁证：同一包内两个上游资产的 gravity 差 10 倍

| 上游资产 | 绝对路径 | gravity | 一致的单位族 |
|---|---|---:|---|
| `flybody/fruitfly.xml` | `venv_body/.../flygym/assets/model/flybody/fruitfly.xml` 第 4 行 | `0 0 -981` | 981 cm/s² = 9.81 m/s²（厘米族） |
| `musculoskeletal/best_combined_...xml` | `venv_body/.../flygym/assets/model/musculoskeletal/` 第 4 行 | `0 0 -9801` | 9801 mm/s² = 9.801 m/s²（**毫米族**） |

**两个值各自都是「自己那套单位下的标准重力」**，倍数恰好是 10，正是厘米/毫米之比。
规范资产继承的是 **9801**，即**毫米族**。

### 2.6 旁证：项目自己的生成器写的也是毫米

| 文件 | 行 | 原文 | 含义 |
|---|---|---|---|
| 规范 XML（生成器写入的表头注释） | **9** | `the keyframe places the thorax at z=2.0453 mm` | 生成器自己叫它 **mm** |
| `tools_stand_and_ground.py` | **256** | `# at x=+0.52 -- and the abdomen runs back to x=-1.91, so the body is 2.6 mm long along x` | 生成器把模型坐标当 **mm** 用 |

**于是出现最尖锐的自相矛盾**：同一个数字 `2.04535`

- 规范 XML 第 9 行说它是 **2.0453 mm**；
- `engine/canonical_physics.py` 第 12 行说它是 **2.04535 (cm) = 20.4535 mm**。

两者不可能同时成立，而 §2.2–§2.5 的四条独立证据都站在 **mm** 这一边。

### 2.7 时间单位

`MODEL_TIME_UNIT_S = 1.0`（`engine/canonical_physics.py:55`）。
**没有任何文件、任何字段、任何上游资产反对这一点**：`<option timestep="1e-04">`、
`solref` 的 `0.02`、`armature`/`damping` 全部按秒解读才合理。**时间单位 = 秒，无争议。**

---

## 3. 问题 B：gravity = 9801 与长度单位是否自洽

`<option timestep="1e-04" gravity="0 0 -9801" />`（规范 XML 第 26 行）

| 若 1 模型长度单位 = | 则 gravity 读作 | 判定 |
|---|---:|---|
| **毫米** | **9801 mm/s² = 9.801 m/s²** | ✅ 标准重力 |
| 厘米 | 9801 cm/s² = **98.01 m/s²** | ❌ 约 10 倍重力 |

### 这里必须说清楚一件事（否则结论会被夸大）

**gravity 单独并不能「证明」单位系统**。一个 MuJoCo 模型在任意单位制下都自洽：只要质量、长度、时间、
力、刚度全部按同一套单位写，动力学就一样（这正是为什么**改标签不改物理**，见 §6）。
所以正确的表述是：

> `gravity=9801` **不决定**单位，但它**记录作者在写哪套单位**。
> 一个写 `9801` 的人写的是毫米；一个写 `981` 的人写的是厘米——同一包里两个上游资产正好各写了一个（§2.5）。
> 规范资产继承 `9801`，且它的 mesh 就是毫米（§2.2），两者**互相印证**。
> 相反，`canonical_physics.py:56` 把 `9801` 注释成 `cm/s^2`——**它把毫米族的数字配上了厘米族的标签**。

---

## 4. 问题 C：总质量换算与矛盾证据（含精确行号）

### 4.1 质量数值的来源字段

| 项 | 值 | 位置 |
|---|---|---|
| 来源字段 | `<inertial ... mass="..."/>` | 规范 XML 第 125, 137, 140, 143, 146, 149, 157, 160, … 共 **71 处** |
| 和 | **`0.002494271478`** | 由 71 个 mass 相加得到 |
| `density=` 出现次数 | **0** | 全文件无任何密度；因此**文件内部没有任何东西把质量与体积绑起来** |
| body 数 / inertial 数 | 72 / 71 | 唯一没有 `<inertial>` 的 body 是 `ground`（第 123 行），无质量 |
| 上游同名和 | `0.002494271478` | 与上游**完全相同**，比值 1.0 → 本项目**未重标定质量** |

**这一点很重要**：因为**每个 body 都有显式 `<inertial>`**，MuJoCo 不会用 `geom 体积 × 默认密度 1000`
去补质量。所以 `sum(body_mass)` 就是这 71 个数之和，**质量总量与长度单位无关**——
长度单位说什么，都不会改变这个数。**质量的 SI 含义完全取决于一个外部约定，而这个约定在文件里不存在。**

### 4.2 项目内部的三种互斥说法（这是本次审计的核心矛盾，我不替它选）

| 说法 | 推出的总质量（kg） | 推出的总质量（mg） | **证据位置** |
|---|---:|---:|---|
| **A. 数值是 kg** | 2.494271478e-3 kg | 2494.27 mg | `engine/canonical_physics.py:9`（`total mass = 0.002494271478 kg`） |
| **B. 数值是 g，即 2.494 mg** | 2.494271478e-6 kg | **2.49427 mg** | `engine/canonical_physics.py:9`（同一句后半：`= 2.494 mg`）；`tools_stand_and_ground.py:704`（`# model units = gram`）＋`:706`（`mass*1000` → mg）；`diagnostics/diag_leg_load_test.py:38-45`（`MASS_MG`，`cur = sum(m.body_mass)*1000.0`）；`LEG_TUNING.zh-CN.md:117`（`2.49427 mg`） |
| **C. 1 模型质量单位 = 1 mg** | **2.494271478e-9 kg** | 2.494271478e-3 mg | `engine/canonical_physics.py:54`（`MODEL_MASS_UNIT_KG = 1e-6  # 1 model mass unit = 1 mg`）；`engine/canonical_physics.py:419`（`"mass_model": "mg"`）；`engine/embodied/body_backend.py:43`（沿用同样说法） |

**三者互相矛盾，且 A 在自己的句子里就已经坏了**：

> `canonical_physics.py:9`：
> `(gravity = 9801, total mass = 0.002494271478 kg = 2.494 mg).`
>
> `0.002494271478 kg` 等于 **2.494 克**，不等于 **2.494 毫克**。这句话内部就差 1000 倍。

**A ×10³ ≠ B；B ÷ C = 10³。** 三选一必然要放弃另外两个，而**资产本身一个字都没说**。

### 4.3 物理约束：只有「克」活着（这是推论，不是声明）

资产没有密度，所以唯一独立约束来自 §2.2 已经量过的 mesh 几何。
我算了缓存里全部 71 个 mesh 的 |有符号体积| 之和：

```
mesh 体积之和 = 1.467890e-09（原始 STL 单位 = m³）= 1.4679 mm³
（这是**上界**：相邻节段的 mesh 在关节处互相重叠，被重复计了一次）
```

| 若 1 模型质量单位 = | 总质量 | 推出的密度 | 对照 |
|---|---:|---:|---|
| **kg** | 2.494271e-3 kg | **1,699,222 kg/m³** | 比锇（22,590）还重 75 倍，**排除** |
| **g** | 2.494271e-6 kg | **1,699 kg/m³** | 水=1000、几丁质≈1400；同量级，**唯一活着** |
| **mg** | 2.494271e-9 kg | **1.70 kg/m³** | 空气=1.20；比空气仅重 0.5 kg/m³，**排除** |

**诚实的方向性说明**：体积是**上界**，所以真实体积更小 → 真实密度**比 1699 更高**。
也就是说，即使按「克」读，这只模型果蝇的密度也**高于几丁质**——这与项目自己在
`LEG_TUNING.zh-CN.md` §4 与 `diagnostics/diag_leg_load_test.py:38` 记录的
「**模型比真果蝇重 2.44 倍**」是一致的，不是新问题。
但这不改变结论：**mg 差约 600 倍，kg 差约 10⁶ 倍，只有 g 落在生物组织的一个数量级内。**

### 4.4 我的结论表述（按任务要求，**不替项目选一个**）

> **XML 没有声明质量单位；项目内部有三种互斥声明（行号见 §4.2），本审计不采纳其中任何一个作为「声明」。**
> 唯一能说的是：**mg 与 kg 被几何排除，克是唯一物理可行的读法**——但这是**从几何反推的推论**，
> 不是**资产或项目作出的声明**。
> 因此契约产物里 `units.mass_si_conversion` 如实写：**「未建立 / NOT ESTABLISHED」**。

**不要再引用 `2.494 mg` 作为已确立的 SI 质量。** 它是 `tools_stand_and_ground.py:704` 的
「model units = gram」假设的下游产物，而这个假设在 `canonical_physics.py:54` 被同一项目推翻了。

---

## 5. 问题 D：尺寸是否生物学合理

### 5.1 模型里的尺寸（两种解释并列）

| 量 | 模型单位 | **按 mm 读（本审计结论）** | 按 cm 读（`canonical_physics.py:52`） | 出处 |
|---|---:|---:|---:|---|
| 站立时胸（free joint root）高 | 2.04535 | **2.0454 mm** | 20.4535 mm | 规范 XML 第 561 行 keyframe `qpos[2]`；XML 第 9 行注释 |
| 体长（沿 x，喙尖→腹末） | 2.6 | **2.60 mm** | 26.00 mm | `tools_stand_and_ground.py:255-258`（生成器自己的 mm 叙述） |
| 编译后模型 extent | 5.78307 | **5.78 mm** | 57.83 mm | 规范 XML 第 31 行 `<statistic extent=...>`（MuJoCo 编译期算出） |
| 胸 mesh 尺寸 | 1.0692×0.9470×1.0850 | **1.069×0.947×1.085 mm** | 10.69×9.47×10.85 mm | `Thorax.stl` × scale 1000 |
| 前胫节 mesh 长 | 0.5679 | **0.568 mm** | 5.68 mm | `LFTibia.stl` × scale 1000 |
| 地板（plane half-size） | 10 × 10 | 20 × 20 mm | 200 × 200 mm | 规范 XML 第 121 行 `<geom name="floor" ... size="10 10 0.125">` |

### 5.2 与已发表形态学比较

| 量 | 模型（按 mm） | 真实 *D. melanogaster* | 判定 |
|---|---:|---|---|
| 站立胸高 | 2.045 mm | 成虫体高约 1.5–2 mm 量级 | ✅ 同量级 |
| 体长 | 2.60 mm | 成虫体长 **约 2.5–3 mm** | ✅ 落在范围内 |
| 腿展（extent） | 5.78 mm | 六足展开约 5–6 mm | ✅ 同量级 |
| 胸长（mesh） | 1.069 mm | 胸长 **约 1.0–1.2 mm** | ✅ 恰好落在范围内 |
| 后胫节 | 0.759 mm | 后胫节约 0.7–0.9 mm | ✅ 同量级 |
| 总质量 | 2.494 mg（按「克」读） | **约 1.0 mg** | ⚠️ 重约 2.4 倍（项目自己已记录，见 §4.3） |
| 总质量 | 0.002494 mg（按 `canonical_physics.py:54` 读） | 约 1.0 mg | ❌ 轻约 400 倍 |

**按 cm 读会得到什么**：站立胸高 20.45 mm、体长 26 mm、腿展 57.8 mm、胸长 10.7 mm、单个前股节 6.1 mm。
**这是一只 8–10 倍于真果蝇的动物**——在生物学上不可能是 *Drosophila melanogaster*。

### 5.3 外部来源（并如实标注我**打开过什么、没打开什么**）

**我打开并读过全文的**：

- **FlyMimic 论文全文**（本资产的直接上游）：<https://ar5iv.labs.arxiv.org/html/2509.06426>
  → 它明确写：这是基于**多个真实标本的高分辨 X 射线扫描**重建的 Hill 型肌肉骨骼模型，经
  MyoConverter 从 OpenSim 转成 MuJoCo。**它全文没有声明长度或质量单位**——这正是本审计必须去读 mesh 的原因。
- **arXiv 摘要页**：<https://arxiv.org/abs/2509.06426>
- **FlyMimic 源码仓库**（由 mesh 缓存 `metadata.yaml` 的 `source:` 字段点名）：
  <https://github.com/gizemozd/FlyMimic>
- **NeuroMechFly（本资产所属的模型谱系）**：<https://www.nature.com/articles/s41592-022-01411-8>
  → 出处见 `venv_body/.../flygym/compose/fly/musculoskeletal.py:167-177`

**我只拿到链接、没有打开的**（我的抓取工具对 `application/pdf` 返回
`Error: unsupported content type "application/pdf"`，所以我**不谎称读到了内容**）：

- 果蝇属 29 个种的两性平均质量表：
  <https://www.biorxiv.org/content/10.64898/2026.03.15.711943v1.full.pdf>
- 果蝇属胸长均值附表：
  <https://besjournals.onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1111%2Fj.1365-2435.2008.01533.x&file=FEC_1533_sm_TableS5.pdf>

**「真果蝇约 1.0 mg」这条数字的来源要说清楚**：我不是从外部文献查到的，而是**项目自己记录的**——
`LEG_TUNING.zh-CN.md:117-119` 把「本模型 2.49427 mg / FlyGym 行走模型 1.02431 mg / 真实果蝇约 1.0 mg」
列成一张对照表，`diagnostics/diag_leg_load_test.py:38-42` 重复了同样的三个数。
**本审计没有独立复测这两个数字**，只把它们当作项目内部的既有记录引用。

**因此，「尺寸生物学合理」这条结论最强的支点不是文献，而是两条本机证据**：
① mesh 是真实标本重建的、且尺度是米制（§2.2）；② 模型自身的节长、胸长、腿展在毫米下自洽（§5.2）。

---

## 6. 问题 E：单位读错会连带影响哪些已发布数字（**只给方向**）

**先立一条原则**：**改标签不改物理。** MuJoCo 在任意单位制下自洽，所以只要一个数字**没有**被换算回去喂进模型，
它的**数值和动力学都不受标签影响**。真正会出事的是两类地方：**(i) 对外换算成 SI 的数字；(ii) 被标签误导而
重新换算后灌回模型的数字**（后者就是 `canonical_physics.py` 自己记录的 D2 缺陷）。

### 6.1 「胸高 20.449 mm」/「20.4535 mm」

**方向：标签放大 10 倍。** 正确值应为 **2.0449 mm** / **2.04535 mm**。

- `canonical_physics.py:12`：`the standing height is 2.04535 (cm) = 20.4535 mm`
- `canonical_physics.py:85-87`：`the round-37 best pose compiles to 14 floor contacts at a thorax height of 20.449 mm`
- `canonical_physics.py:93-95`：`SEATING_PENETRATION_MM = -0.5` + `mm_to_model(-0.5) = -0.05` 模型单位

**连锁方向**：`model_to_mm()`（第 59–61 行）乘 10，因此 `StanceFacts.root_z_mm`（第 262–271 行）
与 `as_dict()` 里所有 `*_mm` 字段、以及所有经 `model_to_mm` 出口的距离/高度/塌陷量/穿透量，**全部大 10 倍**。
`tools_stand_and_ground.py:256` 的「2.6 mm 体长」在 cm 假设下会被当成 26 mm。

**⚠️ 一个还没被发现的推论（值得下一轮优先处理）**：
`canonical_physics.py:93` 把历史穿透标成 `-0.5 mm`，理由是「`-0.05` 模型单位 = `-0.05 cm` = `-0.5 mm`」。
**若模型单位是毫米，则 `-0.05` 模型单位就是 `-0.05 mm`**——**round 39 这个「D1 修复」本身在同一个方向上错了 10 倍**。
好消息是它把**模型空间**的值保住了（`SEATING_PENETRATION_MODEL = -0.05`，第 95 行），所以**喂给 MuJoCo 的数没变**；
坏消息是**所有对外标签反错了 10 倍**。这意味着 round 39 的 D1「修复」**方向是反的**。

### 6.2 逃逸时间

**方向：秒数本身不受标签影响，但「用哪个判据算」和「初始位姿是不是真的」会变。**

- 时间单位无争议（§2.7），所以已记录在 `outputs/autonomous_goal_status.json` 的
  `escape_*_s`（0.0194 / 0.0259 / 0.0341 / 0.0875 / 0.1139 s 等）**作为「秒」是可信的**，
  单位标签不会让它们变。
- **但两件事会变**：
  1. **阈值是 mm 命名的**（同文件键名 `escape_to_1.8mm_s`、`escape_to_1.0mm_s`、
     `foot_lift_0.05mm_s`、`root_speed_5mm_s`）。若这些阈值曾经按「厘米→毫米」换算过，
     判据本身就被搬动了 → **同一段时间序列会给出不同的逃逸时间**。
     （方向：把阈值当 cm 换算会把判据放大 10 倍，逃逸被**更晚**判定，即逃逸时间**变长**。）
  2. 更严重的是 D2：**凡是把 `*_mm` 值除以 10 再写回 `qpos` 的运行**，fly 被放低了 10 倍而**埋进地面**，
     随后测到的是「地板把身体弹出来」，**不是果蝇自己塌陷**。`canonical_physics.py:16-20` 自己记录了
     这个失败态（179 个接触、身体低于地面）。
     **方向：这类运行的逃逸时间被人为缩短、接触数被人为抬高、塌陷量被人为放大**，
     它测的不是这只蝇。注意 `canonical_physics.py` 第 17–19 行写「4.3 mm 低于地面」「contact z median -7.3 mm」，
     在毫米读法下这些标签也各有 10 倍虚高（实为 0.43 mm / 0.73 mm）。

### 6.3 肌肉力 / 关节力矩 / 刚度

**力单位不是标签问题，而是被质量单位直接决定的**：`1 模型力单位 = 1 质量单位 × 1 长度单位 / s²`。

| 单位组合 | 1 力单位 | 1 力矩单位 | 与项目自身打印的关系 |
|---|---:|---:|---|
| **mm + g**（本审计认为唯一物理可行的组合） | 1 g·mm/s² = **1 µN** | 1 µN·mm | 与 `diagnostics/diag_leg_load_test.py:56` 打印的 `uN*mm/rad`、`LEG_TUNING.zh-CN.md:82` 的 `0.4 µN·mm/rad`、`:86` 的 `4.07 µN` **完全吻合** |
| mm + 1 mg（=`canonical_physics.py:54`） | 1e-3 µN = 1 nN | 1 nN·mm | 全部力/力矩**小 10⁶ 倍** |
| mm + kg（=`canonical_physics.py:9` 的字面读法） | 1e3 µN = 1 mN | 1 mN·mm | 全部力/力矩**大 10³ 倍** |

**方向**：
- `canonical_physics.py:54` 的 `MODEL_MASS_UNIT_KG = 1e-6` 若被任何力/力矩/功率换算采用，
  任何以 N / µN 报出的**肌肉力、地面反力、关节力矩**都会**错 10³–10⁶ 倍**。
- 反过来，项目里**更早、更仔细**的那批工作（`LEG_TUNING.zh-CN.md`、`diagnostics/diag_leg_load_test.py`）
  用的是 **mm + 克 → µN**，与 §2、§4.3 的结论**一致**。
  也就是说：**round 39 新引入的 `canonical_physics.py` 单位声明，比它想修的那些老文件更不可靠。**
- 另有一个**与单位标签无关**的标定误差（项目自己已发现）：
  模型总质量 2.49427 mg 比真果蝇重 **2.44 倍**，因此**每一个载荷、力矩、塌陷量都放大 2.44 倍**
  （`LEG_TUNING.zh-CN.md` §4、`diagnostics/diag_leg_load_test.py:38`）。
  **方向**：修正它会让塌陷量、所需关节刚度、单腿承载全部**变小**。

### 6.4 契约产物本身的默认值

`canonical_physics.py:415-423` 的 `write_result()` **硬编码**了：

- 第 **419** 行 `"mass_model": "mg"`  ← 与同文件第 **54** 行 `1e-6`「1 单位 = 1 mg」等价，即总质量 2.494e-9 kg
- 第 **423** 行 `"illustrative_vs_measured": "measured (all numbers below are simulator output)"`

这两个硬编码值**对本问题都不适用**（正如任务说明所指出的），因此本次**没有**使用 `cp.write_result()`，
而是自建了等价 schema（`units` 字段如实写「SI 质量换算未建立」）。
**方向**：只要还有产物走 `cp.write_result()`，它们就会继续把 419 行的 `mg` 写进契约里，
读者会顺理成章地把总质量读成 2.494e-9 kg。

### 6.5 一个我**没有**解决的观察（如实记录）

`outputs/electrode_payload/episode_*/vision*.json` 里有字段 `touch_force_uN = 1674.174943583333`。
按 §6.3 的 mm+g 读法，整只蝇的体重 = 2.494271478 g × 9801 mm/s² = **24.45 µN**，单腿 1/6 = **4.07 µN**。
即这个被标为 µN 的数是**单腿体重的约 411 倍**（或整只蝇体重的约 68 倍）。
这**可能是**另一个独立的单位/标定问题，**但本审计没有追它**——它不影响 §2–§5 的任何结论，
列为待查项，**不当作已证事实**。

---

## 7. 明确「无法证实」的部分

1. **质量单位的 SI 换算未建立。** 资产无质量单位、无密度；项目内三种互斥声明。
   「克」只是**几何上唯一可行**的读法，是**推论**而非**声明**。**不得**把 2.494 mg 当作已确立的 SI 质量。
2. **本审计没有重新测量任何动力学。** 没编译 MuJoCo 模型、没解接触、没积分轨迹。
   它**不能**验证或推翻任何逃逸时间、接触数、肌肉力。它只说这些数字的**单位标签该怎么读**。
3. **本审计没有说历史运行因为标签而物理错误。** 纯标签错误不改变任何未做换算的仿真。
   只有**真的做了换算**的代码路径（`canonical_physics.py` 的 D2：`qpos[2] = root_pose_mm/10.0`）改变了物理。
4. **本审计没有改正任何已发布报告。** §6 只给**方向**；逐条重算是一个**尚未开始**的独立任务。
5. **mesh 缓存的版本没有被校验过。** 缓存只有目录名 `neuromechfly_musculoskeletal_meshes_20260623a`，
   项目里**没有记录上游 mesh 的 hash**，所以 §4.3 的密度论证锚定的是**本机今天**这份缓存，
   无法证明它与历史运行所用的是同一版。
6. **§6.5 的 `touch_force_uN` 异常没有追。**
7. **两个外部 PDF（果蝇质量表、胸长表）我没能打开**（抓取工具拒绝 `application/pdf`），
   所以 §5.2 里「真果蝇约 1.0 mg」等数字的来源是**项目内部记录**（`LEG_TUNING.zh-CN.md`），
   **不是**本审计独立查证的外部文献。
8. **`LEG_TUNING.zh-CN.md` 里「网格体积 × 密度 1000」的算法我没有复算。**
   我独立算出全部 mesh 体积和 = 1.4679 mm³（上界），但没有去复现上游那一步的换算路径。

---

## 8. 产物清单（绝对路径）

| 交付物 | 绝对路径 | 状态 |
|---|---|---|
| 审计脚本 | `/run/media/sensen/Data2/cell_wound_prototype/run_body_units_audit.py` | 新增，退出码 0 |
| 本报告 | `/run/media/sensen/Data2/cell_wound_prototype/BODY_UNITS_PROVENANCE.zh-CN.md` | 新增 |
| 契约产物 | `/run/media/sensen/Data2/cell_wound_prototype/outputs/body_physics_audit/units_provenance.json` | 新增，31125 字节 |
| 运行日志 | `/tmp/units_audit_run.log` | 新增（临时） |

**契约产物的 schema**（**未使用** `cp.write_result()` 的默认值）：
`status` / `status_meaning` / `question` / `measured` / `provenance` / `units` /
`mass_unit_contradiction` / `length_unit_contradiction` / `what_this_does_not_support` / `run`

其中：

```json
"units": {
  "length_model": "mm",
  "length_model_established": true,
  "mm_per_model_length": 1.0,
  "time": "s",
  "gravity_model": -9801.0,
  "gravity_in_SI_m_s2": 9.801,
  "mass_model": "NOT DECLARED BY THE ASSET",
  "mass_si_conversion": "未建立 / NOT ESTABLISHED. ...",
  "mass_unit_physically_admissible": ["gram"],
  "illustrative_vs_measured": "measured -- every number above is read out of a named file..."
}
```

`status` 为 `"PASS"`，含义**仅指审计本身跑完且 hash 未变**，`status_meaning` 字段里已明写
「这不是在说项目的单位声明是对的——那些被查出互相矛盾，长度单位则被查出是错的」。

---

## 9. 本审计自身可能出错的地方（自我否证）

写在最后，因为一个不做自我否证的审计不值得信：

1. **我可能把 `mesh scale` 的方向搞反。** 我判定 MuJoCo 的 `mesh@scale` 是「乘进模型单位」。
   若实际是「除」，结论会反转。但我用两条独立事实锁定它：① STL 原始跨度是 1e-3 量级（米），
   ② model 里同一个节的 `<body pos>` 偏移与 mesh 跨度比值中位 0.94（§2.3）。两者只有在「×1000 → 毫米」下同时成立。
2. **mesh 缓存可能不是历史运行用的那一版**（§7.5）。若上游换过 mesh，§4.3 的密度会变，
   但 §2.3 的**比值论证**仍然成立（它只用同一份文件内的一致性）。
3. **`0.9408` 这个中位数确实不是 1.0。** 我把它解释为「mesh 未覆盖整个节 + 关节缝隙」。
   若这个解释错了，另一种可能是模型里存在一个非 1 的几何缩放——但那会同时破坏 §2.2 的节长生物学合理性，
   所以我不认为这是更可能的解释。**这是一个可被下一轮推翻的点。**
4. **质量「克」的结论依赖 mesh 体积是上界。** 上界意味着真密度 ≥1699 kg/m³，
   所以「克」这一读法本身也已经偏重；我只是说它是**唯一不荒谬**的候选，不是**恰好正确**的候选。

---

## 10. 规范资产完整性（审计前后）

```
$ sha256sum /run/media/sensen/Data2/cell_wound_prototype/outputs/muscles_six_legs/fruitfly_six_leg_standing.xml
65d3904dce9802f46997a92cb3f80cd13c032f847c1e1f6710694d65cfdd73b6  .../fruitfly_six_leg_standing.xml
```

与要求的 `65d3904dce9802f46997a92cb3f80cd13c032f847c1e1f6710694d65cfdd73b6` **一致，未变**。
审计脚本第 0 步也在运行时断言了这个 hash，不一致就直接拒绝继续。
`engine/canonical_physics.py`、`engine/memory_guard.py` 以及任何历史日志、`outputs/` 下的既有产物**均未改动**。
