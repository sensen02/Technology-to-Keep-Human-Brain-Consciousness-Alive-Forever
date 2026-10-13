# 修复与误导性产物审计（本轮）

**手工搭建的研究原型。** 本文记录本轮"全项目扫描 → 修复 → 标记误导产物"的结果，包含**两处对我自己先前结论的更正**。

---

## 1. 已修复

### 1.1 像素门限未按帧高归一化（这是高分重建失败的真根因）

**文件**：`/run/media/sensen/Data2/cell_wound_prototype/run_arena_reconstruct.py`

**问题**：`associate_static` 的门限（2.5 / 3.0 / 4.0 px）是在**帧高 200 行**上标定的。门限是**物理容差**，要表达"同一个毫米位移"；保持数值不变而把行数翻 4 倍，物理容差就翻了 4 倍。

**修复**：新增 `GATE_REFERENCE_ROWS = 200` 与 `gate_px_at_rows()`，按每台相机**自己的帧高**换算；`associate_static` 返回 `gate_px_per_camera` 与 `gate_normalised_by_rows`，让门限本身可查。`normalise_gate=False` 仅用于复现历史运行。

**自检（已通过）**：在 200 行上门限**逐位等于**历史值（3.0→3.0、2.5→2.5、4.0→4.0），因此所有已记录的 200 行结果不变；640×800 下 3.0→12.0 px。

**实测方向确认**：seed0（640×800）在有效 **4 px → 0/60 帧**，有效 **16 px → 59/60 帧**。方向正确。

### 1.2 文档自相矛盾：错误倍数与紧邻的更正并存

**文件**：`/run/media/sensen/Data2/cell_wound_prototype/ABLATION_FIXED.zh-CN.md` 第 25 行

正文仍写"这股 **40 倍体重**的内力"，而紧邻上方第 21–23 行已更正为 **16.5 倍**（403 力单位 ÷ 24.4464 力单位）。读者略过更正框仍会拿到错数字。**已改为 16.5 倍。**

### 1.3 `tools_limb_eval.py` 外推律（上一轮已修）

该文件曾以"`角度×间距` 乘积恒定"为由，把 6–12 mm 基准点对的结果下推到 0.7 mm 的骨。**实测乘积在 0.476–6.881 deg·mm 之间，散布 14.4 倍，不是常数。** 现已打印散布倍数、整表标 `UNSUPPORTED EXTRAPOLATION`，JSON 增加 `bone_angle_extrapolation_supported: false` 与说明字段。重跑数字完全复现（3.675 deg·mm、5.212°、64.1 µm）——**被否掉的是外推，不是测量**。

---

### 1.4 力标度：1000 倍单位错误（你提示"查查往期 log"，你是对的）

**这是我犯的第三个错误，而且方向是反的**：我把"力标度 11,400 × 体重"当成**未解决的研究级缺陷**列出，还建议下一轮做。**它早在往期会话里就被诊断为纯单位错误并修好了**，往期 log 的原话是：

> 单位换算错误（把模型力单位当 mN）。模型质量单位是**克**，`m·g = 1.02431e-3 × 9810 = 10.0485`

我只查了项目目录里的 `*.log`（那些是 P1–P4 细胞力学阶段的，不含力的讨论），**没有去查 `~/.dsh/sessions/` 的会话记录** —— 那才是"往期 log"。找到命中 87 和 76 处的那两个会话后才看到。

#### 错误本体

`/run/media/sensen/Data2/cell_wound_prototype/run_tactile_record.py` 原第 91–95 行：

```python
#: the fly's mass is 1.02e-3 kg), so a force of one model unit is kg*mm/s^2 = 1e-3 N = mN
MODEL_FORCE_TO_N = 1.0e-3
```

**这句注释是自我矛盾的**：按 kg 读，模型果蝇重 **1.024 g**，是真果蝇（约 1 mg）的 **1000 倍**。模型质量属性 `body_mass.sum() = 0.00102431` 的正确单位是**克** → **1.02431 mg**（正确）。于是力单位是 `g·mm/s² = **1e-6 N = 1 µN**`。

**独立印证（定论依据）**：关掉致动器让果蝇静置，六腿法向力之和实测 **6.59 / 8.85 / 11.60 / 15.31 模型单位**，而体重 = `1.02431e-3 × 9810 = **10.0485 模型单位**`。静止物体必须被自身体重支撑 —— **它在模型单位下是平衡的**，所以模型单位就是微牛。

#### 修复

| 文件 | 改动 |
|---|---|
| `run_tactile_record.py` | `MODEL_FORCE_TO_N` **1.0e-3 → 1.0e-6**；注释重写（含两条独立印证） |
| `engine/embodied/scene.py` | 文档头"total body_mass sums to **1.0243e-3 kg** … weight is **10.05 mN** … 本模型比真果蝇重三个数量级"→ 改为**克 / 10.0485 µN / 模型质量是准确的**；`MODEL_UNITS_NOTE` 的"radians and **kilograms** … 1e-3 N = mN"→ 克 / µN；账本 `body_mass_total` 单位 kg→g、`weight_total` 10.048 mN→10.0485 µN；第 1533 行运行时提示的 mN→µN |
| `engine/embodied/adapters.py` | "CONTACT-FORCE CALIBRATION IS UNRESOLVED" 一节重写：单位已定，**残留的 ~16 倍**是另一个问题（该通道含位置伺服内应力，非外部载荷） |
| `make_status_report.py` | 状态条目由"UNRESOLVED"改为"UNIT ERROR FIXED + 残留 ~16× 待解释" |
| `TACTILE_RECORDING.zh-CN.md` | 3.3 节整体作废并加更正批注 |
| `PROGRESS_AND_MIXUP.zh-CN.md` | 触觉强度行改为已更正数字 |
| `outputs/tactile/*/tactile.npz` | **已用新常量重跑**（10 段 × 400 帧；旧数据备份于 `/tmp/tactile_backup_before_unitfix`） |

#### 改正后的实测数字（本次重跑验证，非引用）

| | 值 |
|---|---|
| 每帧六腿合力 均值 | **4.29 – 12.16 × 体重**（10 段） |
| 每帧六腿合力 峰值 | **25.26 – 27.79 × 体重**（10 段，高度一致） |
| 稳态行走应有 | ≈ 1 × 体重 |
| 残留偏差 | 均值约 4–12 倍、峰值约 26 倍（**不是 11,400 倍**） |

核对一致：记录器打印的 `force_peak_uN = 257.66` 恰等于 `25.641 × 10.0485`，即每帧六腿**合力**峰值。

**仍未解决、且不是单位问题的**：该残留（~4–26 ×）尚未解释。已排除致动器（`actuator_gain` 1000× 与 `actuator_forcerange` 130× 只各带来 18 % / 16 % 变化），方向指向刚性接触响应，幅度需按新尺度重估。

---

### 1.5 重建改为**从图像重新检测**斑块（使重建不再依赖录像端的斑块过滤）

**文件**：`/run/media/sensen/Data2/cell_wound_prototype/run_arena_reconstruct.py`

**问题**：重建脚本第 8 行明写它读 **"ONLY the ``px/`` group"** —— 即录像时由录像脚本自己的检测器产出的候选表。而那个检测器曾用**写死的 60 px** 作斑块面积上限（只对 200 行的帧成立），在 640×800 下把约占 490 px 的基准点整批当"过大斑块"拒掉。**这是"读数被污染"，任何门限调整都救不了 —— 基准点根本不在表里。**

**为什么这是可修的**：标定脚本 `run_arena_calibrate.py` 与 `tools_limb_eval.py` **从来都是从 PNG 图像重新检测**（阈值 ≥240、连通域 ≥3 px），**绕过**了这个 bug —— 这正是它们能在重建用不了的 episode 上给出有效结果的原因。

**修复**：新增 `blobs_from_png(path, thr=240.0, min_px=3)`，并加 CLI `--blobs {png,recorded}`（**默认 `png`**）；结果 JSON 记录 `blob_source` 与 `inputs_read` 以便追溯，`recorded` 保留用于复现历史运行。

**交叉验证（证明这是等价替换，而非另一种测量）**——同一个 seed3 上两条来源给出**完全相同**的结果：

| 斑块来源 | 帧 | 三角化基准点 | 世界系 RMS 中位 |
|---|---|---|---|
| `png` 重新检测 | 46/46 | 7/12 | **0.8581 mm** |
| `recorded` 录制表 | 46/46 | 7/12 | **0.8581 mm** |

**效果**：seed0（候选表命中率仅 1.4%，原本不可用）现在给出 **60/60 帧、7/12、世界系 RMS 0.8554 mm**，且经检验这 7 个是**真基准点**（其中 5 个与声明位置吻合到 **0.031–0.186 mm**；草屑不可能落在声明的 x,y 上）。

**同时修正了一处方向写反的注释**：原注释称该门限在更高分辨率下"要按比例收紧"，这与物理相反 —— 门限是**物理容差**，行数增加时应**按帧高放大**（见 1.1）。

---

## 2. 重要发现：`arena_bare_seed0` 是一个**损坏的 episode**

> ### 2.0 第二次修订（2026-10-08，本节结论已被修复并**部分撤回**）
>
> 本节下面的诊断（录像候选表无基准点）**成立**，但由它推出的"因此那些数字都不可用"**是错的**，
> 而且**已经修好**。修正要点：
>
> 1. **根因是可修的，已修**。标定脚本（`run_arena_calibrate.py`）与 `tools_limb_eval.py`
>    从来都是**直接从 PNG 重新检测**斑块，**绕过**录像端的 `max_px` bug —— 只有重建脚本
>    死读 `px/*`（第 8 行明写 "ONLY the ``px/`` group"）。现已给重建加上同样的能力：
>    `blobs_from_png()` + `--blobs {png,recorded}`，**默认 `png`**，结果里记录 `blob_source`。
> 2. **等价性已交叉验证**：在 seed3（候选表完好，命中率 93.8%）上，两条来源给出
>    **完全相同**的结果 —— 46/46 帧、7/12、世界系 RMS **0.8581 mm**。所以 PNG 路径是**等价替换**，
>    不是另一种测量。
> 3. **seed0 已恢复可用**：60/60 帧、7/12、世界系 RMS **0.8554 mm**，逐帧中位 0.8543。
>    并且**验证了这 7 个是真基准点而非草屑**：其中 5 个与声明位置吻合到 **0.031–0.186 mm**
>    （草屑不可能落在声明的 x,y 上）。
> 4. **因此 round4 的 0.8828 mm 是有效的**，与新的 0.8554 mm 在 3% 内一致。
>    我先前在 `ARENA_FIX_ROUND4.zh-CN.md` 里写的"勿引用"批注**已撤回**。
>    教训：**读数不可信 ≠ 结论不可信** —— 我在同一条链上连着犯了两次方向相反的错。
> 5. **新发现、仍未解释**：那 0.855 mm 的 RMS 完全由**两个点**主导 —— 顶层 mk9、mk10
>    各低 **2.0259 mm**（两点数值**完全相同** ⇒ 系统性），而另外 5 点是 0.031–0.186 mm。
>    同为 z=11 的环心点 mk11 却是准的（+0.031 mm）。**所以骨骼末端位置误差该用的尺度是
>    ~0.03–0.19 mm，不是 0.86 mm。** 顶层偏心基准点的系统性偏差是下一步要查的。
>
> 下面原文保留，以便追溯诊断过程。

### 2.1 原始诊断（成立，但推论已按 2.0 撤回）

### 证据

| 检查 | seed0 | seed3（对照） |
|---|---|---|
| 帧尺寸 | 800×640 | 800×640 |
| 每相机每帧候选斑点数 | 229–370 | 228–398 |
| **12 个基准点 × 30 帧 × 6 相机 = 2160 次投影中，有候选落在 3 px 内的比例** | **1.0 %（21/2160）** | **93.8 %（2025/2160）** |
| 投影到最近候选质心的全局最小距离 | 10.9 px（仅 1 例 0.58 px，属偶然草屑） | 0.6 px |

### 我直接看了图（这是判据，不是推断）

把 12 个基准点的**投影位置**与**最近检出斑点**画到真实帧 `f00000.png` 上，再对前 6 个点做 8 倍放大：**每个红叉（投影）旁边确实有基准点的白色饱和圆斑，半径约 8–12 px**。也就是说——

- 基准点**在 PNG 图像里**（渲染正常，肉眼可见）；
- 基准点**不在录像记录的 `px/*` 候选表里**。

### 根因

`run_arena_record.py` 早期的斑块尺寸门限是**写死的 `max_px=60`**，这个值只对 200 行的帧成立（那时 0.30 mm 半径的基准点约占 17 px）。在 640×800 下同一个基准点占约 **490 px**，于是**被判为"过大斑块"整批拒掉**，只剩下草屑。修法已存在于当前代码：

```python
blob_max_px = int(max_px) if max_px else int(round(60.0 * (_H / 200.0) ** 2))   # = 960 px at H=800
```

但 **seed0 是在这个修法之前录的**，所以"磁盘上这个 episode"是坏的。

### 牵连与不牵连（这里我要更正自己一次）

- **不牵连**：`/run/media/sensen/Data2/cell_wound_prototype/tools_limb_eval.py` **不读录像的 `px/*`**，它用 `blobs_from_png()` **从 PNG 重新检测**（`min_px=20`，基准点约 300–550 px）。所以它**绕过**了录像的 bug，`limb_eval.json` 里那 6 个静态基准点的 11–83 µm 是**图像直接测得的，有效**。
- **牵连**：`reconstruction.json` 这一类**读录像 `px/*`** 的产物，在 seed0 上是**没有基准点可用的**，其数字无意义。

### 关节角精度该用哪个数（第三次修订，2026-10-08）

我在这里先后给过两个互相矛盾的结论，两个都错，现在给出定稿。

- **第一次**：用 **0.883 mm** 当"实测 3D 误差"，得股骨 **57°**。
- **第二次**（本节原文）：说 0.883 mm 不可靠，改用 **65 µm**，得股骨 **6.7°**。
- **定稿**：**0.883 mm 那个读数本身是有效的**（见 2.0 节：PNG 路径独立复现为 0.8554 mm），
  第二次的"不可靠"说法**已撤回**。但 **57° 仍然是错的**，原因不是它的读数，而是**用错了口径**：

| 口径 | 值 | 该不该用来推骨骼末端 |
|---|---|---|
| 全部 7 个基准点、未加门限的世界系 RMS | **0.855–0.883 mm** | ❌ **不该** —— 它完全由顶层 mk9/mk10 两点各低 2.0259 mm 主导 |
| 同上、加 3 px 重投影门限（`limb_eval.json`） | **65 µm** | ✅ 该用 |
| 逐基准点中位误差（5 个吻合良好的点） | **0.031–0.186 mm** | ✅ 该用（这是原始观测，非拟合量） |
| 8 根骨的**关节角误差直接测量**（跟踪器，`limb_tracking.json`） | **中位 1.67–3.69°（股骨）/ 0.74–1.45°（胫骨）** | ✅ **这是最该引用的数** |

**所以："骨骼末端位置误差 ~0.03–0.19 mm"是本次能给出的尺度；关节角精度引用跟踪器的直接测量
（全表 173 个角，中位 1.730°、p95 10.791°）。** 57° 与 6.7° 都是把不同口径套到骨长上的产物。

**仍未解决、下一步要查**：顶层偏心基准点（mk9、mk10，声明 z=11 mm）系统性低 **2.0259 mm**，
而同一高度的环心点 mk11 是准的。这是当前 3D 精度的真正瓶颈。

---

## 3. 我已删除的无效产物（本轮我自己生成，非交付物）

| 文件 | 为什么无效 |
|---|---|
| `outputs/arena/arena_bare_seed0/reconstruction_gatefix.json` | 有效 16 px 门限下 59/60 帧"拟合成功"，但那是拟合**随机草屑**（12 个基准点只有 3 个凑够 2 视图，且投影距最近候选 10.9 px 起）。世界系 RMS 0.119 mm 是假数字。 |
| `outputs/arena/arena_bare_seed0/reconstruction_gate4px.json` | 有效 4 px 门限下 0/60 帧，无任何有效结果。 |

**两者都不是修复的证据，已删除，未交付。**

---

## 4. 已执行的删除（经你确认：只删派生数字与已废弃探针，保留 seed0 的 PNG）

### 已删除

| 项 | 绝对路径 | 理由 |
|---|---|---|
| 废弃探针 | `/run/media/sensen/Data2/cell_wound_prototype/tools_axis_probe.py` | 建立在被推翻的前提上（"列错数十 px ⇒ 图像轴约定错"）；其 0.15 mm 标记被草淹没（62 个饱和像素来自草、15 个来自标记），无有效结论；被 `tools_axis_probe2.py` 明确否定 |
| 废弃探针 | `/run/media/sensen/Data2/cell_wound_prototype/tools_one_marker.py` | 同一被推翻前提下的"最小复现"，复现的是修复前的坏状态 |
| 废弃探针 | `/run/media/sensen/Data2/cell_wound_prototype/tools_clean_probe.py` | 同上；其输出即下一行的派生数字 |
| **派生数字** | `/run/media/sensen/Data2/cell_wound_prototype/outputs/arena/clean_probe.json` | 记录的是修复前状态（行对、**列恒差约 48 px**），会被误读为新的轴 bug 证据 |
| **派生数字** | `/run/media/sensen/Data2/cell_wound_prototype/outputs/arena/arena_bare_seed0/reconstruction.json` | 从录像 `px/*` 重建，而该候选表**不含基准点**（2160 次投影仅 1.0% 命中，seed3 为 93.8%），世界系 RMS 2387 mm 无意义 |
| 桌面副本 | `/home/sensen/Desktop/cell_wound_prototype/arena_grass/tools_axis_probe.py`、`tools_one_marker.py`、`tools_clean_probe.py` | 同上三个探针的副本 |

### 明确保留（附理由）

| 项 | 理由 |
|---|---|
| `/run/media/sensen/Data2/cell_wound_prototype/tools_axis_probe2.py` | 它给出了**最终有效**结论（+x/+y/+z 位移 cos = **1.000**），被 `run_arena_calibrate.py` 第 8 行引为证据，方法有效 |
| `/run/media/sensen/Data2/cell_wound_prototype/tools_arena_calib_check.py` | 把基准点投影画到真实帧上；**本轮发现 seed0 损坏所用的正是这个方法**，应保留 |
| `/run/media/sensen/Data2/cell_wound_prototype/tools_dlt_check.py` | 从 6 个已知点反解 P 矩阵，方法有效 |
| `.../outputs/arena/arena_bare_seed0/frames/cam{0..5}/*.png`（6×107 张，约 684 MB） | **你指定保留**。它是"图像重新检测"（`tools_limb_eval.py` 走 `blobs_from_png`）唯一可用的证据来源；删掉就无法再验证本轮的诊断 |
| `ARENA_FIX_ROUND4.zh-CN.md`、`ARENA_ACTION_PARSING_READINESS.zh-CN.md`、`ARENA_GRASS_STATUS.zh-CN.md` 的相关段落 | **你未选择删除文档段落**（我建议的也是不删）。已就地加**更正批注**指向本文，保留"为什么 0.8828 mm 不可信"这段推理的可追溯性 |

删除清单全文：`/home/sensen/Desktop/cell_wound_prototype/repair_audit/删除清单_DELETED.txt`

### 删除后的回归验证（全部通过）

```
语法检查 run_arena_reconstruct.py / tools_limb_eval.py / limb_tracking.py /
         tools_limb_angle_series_figure.py ................ OK
门限归一化：200 行上 2.5/3.0/4.0 逐位等于历史值 ......... PASS
            640×800 (H=800) 下 3.0 -> 12.0 px ............. PASS
无残留代码依赖（grep --include=*.py）..................... 空
seed0 PNG 保留：cam0..cam5 各 107 张 ................... OK
```

---

## 5. 复现

```bash
cd /run/media/sensen/Data2/cell_wound_prototype
# 门限归一化的逐位不变自检
venv/bin/python -c "import sys;sys.path.insert(0,'.');from run_arena_reconstruct import gate_px_at_rows as g;assert g(3.0,200)==3.0;print('PASS', g(3.0,800))"
# 基准点是否在候选表里的判据（本轮的判据）
#   seed0: 1.0%   seed3: 93.8%   —— 见本文第 2 节表
```
