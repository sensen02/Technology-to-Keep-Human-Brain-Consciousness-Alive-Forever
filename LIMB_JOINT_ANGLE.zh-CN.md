# 肢体关节角解析 —— 一个脚在 t 时刻的关节角度

**回答「你能解析肢体吗，例如一个脚在 t 时刻的一个关节角度，而不是只记录身体」：能，而且已经跑出来了 —— 但只在两个 episode 中的一个成立，并且精度只在模拟器上验证过。**

**这是手工搭建的原型，不是成品。** 没有任何一节代码是认证过的测量仪器；下文的"真值"全部是 MuJoCo 模拟器自己的数，不是真果蝇。

---

## 1. 直接可引用的结果

`t = 99.6 ms`，右中腿胫骨（`rm_tibia`）：

| 量 | 从 6 张相机图像量出来的 | 模拟器真值 | 误差 |
|---|---|---|---|
| 相对水平面仰角 | **−58.893°** | −59.034° | |
| 方位角 | **−98.172°** | −97.149° | |
| 骨方向夹角误差 | | | **0.546°** |
| 该帧看到的相机数 | 两端的标记分别被 5 台 / 4 台看见 | | |

`t = 122.7 ms`，同一条胫骨：仰角实测 **−55.884°** / 真值 −55.797°，方位 −119.964° / −120.234°，**误差 0.175°，两端各被 6 台 / 4 台相机看见**。

`t = 102.9 ms`，右前腿股骨（`rf_femur`，误差较大的那类）：仰角实测 **65.102°** / 真值 67.351°，**误差 2.254°**。

逐帧全表（173 个关节角）：
`/run/media/sensen/Data2/cell_wound_prototype/outputs/arena/arena_bare_seed3/limb_joint_angles.csv`

---

## 2. 汇总精度（seed3，46 帧，20 个肢体标记真实渲染并检出）

全表 173 个关节角，覆盖 46 帧中的 40 帧：

| | 误差 |
|---|---|
| 中位数 | **1.730°** |
| p95 | **10.791°** |
| 最大 | **28.37°** |

分骨（中位 / p95，单位度）：

| 骨 | 帧数 | 中位 | p95 |
|---|---|---|---|
| rh_tibia | 28 | **1.449** | 2.910 |
| rf_tibia | 27 | **0.814** | 10.064 |
| lm_tibia | 24 | **1.425** | 2.229 |
| rm_tibia | 15 | **0.742** | 2.450 |
| rh_femur | 19 | 3.388 | 11.763 |
| lh_femur | 18 | 3.691 | 12.197 |
| rf_femur | 22 | 2.815 | 26.563 |
| rm_femur | 16 | 1.669 | 6.647 |
| lm_femur | 3 | 6.260 | 6.303 |
| lf_femur / lf_tibia / lh_tibia | 0–1 | — | — |

**胫骨比股骨准，且这是有原因的**：胫骨长 0.987–1.352 mm，股骨只有 0.705–0.836 mm，同样的位置误差在更短的骨上放大成更大的角度误差。

标记三维位置误差：中位 **42.2 µm**、p95 **328.3 µm**（n=481）。**自洽性检验**：由 1.669° 的中位角度误差反推股骨末端误差得 `tan(1.669°)×0.784 mm = 22.9 µm`，由 1.449° 反推胫骨得 `tan(1.449°)×1.352 mm = 34.2 µm` —— 与实测的 42 µm 位置误差同量级，**说明跟踪器本身没有引入额外误差，精度完全由标记点重建精度决定**。

---

## 3. 瓶颈是"看得见"，不是"像素够不够"

seed3 实测可见率（帧内 ≥3 台相机看见的比例）：

| 标记 | 可见率 |
|---|---|
| c_thorax / c_abdomen3 | **1.000 / 1.000** |
| 6 条腿的 tarsus5 | 0.935 – 1.000 |
| 6 条腿的 tibia | 0.674 – 1.000 |
| 6 条腿的 **trochanterfemur** | **0.065 – 0.326** |

一个股骨关节角需要 trochanterfemur 和 tibia **同时**被看到，所以它的可行帧数被压到 3–22 / 46。**trochanterfemur 标记在体—腿交界处，被身体和草挡住的概率是 67–93%** —— 这是相机布点问题，加像素、换更高分辨率传感器都不解决。

噪声蒙特卡洛（0.5 px、假定完美标定）给出的对照：当前 r=22 mm / 200 行的环上，股骨 4.967°、胫骨 3.637°；把相机拉到 r=3 mm / 480 行则是 0.271° / 0.262°。**几何上并不缺精度，缺的是标定和可见性。**

---

## 4. 它并不总是成立 —— seed1 是失败的

`arena_bare_seed1`：只有 **96/920** 个帧-标记槽被接受；**连 c_thorax 和 c_abdomen3 的可见率都是 0.000**（整只果蝇朝向背对相机并被草遮挡）。结果是 12 根骨里只有 9 帧凑出了 1 个角度，中位误差 1.276°。

**两个 episode 里有一个接近完全失败。** 不能把 seed3 的结果当成"系统能力"，它只是一个好样本。

---

## 5. 已纠正的一个错误结论（重要）

先前 `limb_eval.json` 里给出"股骨关节角误差 5.21°"，依据是"`角度×间距` 乘积恒定，所以可以从 6–12 mm 的基准点对下推到 0.7 mm 的骨"。**这个前提被它自己的数据推翻了**：实测乘积在 **0.476 – 6.881 deg·mm** 之间，散布 **14.4 倍**，不是常数。

已改：`/run/media/sensen/Data2/cell_wound_prototype/tools_limb_eval.py` 现在打印散布倍数、把整张换算表标为 `UNSUPPORTED EXTRAPOLATION`，并在 JSON 里写 `bone_angle_extrapolation_supported: false` 与说明字段。重跑后数字完全复现（3.675 deg·mm、5.212°、64.1 µm），可复现性本身没问题，**被否掉的是外推，不是测量**。

可以引用的每骨数字是第 2 节跟踪器**直接测这条骨自己的两个端点**得到的，不需要任何外推。

---

## 6. 明确的边界（不能声称的东西）

- **真值是模拟器**，不是真果蝇。整套东西没有任何生物测量。
- **第 0 帧的身份来自模拟器**（`limb_tracking.py:281`）：所有标记都是同一个自发光小球，像素里没有身份信息，真系统靠人工标注第一帧（DeepFly3D 就是这么做）。这里没有人标注，所以第 0 帧用模拟器当真值的替身，**之后每一帧只靠 4 道门限**，且跟踪器自己上报误差，所以这个替换藏不住。
- **`limb_eval.json` 用的 seed0 里，果蝇自身的标记从未真正渲染**（`fly_markers_rendered: false`，编译后的模型里 0 个 `mk_*` 几何体），所以那份评估没有果蝇标记实测可对。
- 力/触觉那条链路与本文件无关，其力标度仍不可信（见 `TACTILE_RECORDING.zh-CN.md`）。
- 未做：真实果蝇验证、身份交接的人工标注、多 episode 统计（目前评估了 2 个）、trochanterfemur 遮挡的相机重新布点。

---

## 7. 复现

```bash
cd /run/media/sensen/Data2/cell_wound_prototype
OPENBLAS_NUM_THREADS=1 ./venv/bin/python limb_tracking.py --episode outputs/arena/arena_bare_seed3
OPENBLAS_NUM_THREADS=1 ./venv/bin/python tools_limb_angle_series_figure.py arena_bare_seed3
OPENBLAS_NUM_THREADS=1 ./venv/bin/python tools_limb_angle_precision.py
```

产物绝对路径：

- 逐帧关节角 CSV：`/run/media/sensen/Data2/cell_wound_prototype/outputs/arena/arena_bare_seed3/limb_joint_angles.csv`
- 逐帧序列图：`/run/media/sensen/Data2/cell_wound_prototype/outputs/arena/arena_bare_seed3/limb_angle_series_figure.png`
- 汇总 JSON：`/run/media/sensen/Data2/cell_wound_prototype/outputs/arena/arena_bare_seed3/limb_tracking.json`
- 可见率：`/run/media/sensen/Data2/cell_wound_prototype/outputs/arena/marker_coverage.json`
- 静态评估（含纠正标记）：`/run/media/sensen/Data2/cell_wound_prototype/outputs/arena/limb_eval.json`
- 噪声蒙特卡洛：`/run/media/sensen/Data2/cell_wound_prototype/outputs/arena/limb_angle_precision.json`
