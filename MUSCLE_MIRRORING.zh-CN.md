# 把肌肉从 1 条腿镜像到 6 条腿：做了什么、拿到了什么、还缺什么

## 一、"找别的项目"的结论：**没有**

当前唯一、也是最先进的肌肉模型就是 **Özdil, Ning, Phelps, Wang-Chen, Elisha, Blanke, Ijspeert, Ramdya,
"Musculoskeletal simulation of limb movement biomechanics in Drosophila melanogaster"
（arXiv 2509.06426, ICLR 2026）**。它自己写着：

> **"We modeled 15 muscle-tendon units (MTUs) per foreleg"**

**FlyGym 里那 15 条肌肉的模型就是这一篇的模型，作者声明只做了前腿。** 排除理由也写了：tibia 里的肌肉因
X 光扫描未覆盖 tibia 而排除，trochanter 肌肉因功能不明而省略。真实解剖是**每腿约 19 条肌肉、约 69 个运动神经元**。

**而它给出了镜像的权威依据**：

> **"Notably, the muscle structure across legs is nearly identical, with a few exceptions including
> tergal depressor of the trochanter (TDT) muscles in the middle legs that facilitate jump escape."**

## 二、神经侧本来就齐了（实测）

BANC 里**六条腿的运动神经元全都有，按腿神经分开、按肌肉命名**：

| 腿 | 腿神经 | 肌肉命名运动神经元 |
|---|---|---|
| 前左 / 前右 | prothoracic | 29 / 29 |
| 中左 / 中右 | mesothoracic | 45 / 46 |
| 后左 / 后右 | metathoracic | 48 / 46 |
| | | **合计 243** |

而且各腿数目高度一致（`LFTibia_flex`→每腿 5 个、`LFTibia_extensor`→每腿 2 个、`LFF_trochanter_flexor`→中/后每腿 7 个），
**这本身就是镜像依据的独立佐证**。产物 `outputs/mn_muscle_map_per_leg.json` 里每条肌肉×每条腿都带神经元 ID。

## 三、镜像实施（`tools_mirror_muscles.py`）

生成 `outputs/muscles_six_legs/fruitfly_six_leg_muscles.xml`：

- **90 条肌腱 + 90 条肌肉执行器，六条腿各 15 条**
- 自一致性：**231 个 site、0 重名、222 个引用全部解析、90 条肌肉执行器→肌腱引用全部解析**
- **编译通过**（经 FlyGym 的网格加载器）：`nq=14 nv=14 nu=90 ntendon=90 muscle_actuators=90`
- **与原始模型逐项对照**：

| 量 | 原始（1 腿） | 镜像后（6 腿） | 差 |
|---|---|---|---|
| nu（执行器） | 15 | **90** | **+75** |
| ntendon | 15 | **90** | **+75** |
| nq / nv / njnt | 14 / 14 / 14 | 14 / 14 / 14 | **0** |
| nbody / ngeom | 73 / 72 | 73 / 72 | **0** |
| 总质量 | 0.00249427 | 0.00249427 | **0** |
| 原 15 条执行器 | — | **全部保留** | — |

**只有执行器和肌腱增加，其它一个没动。**

### 镜像不是一一对应，两处被我实测出来并记录

1. **只有前腿有 trochanter 体段**：前腿 9 段（Coxa, Trochanter, Femur, Tibia, Tarsus1-5），中/后腿 8 段（**无 Trochanter**）。
   所以 trochanter 上的 site 映射到该腿的 **femur**，**24 个 site 受影响**，逐条记录在 `mirror_decisions.json`。
2. **各腿骨长不同**（实测 rest pose）：femur 前 0.575 / 中 0.784 / 后 0.836 mm；tibia 0.511 / 0.667 / 0.685。
   纯刚体变换会把肌肉留在前腿的**绝对**偏移上，所以每个 site 再按所在体段的**轴向长度比缩放**。
   **这是近似**：横向走的肌纤维本应只做轴向缩放，我用的各向同性缩放。
3. **7 条肌肉是胸肌，site 挂在共享的 `Thorax` 上**。用"体段到体段"的变换对它们**等于恒等变换**（根本不动，会被插到错误的腿窝），
   所以改用**以该腿 coxa 为锚的世界变换**（45 个 site 走这条路径）。

## 四、还缺什么：**中/后腿根本没有关节**

这是镜像之后暴露出来的**真正阻塞**。模型只有 **14 个关节，全部在前腿**：

```
joint_LFCoxa_yaw/pitch/roll, joint_LFTrochanter_yaw/pitch/roll, joint_LFTibia_pitch   (7)
joint_RFCoxa_roll/yaw/pitch, joint_RFTrochanter_yaw/pitch/roll, joint_RFTibia_pitch   (7)
```

**中腿（LM/RM）和后腿（LH/RH）没有任何关节** —— 它们的 body 是焊在胸上的静态几何。

意味着：

- **真收获**：**右前腿原本有 7 个关节却一条肌肉都没有，现在它有 15 条 → 真正可肌驱的腿从 1 条变成 2 条。**
- **没到位的**：中/后腿那 60 条肌肉虽然编译通过、结构正确，但**拉的是不能动的 body**，产生不了运动。

## 五、下一步（明确、可做）

**给中/后腿补关节**，按其自身解剖对照前腿的关节组（`Coxa yaw/pitch/roll` + `Femur pitch` + `Tibia pitch`，
**无 Trochanter 段所以 3+1+1**），4 条腿 × 5 = **20 个关节**。补完之后：

1. 中/后腿的 60 条镜像肌肉才有作用对象；
2. 每腿的肌肉↔运动神经元映射（`mn_muscle_map_per_leg.json`）才能接到具体关节上；
3. 才谈得上"整只动物的行为由神经元经肌肉产生"。

**在那之前，"6 条腿都能被神经控制"是不成立的，我不打算说它成立。**

## 六、诚实边界

- 镜像的 site 位置**对 5 条腿是推导值，不是实测解剖**；已写进生成文件的 XML 注释与 `mirror_decisions.json`。
- MN→肌肉映射是**按命名匹配推导**（`DERIVED`），权威表在 Azevedo et al.（bioRxiv 2022.12.15.520299），本次**未取到全文**，未经其核对。
- 源模型本身缺 tibia 肌肉、缺 trochanter 肌肉、**缺中腿 TDT**，镜像原样继承。

---

# 续：补全关节 + 修数值（已可用）

## 一、又抓到我自己的一个错误结论

上一节我写"右前腿现在有肌肉了 → 可肌驱的腿从 1 条变 2 条"。**这是错的。** 实测关节定义后发现：

**右前腿的 7 个关节被 7 条 `<equality>` 约束锁死**（`polycoef 0 0 0 0 0`），在源模型里是**刚性**的。
所以源模型**只有左前腿一条腿能动**；我只加肌肉不加关节/不解锁，那些肌腱拉的是不能动的 body。

## 二、补了什么

1. **给中/后腿补 28 个关节**（4 条腿 × 7）：`Coxa_yaw/pitch/roll + Femur_yaw/pitch/roll + Tibia_pitch`。
   中/后腿**没有 trochanter 段**，所以原来 trochanter 那 3 个自由度落到 **femur** 上。
   关节轴**不是照抄**：把前腿的轴经**同一条以 coxa 为锚的变换**转到目标腿自身的体段坐标系里，
   这样 yaw/pitch/roll 的**解剖语义**保持一致。
2. **可选解锁被锁的腿**（`--unlock-locked-legs`），删掉那 7 条 equality —— 默认不动，要显式开。
3. **按各自实测静息长度重标 `lengthrange`**（`--rescale-lengthrange`）。

## 三、修掉的数值 bug（我自己的，且是静默的）

**`parse_tree` 返回 `(位置, 旋转)`，而我在 site 变换处写成 `Rs, ps = world[body]` —— 正好反了。**
NumPy 没有报错（矩阵+标量、点积都会静默广播），后果是：

| | 修前 | 修后 |
|---|---|---|
| site 局部偏移量级 | 2.1–13.4 mm（源为 0.11 mm） | 正常 |
| `LMTibia_flex` 肌腱长度 | **22.21 mm** | **0.801 mm** |
| 镜像肌肉峰值力 | **21,000–33,000**（原始 137） | **137.2** |
| `lengthrange` 合规数 | LM 0/15 | **LM 12/15** |
| `unstable` 警告 | 大量 | **0** |

**并加了两道防线**（这次是它们把我拦住/提醒的）：
- 断言"被当作旋转用的矩阵必须真的是正交阵"（**这才是能抓住解包调换的检查**；我第一版写的"离体段原点距离不变"是错的——共享 Thorax 上的 site 在世界里**本来就应该移动**）；
- `lengthrange` 合规性逐腿统计。

## 四、最终验证（全部实测）

```
njnt=42  nq=42  nu=90  ntendon=90  muscle-typed actuators=90
lengthrange 合规（逐腿）: LF 12/3  RF 12/3  LM 12/3  RM 12/3  LH 12/3  RH 12/3
```

**六条腿完全一致**，而且那 3 条超范围与**源模型左前腿自身的 3 条相同** —— 是源的性质，不是镜像引入的。

功能测试（满激活 0.2 s，逐条测一条肌肉）：

| 肌肉 | 关节 | 峰值力 | Δ关节 rad | Δ肌腱 mm | |
|---|---|---|---|---|---|
| `LFTibia_flex`（原始） | LFTibia_pitch | 136.62 | 2.327 | 0.027 | MOVED |
| `RFTibia_flex`（镜像） | RFTibia_pitch | **136.69** | 1.660 | 0.018 | MOVED |
| `LMTibia_flex`（镜像） | LMTibia_pitch | **137.20** | 0.233 | 0.019 | MOVED |
| `RMTibia_flex` | RMTibia_pitch | 137.20 | 0.233 | 0.019 | MOVED |
| `LHTibia_flex` | LHTibia_pitch | 137.10 | 0.314 | 0.020 | MOVED |
| `RHTibia_flex` | RHTibia_pitch | 137.10 | 0.314 | 0.020 | MOVED |
| `LFC_pleural_remotor`（原始） | LFCoxa_pitch | 57.51 | 0.668 | 0.020 | MOVED |
| `LMC_pleural_remotor`（镜像） | LMCoxa_pitch | **58.02** | 0.188 | 0.003 | MOVED |

**六条腿的力与原始腿一致到 0.4–0.9%，全部 MOVED，0 条 unstable。**

## 五、现在的状态与剩下的边界

**已经成立**：六条腿都有**可动的关节**和**能产生正确量级力的肌肉**，且与源模型逐腿一致。

**仍然不是"神经元在控制"**：
- 这套肌肉还没有接到连接体上。要接的话，链路是
  **下行/运动神经元脉冲 → 90 条肌肉的激活 → 关节 → 身体**，
  而每腿的肌肉↔运动神经元映射（243 个神经元、带 ID）已经在 `outputs/mn_muscle_map_per_leg.json` 里。
- **中/后腿的肌肉是推导值，不是实测解剖**（镜像 + 长度缩放），这点写进了生成 XML 的注释与 `mirror_decisions.json`；
  论文用 NSGA-II 对**前腿**做过参数优化，中/后腿没有对应优化，**因此它们的动力学保真度低于前腿**。
- 源模型本身缺 tibia 肌肉、缺 trochanter 肌肉、**缺中腿 TDT**，镜像原样继承。

## 六、复现命令

```bash
# 生成六腿肌肉模型（90 MTU / 42 关节），解锁右前腿，并按实测长度重标 lengthrange
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 venv_body/bin/python tools_mirror_muscles.py \
    --unlock-locked-legs --rescale-lengthrange

# 功能验证：肌肉是否真的让关节动、力量级是否与原始腿一致、是否稳定
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 venv_body/bin/python tools_six_leg_functional_test.py
```
