# 找到根因了：**模型声明了解剖行程，却把限位关掉了**

---

## 0. 先纠正我上一轮的两个错误结论

| 我上轮说的 | 真相 |
|---|---|
| 「**没有一条腿能承担 1/6 体重**」 | **错。** 那个试验给一只脚加 1/6 体重的同时，六条腿还得撑住全体重，等于要它们撑 **1.167 倍体重**——果蝇当然一直掉到肚子撞地。我量到的 1.33 mm **根本不是柔度**，而是「坐好高度 2.03 − 肚子贴地高度 0.69」这个距离。 |
| 「**接触参数是元凶**」 | **也错。** 把接触时间常数从 0.02 扫到 0.0005（**刚度提高 1600 倍**），塌陷**一模一样是 1.33 mm**——因为它被肚子贴地卡住了，根本没在量接触。 |

**正确的隔离试验**：把身体**焊住不动**，再给一只脚加载。结果：

| 腿 | LF | RF | LM | RM | LH | RH |
|---|---|---|---|---|---|---|
| 1/6 体重下脚位移 (mm) | 0.030 | 0.019 | 0.012 | 0.004 | 0.005 | 0.004 |
| 竖直刚度 (µN/mm) | 54 | 89 | 133 | 438 | 363 | 420 |

**腿其实是刚性支柱**（0.004–0.03 mm 偏转）。**「腿不够硬」不是问题。**

---

## 1. 真正的 bug：**42 个关节里 32 个的限位是关掉的**

源模型在**每个**关节上显式写着：

```xml
<joint name="joint_LFCoxa_yaw" limited="false" range="-0.597 0.2745" springref="-0.11"/>
```

**它声明了解剖行程 `range`，却同时写 `limited="false"`。** 而编译器是 `<compiler autolimits="true"/>`——**显式的 false 优先**，于是**行程声明了但完全不执行**。

**实测后果**：让身体只能竖直移动（锁住转动）后，果蝇下沉 1.39 mm，而**最坏的关节跑到了限位之外 110% 的量程**：

```
joint_RHCoxa_yaw   -110%   （在下界之外 110%）
joint_RFCoxa_roll   -60%
joint_LHCoxa_yaw    -27%
```

**关节可以穿过自己的解剖行程**，所以腿会折进任何昆虫腿都到不了的姿态，果蝇就这样塌下去。

**佐证**：质量扫到 0.25 mg 时越界只有 4%，塌陷就只有 0.0275 mm——**越界多少，塌陷多少**，一一对应。

---

## 2. 修复：**把模型自己声明的行程真正启用**

**没有发明任何东西**——行程就是模型自己的（来自它的 OpenSim 转换），只是让它成立：

```python
for j in root.iter("joint"):
    if j.get("name") and j.get("range") and j.get("limited") != "true":
        j.set("limited", "true")     # 42 个关节
```

**实测效果**：

| | 修复前 | **修复后** |
|---|---|---|
| 跗节承担体重 | 0.445 bw | **0.909 bw** |
| 沉降后胸节高度 | 0.6465 mm | **0.7426 mm** |
| **身体压地** | 0.60 bw | **0.19 bw** |
| 接地腿 | 6/6 | 6/6 |

刚坐好那一刻（t=0，up = 1.0000，身体不碰地）**六足全部接地、跗节承担 0.9485 bw**。

---

## 3. 还没解决的部分

**果蝇仍然蹲到 0.74 mm，还有 19% 的体重压在身上**，没有站直。所以这是**进展，不是完成**。

剩下的疑点（按我看到的顺序）：

1. **站姿需要重新解**：这次 IK 是在**限位未启用**的模型上解的，所以贴地散布是 0.106 mm（不齐）。**应该在限位启用后重解站姿**，让六足同时齐平接地。
2. **质量仍然偏大 2.44 倍**（模型 2.494 mg，真果蝇约 1.0 mg）。
3. **转动刚度仍然近乎为零**（只锁平移时身体能自由转到 49–136°），需要单独查。

**判据也要改**：我先前用「1.5 s 时的瞬时状态」和「胸节 > 1.6 mm」判定，两者都不对——**必须在真正平衡后判（|vz| ≈ 0）、且判据是「六足接地且身体任何部位不碰地」**，而不是我随手定的高度阈值。

---

## 4. 复现

```bash
# 重建站姿模型（现在会强制启用 42 个关节已声明的解剖行程）
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 venv_body/bin/python tools_stand_and_ground.py

# 隔离测腿刚度（身体焊住）
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 venv_body/bin/python diagnostics/diag_leg_stiffness_welded.py
# 沉降模式分解（锁转动 / 锁平移）
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 venv_body/bin/python diagnostics/diag_sag_mode.py
# 接触参数扫描（证明不是接触）
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 venv_body/bin/python diagnostics/diag_contact_scale.py
```

**产物**：`outputs/leg_stiffness_welded.json`、`outputs/sag_mode.json`、`outputs/contact_scale.json`、
`outputs/muscles_six_legs/fruitfly_six_leg_standing.xml`（42 个关节限位已启用）、
`outputs/standing_shots/limits_on_*.png`
