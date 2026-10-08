# 修复轮 2：果蝇模型调研结论 + 卡点的新进展（**仍未修好**）

> **手工搭建的工程原型。第 4 节是这轮没修成的。**

## 1. 你问的「有果蝇的模型吗」——有，而且不止一个

### 1.1 本机**已经装了**三套果蝇身体模型

| 模型 | 在哪 | 状态 |
|---|---|---|
| **NeuroMechFly** | `flygym/compose/fly/neuromechfly.py` | 本项目**一直在用**：69 个 body、六条腿各 7 DOF、按 `JointPreset.LEGS_ONLY` |
| **FlyBody** | `flygym/compose/fly/flybody.py` | 论文模型（Lobato-Ríos 2022），带肌肉驱动 |
| **FlyMusculoskeletal (FlyMimic)** | `flygym/compose/fly/musculoskeletal.py` | FlyGym 自带 docstring 标注 **experimental**；15 条左前腿 Hill 型肌肉；**只驱动左前腿** |

所以你担心"只有人类的"——**身体模型不缺**。缺的是**姿态骨架定义**。

### 1.2 运动捕捉/姿态估计：有果蝇专用的，而且**正是多相机**

| 资源 | 关键事实（来自论文原文） |
|---|---|
| **DeepFly3D**（Günel et al., eLife 2019, doi:10.7554/eLife.48571）| **7 台相机**，**每个动物 38 个关键点**：每条腿 5 个（thorax-coxa、coxa-femur、femur-tibia、tibia-tarsus 关节 + pretarsus）、腹部 6 个（每侧 3 个）、每根触角 1 个（测头部旋转） |
| 同上，**标定方法** | 论文明确说：给 2.5 mm 的动物做多相机配准需要"prohibitively small checkerboard"，所以**不用外部标定板，直接拿果蝇自己当标定目标**，用捆绑调整同时解出 3D 点和每台相机的外参（每相机 10 DOF = 3 平移 + 3 旋转 + 4 畸变）。他们还用**对极几何**剔除错误 2D 点：相机成环朝内，对极线近似水平，所以正确的对应点应落在几乎同一图像行 |
| 同上，**标记的态度** | 论文明确说标记在亚毫米肢体上"likely hamper movements and are difficult to mount"，**而且每条腿一两个标记不足以描述 3D 肢体运动学** |
| **DeeperFly** | NeLy-EPFL 的后续版本（[repo](https://github.com/NeLy-EPFL/deeperfly)） |
| **Anipose** | 有果蝇配置（`config_fly-anipose.yaml`），Tuthill lab |

**结论：你要的"多视角 + 果蝇专用骨架"确实存在，而且我上一轮的做法（贴 0.8 mm 大珠子）方向是错的**——DeepFly3D 的论文正好说标记方案在果蝇上不合适：珠子会妨碍运动，而且**每腿 3 个点不足以定出完整 3D 肢体运动学**。正确的路子是**无标记 2D 关键点检测 + 多视角三角化 + 骨架先验（骨长分布）+ 由图结构做一致性修正**。

而**我卡住的那个问题，DeepFly3D 的答案正好也是解法**：不要相信声明的相机位姿，用**数据自己解**（捆绑调整）。这轮我按这个思路做了，见第 2 节。

---

## 2. 这轮做了什么：用捆绑调整替代"相信声明位姿"

新增 `run_arena_bundle.py`：

- **同时解** 3D 点坐标和**每台相机的外参**（6 DOF：增量旋转向量 + 平移），Huber/soft-L1 损失以抗错误对应；
- 用 `scipy.optimize.least_squares` + **数值雅可比**。原因是手写解析雅可比连错三次（点雅可比对了并做过有限差分校验，但旋转雅可比两次用错坐标系、一次符号错）——数值雅可比**没有这一类 bug**，54 个参数规模下代价可忽略；
- 带**合成自检** `--self-test`：造一套已知相机和点，扰动 3°/0.5 mm，再恢复。

### 2.1 自检结果与它暴露的问题

```
self-test: 36 obs, rms 2.6616 px
  recovered rotation corrections (deg, injected ~3): [17.63, 20.58, 26.74, 30.65, 26.84, 18.37]
self-test: FAIL
```

**这里查出一个实质结论**：把点和相机**都自由**时，自检能压到 **rms 0.36 px**（正好是注入的 0.5 px 噪声水平），但**点坐标错了 1.03 mm**。原因是**近共面的点构型存在规范自由度**（一个刚体运动可以在点和相机之间互相吸收），投影完美而点跑了。

改成**锚定声明点位置、只解相机**（`free_points=False`）之后，规范被固定，但恢复出的相机旋转偏了 17–30°——因为**6 个近共面的基准点标定不了 6 台相机**，解是病态的。

**所以真正的问题不是求解器，是基准点构型**：5 个在同一高度（1.0 mm）+ 1 个在 2.0 mm，横向跨度只有 2.6 mm，三维张开度太差。**下一步要加的是分布在 3D 体积里的基准点**（多高度、更大跨度），不是更好的求解器。

---

## 3. 下一轮的具体做法（按 DeepFly3D 的路子）

1. **先把场地几何修好**：基准点改为分布在 ~30×30×20 mm 体积里的 12–16 个点（多高度），并在可见性上留出无遮挡角度（清理盘扩大到覆盖整个基准点区）。用自检先证明"这个构型能标定 6 台相机到 <1 px / <0.5°"。
2. **再用无标记 2D 关键点**替代大珠子：DeepFly3D 的 38 关键点定义可以直接采用（每条腿 5 个 + 腹部 6 + 触角 2）。渲染侧的做法是先按体节给**唯一材质/颜色**，再对整个 6 相机序列做 2D 关键点检测；这一步比贴珠子更接近真实实验室做法。
3. **骨架先验用 FlyGym 已有的 69 body 层级**当骨长分布，做 DeepFly3D 式的图示结构 / belief propagation 一致性修正。
4. **对极几何预筛**：6 台相机成环朝内，对极线近似水平，先丢掉落在不同图像行的对应点（DeepFly3D 的做法），再进三角化。

---

## 4. 仍未修好 / 仍未做

- **六相机绝对角度仍未端到端跑通**。卡点已从"相机模型哪错了"缩小到"**基准点三维构型病态**"，并用合成自检给出了证据（0.36 px 投影 vs 1.03 mm 点误差；锚定后相机偏 17–30°）。
- **负载质量注入仍未修好**（上一轮的结论：额外刚体在编译期被剔除；geom 级字段不存在；`body_mass` 可写但接到 BodyBackend 后读回不变）。
- 上一轮贴 0.8 mm 珠子的方向**按 DeepFly3D 的证据是错的**，应从无标记关键点重做（珠子比真实反光珠大、且会妨碍运动）。
- 束缚线**拉力**未建模；滤波器/ADC/硬件接口仍未做。

## 5. 引用

- DeepFly3D：Günel, Rhodin, Morales, Campagnolo, Ramdya, Fua, *eLife* 8:e48571 (2019)，[doi:10.7554/eLife.48571](https://doi.org/10.7554/eLife.48571)，[全文 XML](https://cdn.elifesciences.org/articles/48571/elife-48571-v2.xml)（CC BY 4.0）
- DeeperFly：[github.com/NeLy-EPFL/deeperfly](https://github.com/NeLy-EPFL/deeperfly)
- FlyGym 身体模型：[neuromechfly.org API](https://neuromechfly.org/api_reference/flygym/compose/fly/flybody/)
