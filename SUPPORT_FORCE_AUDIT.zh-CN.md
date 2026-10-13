# 同姿态受力审计：支撑、质心与约束

**手工实施并测试的研究原型，不是成品或自主站立验收。**

## 问题与对照

上轮最佳74肌与最佳90肌来自不同姿态，不能直接用高度差归因16肌贡献。本轮固定各候选姿态，在每个姿态上分别计算原生下限、该姿态保存的74肌命令、该姿态保存的90肌命令。

受力证据优先于混合广义范数：

1. 将每个地面接触的局部接触力变换到世界坐标，并选择作用在果蝇上的方向。用独立球—平面测试核对正负号。
2. 全部身体质量加权的质心雅可比乘广义加速度，得到零速度初态的全身质心加速度。**根部加速度不等于质心加速度。**
3. 核对整体牛顿方程 `总质量 × 质心加速度 = 重力 + 地面力 + 其他外力`。仅在额外外力确实为零时才可简化。
4. 把约束雅可比转置乘约束力按类型分解，核对合计等于仿真的约束广义力。根部力与角自由度力矩分别报告，不相加成所谓力比例。
5. 接触条目数不等于脚数；分别记录接触与接触几何体身份。

## 保持不变的条件

规范 XML、质量、肌肉、刚度、接触设置不修改；静态快照 qvel=0、act=ctrl，未选择肌肉保持原生下限；每次计算清除历史状态。搜索命令只是隔离力学对照，不接入神经闭环。

## 解释边界

- 关节限位与肌肉是内部作用，通常不能直接提供整个身体的外部合力；但可以改变接触解。快照分解不是关闭限位的因果实验。
- 接触承担足够竖直重量，也不自动证明转动/关节平衡或动态稳定。
- 支撑不足只描述所测姿态和命令，不证明任何姿态都不可支撑。
- 比例只用同量纲的世界竖直地面力/体重；质量SI标度未建立，力保留模型力单位。

## 实测

### 保存记录的同姿态交叉检查

在 round37_explicit_limit_repair 姿态，保存的74/90命令根竖直加速度分别为 -1257.713765 / -1280.210994 mm/s²；在 small_offset_01 姿态，分别为 -1077.956272 / -1123.711074 mm/s²。90肌的全身归一化目标更低，但这两组的根部竖直加速度更负。这不是矛盾：全身目标还包括关节角加速度，优化可能发生权衡。不能把优化目标改善直接解释为竖直支撑改善。

### 六个同姿态快照的受力实测

| 固定姿态 | 命令 | 竖直地面力/体重 | 全身质心竖直加速度 mm/s² |
|---|---|---:|---:|
| round37_explicit_limit_repair | 原生下限 | 0.65644452 | -3367.18725 |
| 同上 | covered74 | 0.83580935 | -1609.23256 |
| 同上 | all90 | 0.84835831 | -1486.24025 |
| small_offset_01 | 原生下限 | 0.66550493 | -3278.38617 |
| 同上 | covered74 | 0.83182404 | -1648.29254 |
| 同上 | all90 | 0.84835207 | -1486.30133 |

由此可见：在这两个姿态的所测命令下，地面支撑仍不足以抵消重量；90肌命令相对同姿态74肌命令有更大的竖直支撑、质心向下加速度较小。**这与根部竖直加速度的趋势不同，进一步说明根部不是全身质心。**

六个快照的整体力平衡最大误差不超过1.60e-14模型力单位，远低于声明容差。编译后空气密度与黏性均为零，施加外力数组清零；其余外部合力项在产物中逐项保留。

所有六个快照有14个地面接触条目，涉及6个不同的具名跗节几何体；只有56行棱锥摩擦接触约束，**没有活动关节限位约束行**。这是这些限位修正后姿态的初态结果，不能外推到原始超限姿态或后续积分；更不能说明限位参数永远没有影响。接触条目、接触几何体及力向量详见产物，56约束行不是56只脚。

**结论仅限快照：** 当前所测命令下存在外部竖直支撑缺口，计算通过独立的质心牛顿方程核对。它不证明最优可达支撑力是多少，不证明肌肉参数不足，也不是神经闭环的因果归因。

## 复跑与交付

- 审计：`/run/media/sensen/Data2/cell_wound_prototype/run_support_force_audit.py`
- 测试：`/run/media/sensen/Data2/cell_wound_prototype/test_support_force_audit.py`
- 数据：`/run/media/sensen/Data2/cell_wound_prototype/outputs/workspace_recheck/support_force_audit.json`
- 图：`/run/media/sensen/Data2/cell_wound_prototype/outputs/workspace_recheck/support_force_audit.png`
- 绘图代码：`/run/media/sensen/Data2/cell_wound_prototype/make_support_force_figure.py`

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python /run/media/sensen/Data2/cell_wound_prototype/run_support_force_audit.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python /run/media/sensen/Data2/cell_wound_prototype/test_support_force_audit.py
```

桌面报告：`/home/sensen/Desktop/cell_wound_prototype/SUPPORT_FORCE_AUDIT.zh-CN.md`。
桌面数据图：`/home/sensen/Desktop/cell_wound_prototype/workspace_recheck/support_force_audit.json`、`/home/sensen/Desktop/cell_wound_prototype/workspace_recheck/support_force_audit.png`。

下一步若继续，应测同一姿态下竖直支撑与关节/角加速度之间的权衡，或在保留全身约束的搜索中将质心支撑显式纳入目标，而不是只追求竖直力而忽视其他自由度。任何改目标得到的结果仍需独立动态验证。
