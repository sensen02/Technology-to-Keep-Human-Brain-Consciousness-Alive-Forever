# 声明式参考零点分支：阴性结果

**手工研究原型，不是成品。** 本分支只改**两处显式声明的假设**，不调任何增益、不改连接、不改递质标签、不加CPG/PD或隐藏驱动；原规范与既有分支保持冻结。

## 一、改动内容（全部预先声明）

- **A1 本体感觉零点**：由"episode自身的起始姿态"改为**模型XML中每个有限关节范围的区间中点**（无限位关节保留初值）。理由：之前的零点等于姿态目标，而本项目文献研究要求零点必须是可追溯坐标。实测有**37个关节**被移动，根自由关节坐标未被改动。
- **A2 接触力归一化**：由声明值`1.0`改为**模型自身的单腿体重份额** `total_mass·|g|/6 = 4.074392`（实测得到的模型量）。
- **未改动**：`current_nA_per_unit`、`reference_angle_rad`、`reference_velocity_rad_s`、网络参数、连接、递质标签、运动映射。

## 二、对照与不变量

| 检查 | 结果 |
|---|---|
| 声明分支的normal与命令回放8类数组精确一致 | 通过 |
| 声明分支输出静默时`act`/`ctrl`严格为0 | 通过 |
| 全部状态有限、无警告 | 通过 |
| 源码哈希前后一致 | 通过 |

## 三、结果：没有改善，反而略差

| 指标 | 已发布25μs基线 | 声明零点分支 |
|---|---:|---:|
| 10%胸高逃逸时间 | 23.15ms | **21.63ms** |
| 质心下降 | 0.961mm | **1.189mm** |
| 200ms质心高 | 0.907mm | **0.679mm** |
| 200ms根高 | 0.832mm | 0.950mm |
| 最大激活 | 1.00 | 1.00 |

两组都倒下；声明分支的质心下降更多、逃逸更早，但根部更晚触地——即姿态折叠方式不同，**没有任何一组达到站立**。

## 四、这说明了什么

1. **"参考零点与归一化缺少测量依据"虽然被证实是缺陷，但修好它并不足以维持站立。** 阴性结果本身有价值：它排除了一条候选解释。
2. 与力矩审计一致：**瓶颈是命令的空间模式与时序**（t=0胫节零支撑、24ms胫节过量而同侧转子不足、崩塌时最缺髋部侧向控制、最大实际力矩>最大所需），而不是参考定义或输入量级。
3. 因此下一步不应继续在"前端标定"上寻找站立，而应研究**输出侧**：运动细胞池的映射与募集（74/90可用、共享池均值译码、所有细胞同一套参数），否则就是在没有依据的情况下反复调整输入。

## 五、产物与边界

- 脚本：`/run/media/sensen/Data2/cell_wound_prototype/run_declared_reference_branch.py`
- 测试：`/run/media/sensen/Data2/cell_wound_prototype/test_declared_reference_branch.py`（5项通过，含"必须如实报告阴性"的断言）
- 结果：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/declared_reference/declared_reference_results.json`
- 轨迹：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/declared_reference/declared_reference_traces.npz`
- 图：`/run/media/sensen/Data2/cell_wound_prototype/outputs/model_branches/declared_reference/declared_reference.png`

边界：4个臂、单一初态、单一身体步长、无参数搜索；A1/A2是声明假设而非文献测量；不据结果反向调参；站立判据与之前一致，未改阈值。
