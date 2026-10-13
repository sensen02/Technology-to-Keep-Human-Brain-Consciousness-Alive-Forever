# 文献约束的 DNa01 神经—运动桥接原型

> **用户纠正后的验收状态：不符合果蝇自身自主闭环目标（NONCOMPLIANT_WITH_AUTONOMOUS_GOAL）。** 本演示含外部行走许可、目标朝向误差注流、独立步态控制器；程序参与决策和操作，而非只翻译。下面的工程PASS与视频仅为历史接口对照，不能作为用户目标完成证据。强制约束见 `/run/media/sensen/Data2/cell_wound_prototype/AUTONOMOUS_LOOP_REQUIREMENTS.zh-CN.md`。

这是手工实现、测试中的仿真原型，不是成品，不是真实果蝇电极记录，不是完整神经肌肉重建。现有肌肉规范模型保持不变；本路线是独立入口。

## 实现

`虚拟朝向误差 → 工程化电流注入 → BANC 子网络电导脉冲模拟 → 已注释 DNa01 左右脉冲 → 因果滤波/差分解码 → FlyGym HybridTurningController → 身体 → 朝向反馈`

- 神经脉冲由实际运行的 ConductanceNetwork 产生，不是按运动轨迹伪造的脉冲。
- 解码规则、滤波时间常数50 ms、100 Hz归一化、增益/符号/限幅都是声明的工程设计；不是逐突触生物重建。
- 前进许可0.8由外部明确提供。此演示不证明 DNa01 决定前进、不证明连接体自发产生步态。
- 低层步态来自已安装 FlyGym 2.1.0 的 HybridTurningController、记录步态轨迹、位置致动与附着；停止采用中立腿姿态和附着，不靠根关节锁/瞬移。
- 感觉侧直接使用仿真朝向误差（误差/.3 rad限幅后乘.05 nA注入对应DNa01），不是已实现复眼视觉链或生物感受器回路。
- 使用12000神经元BANC子网络；本次突触缩放1e-6 uS/synapse、背景电流标准差0，是独立演示配置，不冒充原网络默认参数，也不能和旧默认网络数字混用。

来源：
- FlyGym官方转向教程：https://neuromechfly.org/tutorials/4d_turning_controller/
- 神经转向研究：https://pmc.ncbi.nlm.nih.gov/articles/PMC12279373/ （DOI 10.7554/eLife.102230）。支持DNa01/DNa02与转向、双侧活动差相关；不能据此标定任意速度/转向增益。

## 身份与范围

本地BANC CSV主类型及Soma side注释：
- 左DNa01：720575941535862506
- 右DNa01：720575941432123640

运行时由root ID动态查找局部索引，缺失即报错；不把FAFB标本ID接入BANC。结果绑定代码、身份报告、BANC数据源哈希。

默认12000子网络缺少左DNa02，因此不接单边DNa02凑双边。16000上限可同时保留两对，尚未在本演示使用。oDN1在本地出现为DNg97的别名，但功能解释未独立确认，不编造该通路。

身份报告：`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_motor_bridge/identity_audit.json`

## 验证与干预含义

解码器五项单元测试通过：左右互换、静默、断路清除记忆、时间分块一致性、显式前进许可、剂量比较及错误输入拒绝（部分合在同一测试）。另用固定合成输入作匹配解码因果对照；这部分不是BANC记录。

身体基线：五组2秒平地测试（站立、前进、左转、右转、先行走再停止）通过声明的工程判据。前进位移约27.2 mm，但有约11.5°航向漂移，因此闭环目标不能靠基础步态自然实现。停止最后0.5秒漂移约0.00221 mm。无 equality，根自由。

闭环试验：每组身体先统一站立0.2秒，随后神经与身体按5 ms区间交换信号、模拟2秒，目标为相对初始朝向左转45°。神经步长0.5 ms，身体步长0.1 ms。time_s从身体预备阶段后计时。

初次完整运行结果（最终复验以closed_loop.json为准）：
| 分支 | 身体转向变化 | 最终目标误差 | 含义 |
|---|---:|---:|---|
| intact | +42.11° | 2.89° | 完整工程桥接 |
| sham | +42.11° | 2.89° | 完全相同的无操作对照 |
| swap | -85.97° | 130.97° | 交换左右读出，身体反向转 |
| silence | +6.83° | 38.17° | 仅屏蔽DNa01读出脉冲，神经本身仍在放电；外部行走许可仍在 |
| cut | 接近0° | 45.00° | 切断整个工程桥（包括外部行走许可），不是神经解剖切断 |
| feedback_cut | +6.83° | 38.17° | 切断虚拟朝向→注入电流；身体自身接触/本体反射仍在 |

完整分支58个左DNa01脉冲、右侧0个，身体位移20.50 mm，最大倾角6.85°。这不是说右DNa01失效，而是当前目标要求左转。

各干预是独立闭环分支：身体分化后，后续反馈电流也随之改变，不能声称全部干预收到完全相同的脉冲回放。只有sham是逐值匹配的无操作对照；合成输入解码测试另有严格匹配。

工程验收增加：完整分支终点朝向误差≤10°、所有分支采样倾角≤20°、高度≥0.5mm、有限状态，以及交换符号/静默/断路/sham检查。阈值是工程声明，非生物学标准。姿态按5ms采样，不是连续时间全程证明。

## 文件与运行

主要代码：
- `/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/neural_motor_bridge.py`
- `/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/bridge_body.py`
- `/run/media/sensen/Data2/cell_wound_prototype/run_neural_motor_bridge.py`
- `/run/media/sensen/Data2/cell_wound_prototype/tools_test_neural_motor_bridge.py`

在项目目录运行：
```bash
OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python /run/media/sensen/Data2/cell_wound_prototype/run_neural_motor_bridge.py --seconds 2 --render
```

数据/图/动画：`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_motor_bridge/`
桌面交付：`/home/sensen/Desktop/cell_wound_prototype/neural_motor_bridge/`

## 不声称

没有训练得到真实神经解码；没有真实电极/行为配对验证；没有连接体自发步态或生物停止通路；没有解决原肌肉模型站立；没有粗糙地面或任意长时鲁棒性证明。接触力受 `/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/force_channel.py` 的生理量声明限制。

下一步可扩大到正负目标、多种子和较长轨迹，加入匹配的已记录神经脉冲回放，再考虑DNa02动态差异；只有取得配对神经/运动数据后才能拟合、验证生物定量映射。
