# 右前腿绊脚反射的身体侧学习

本次属于人工搭建大脑项目 S2/S5。按用户要求由两个子代理分别完成碰撞／教师和监督学习／独立复验，主线程核对验收与三部分架构边界。任务是从同一个冻结名义站姿起步，行走中右前腿碰到一个真实静态低障碍后局部抬腿，重新落脚、继续六足换步并撤令停稳。身体使用已验收的三球足端、71 mesh 和原 42 路有限 PD。台架时间为 0.5 秒站立、9.6 秒走动、1.2 秒停稳，原生步 12.5 μs，控制指令每 5 ms 更新。

## 碰撞与同身教师

障碍 A 的地图唯一改动为增加固定盒体，高为 0.29 原生长度，约 0.10 体长。原身体碰撞掩码、质量和惯量保持所用冻结版本。对照用同姿态和同障碍。首次实测冲击只涉及右前腿第二跗节，原生时刻约 2.38595 秒，教师在 2.390 秒首次经有限 PD 修正右前腿七路动作。其余腿在触后 250 ms 窗内的原生最低合力约 0.995 W。右前足垫峰值由对照 0.110497 增至 0.179365 原生长度，增量 0.068868。教师重新落地稳定承重时刻为 2.789 秒，后续可继续行走并停稳。原策略在该轻障碍上也能完成回合，增量实证集中在碰后抬足。逆行向冲量仅变动约 1.1%，接触持续时间接近。标称无障碍对照逐字节复现既有平地行为。

教师独立重放 904,000 原生步，接触、轨迹和 2,260 条决策逐字节一致。预检中出现过导入命名冲突、运行源码身份改变和动作限幅问题，均保留在检查表；正式入库采用修复后仅对右前腿施加纠正的版本。 [教师入库结果](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T060654Z/teacher/TEACHER_PEG_A/TEACHER_ACCEPTANCE.json) · [教师运行账目](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T060654Z/teacher/RUN_CHECKLIST.zh-CN.md)

## 监督学习

用完整同身教师 A 回合训练前馈修正。67 项输入来自当前身体状态，包含右前腿定向触力以及当前足端、关节状态、执行器目标和地面载荷。输出仅是右前腿的七路有界修正，原平地原型提供其余动作。碰障时刻、障碍位置和上一拍动作不作为学生输入。训练中行走教师共 1,920 条 5 ms 决策，61 条存在修正。240 次 Adam 更新后，有修正行的动作 MAE 由约 0.2469 降至 0.0288。训练误差用于诊断，完成与否由保存权重在同一物理身体上的独立复验决定。 [训练账目](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T060654Z/student/FIT_RF_A/RESULT.json)

## 保存权重物理复验

每个回合均从同一全态站姿连续积分 904,000 个原生步，触发前为同场景原型的精确配对。训练和留出障碍均只允许实测到的目标腿段短触事件，原生地面非足触力峰与额外力均为零。两组都通过原生推进、支撑、滑移、六腿周期和起止站稳。两组六足周期按 LF/LH/LM/RF/RH/RM 依次是 1/2/2/2/2/2。

| 试验 | 触后右前足垫峰值，冻结对照 → 学习候选 | 右前足首次持续承重，冻结对照 → 学习候选 | 候选原生最大地面滑移比 |
| --- | ---: | ---: | ---: |
| A，学习时使用的障碍 | 0.110497 → 0.162221 原生长度 | 2.789 → 2.764 s | 0.482326 |
| B，更宽更高的留出障碍 | 0.110468 → 0.179116 原生长度 | 2.789 → 2.889 s | 0.486287 |
| 无障碍 | 无触发，零修正 | 无触发，完全复现冻结轨迹 | 0.481386 |

A 在第一次 1 ms 触点记录后 4 ms 启动动作修正。B 在 5 ms 后启动。两组较冻结对照都抬足更高。B 的受碰脚重新持续承重反而晚了 100 ms，走速及归一化滑移也比同场景冻结对照稍差。两障碍的冻结原型均可行走停稳，现有结果支持触觉引发的局部抬腿，抗严重障碍、预防跌倒、其他足泛化和垂直越过盒顶均需独立验证。推进和停止的上层意图仍按真实颈部结构设计执行；该反射使用身体本地触觉和支撑观测，当前结果属于身体台架。

- [A 候选行为](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T060654Z/student/STUDENT_PEG_A/RESULT.json) · [A 原生审计](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T060654Z/student/STUDENT_PEG_A/NATIVE_AUDIT.json)
- [B 留出行为](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T060654Z/student/STUDENT_PEG_B/RESULT.json) · [B 原生审计](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T060654Z/student/STUDENT_PEG_B/NATIVE_AUDIT.json)

## 独立复验与交付

候选 A 独立加载同一个保存检查点，重新积分 904,000 原生步，决策、毫秒轨迹以及原生接触记录逐字节相同。无障碍回放全程零修正，与同回合冻结对照的物理和决策数组逐字节一致。两种障碍的受碰足重新承重开始后的连续 200 ms 内，其每个原生步载荷均不低于 0.02 W。完整 D1–D8 缺独立全套证据，本轮通过的是身体台架的碰后局部反射和站走停工程验收。

教师短程开发有三类工具或实现失败，经修正后两项完整对照、教师示范与教师独立重放均退出 0。学生拟合、平地对照、A／B／无障碍复跑、A 独立重放与套件审计均退出 0。最终渲染读取保存配置，669 帧完整解码，重点检查了碰前、实际接触、以学生侧为准的抬足峰值、承重恢复和末端静立。 [学生运行及装置记录](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T060654Z/student/REPORT.zh-CN.md)

- [学生候选验收汇总](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T060654Z/student/STUDENT_SUITE.json)
- [候选权重](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T060654Z/student/FIT_RF_A/step_0240.pt)
- [配对视频](/home/sensen/Desktop/engine修复验收/右前腿绊脚反射-20261116T060654Z/绊脚配对回放.mp4) · [关键帧](/home/sensen/Desktop/engine修复验收/右前腿绊脚反射-20261116T060654Z/绊脚关键帧.png)
