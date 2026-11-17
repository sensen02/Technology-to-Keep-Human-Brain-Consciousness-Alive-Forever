# 踏空收腿与辅助腿补撑学生侧实现

对应人工搭建大脑项目 S2/S5。新实现位于 `/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T063132Z/student`。使用同一名义站姿与 71 mesh 原结构，保持 42 路有限 PD、自由根、无附着、无外力。完整试验为 0.5 s 静立、9.6 s 行走与 1.2 s 撤令站稳，控制决策每 5 ms，原生积分步 12.5 μs。

## 已完成的学生模块

前馈策略接收当前关节位姿、速度和执行器目标、各足承重、右前足本地速度与下沉量，并利用其他腿当前目标匹配正常脚的位置与预期负载。动作由当前介入判定和有界关节残差组成，动作通道从合格教师实际运动标签确定。没有时钟、缺口编号、旧动作、跨步隐状态。缺口存在这个环境字段只用于物理建模和验收配对。模型加载路径、原生同场景复跑、纯收右前足消融、逐腿原生承重归因和保持旧绊脚反射的组合入口均已实现。软件烟测的 323 条既有平地决策正序和反序精确匹配，验证的是关闭修正路径。

合格数据门固定同一身体、相同初始全态、完整时间、原生施力为零及各文件散列，要求教师通过行走和站稳，并绑定物理资格。实际失败的完整 AN 教师轨迹送入训练入口后因原生滑移门未过退出 1，退出发生于训练输出目录创建及参数更新之前。该退出为刻意验收的数据拒绝。 [拒绝日志](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T063132Z/student/REJECTED_UNSAFE_TEACHER.log)

无缺口的完整平地对照退出 0。904,000 次原生积分的最大滑移比 0.4813864263，通过起止站立、前进力学及六足周期。与上轮冻结对照相比，共享状态、力和动作数组逐字节一致，新增读出的局部触觉与本体字段有独立口径。 [对照匹配](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T063132Z/student/STOCK_AUDIT.json)

## 未入库的 AN 诊断

真实右前足在 2.764 s 第一次落入缺口。冻结原型在 7.596 s 才返回上表面持续承重。辅助动作版本提前到 3.283 s，纯收右前足版本在 3.829 s 持续承重。完整原生滑移比分别为冻结 0.867、辅助 1.164、纯收腿 1.082，均超平台的原阈值 0.50。

辅助臂与只有收腿的实验在首个右中足动作前原生力完全相等。右中足此前 20 ms 已平均承载 0.241787 W。辅助臂后 200 ms 的右中足平均力 0.242980 W，纯收腿 0.242634 W，差 0.000347 W。后 500 ms 辅助臂反而少约 0.000841 W。与冻结对照相比可见的右中足力增加，大部分无需额外右中足动作即可出现。此对照为未入库的实证诊断，说明了已有承重腿的再分配与收腿带来的间接变化，尚无卸载脚建立新接触的证据。 [同体消融力](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T063132Z/student/AN_TEACHER_ABLATION_AUDIT.json)

AN 控制与辅助教师的同期比较视频从已有配置绘制，解码 709 帧，生成过程没有积分。关键帧实际打开，确认显示几何缺口、趾足落入、重新上表面以及末段状态。播放器和标题明确标示滑移失败。 [发展诊断录像](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T063132Z/student/VIDEO_DEV_SERVO_AN/踏空配对回放.mp4)

## 上轮绊脚回归

采用原来的右前腿绊脚检查点和 A 障碍，附加被动的新缺口观察量后重新积分。该回放加载原绊脚权重，缺口观察只用于记录。904,000 个原生步及 11.3 s 新回合退出 0。完整物理轨迹、原生力、旧观测和所有动作与上轮通过的绊脚回放完全相同，原生门保持通过。新的缺口修正与绊脚策略的动作组合资格仍需实际缺口策略权重。 [实测回归审计](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T063132Z/student/SENSE_ONLY_STUMBLE_REGRESSION/COMPOSITION_AUDIT.json)

## 定格足位后的停稳诊断与本轮边界

浅缺口 AS 采用两个实际 6.450 s 教师诊断回合，拟合和继续步行的正式资格不适用该短试验。固定世界坐标足位的方案和纯收腿制动的方案均可停稳。以 1 ms 取样计，右前足恢复上表面持续承重分别在 3.128 s 与 3.682 s；原生步测得的尾数略小。固定足位缩短恢复时间，未证实右中足额外负载或一条未承重脚落地补撑。 [停稳诊断录像](/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T063132Z/student/VIDEO_BRAKE_BEST/踏空配对回放.mp4)

本轮无通过预登记完整物理门的教学轨迹，未训练新的踏空学生权重，也无学习策略缺口闭环。下一最小动作是从已实测停稳的支撑全态接续重新迈步，在同一地形上使用上表面固定落脚点与原生负载／切向滑移共同筛选下一步，再把停、换足、徐缓接回原步态做成一个连续合格教师。需要补撑证据时另选仍有一条合适的腿尚未承重的实测步态相位，确立可归因的新接触并做单腿消融，在此之前保留已有支撑腿再分配的狭义表述。

运行环境为 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python`、MuJoCo 3.9、EGL 和单线程 BLAS。实际调用 `teacher/physical.py control student/STOCK_CONTROL --native`、`student/compose_stumble.py student/SENSE_ONLY_STUMBLE_REGRESSION --sense-only --case ../20261116T060654Z/teacher/PEG_A.json`、配对 `student/render_gap.py`、解码核对退出 0。首次 stock-floor 命令漏写 control 模式退出 2，正规命令经改正后完成。训练拒绝负例用 `student/fit_gap.py student/REJECTED_UNSAFE_TEACHER --teacher teacher/TEACH_SERVO_AN --flat student/STOCK_CONTROL teacher/CONTROL_AN_FILLED`，退出 1 且没有学习产物。
