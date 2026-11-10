# S2/S5 短学生段＋教师恢复：启动登记

本目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T103413Z`，父版本 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T102045Z`。仅在50%混合模型仍不合格时启动本条件。

A. 假设：连续混合会一直把状态限制在教师附近；短50ms纯学生段＋200ms纯教师恢复可以测到模型自身误差与真实恢复。预登记PREREG已写；新代码归属burst_recovery/check_burst检查在准备，不用计划冒充执行。

- 身体/物理/神经模型输入同父版本，未改限速/力矩/摩擦/PD/重力；完整71mesh，无附着/外力，不写实际qpos/root。
- actor输入实际感觉和上一实际动作，GRU隐藏沿真实执行历史连续；教师时钟仅用于监督来源，不给actor。教师每5ms在当前实际状态查询，纯学生段标签仍不执行；恢复段不重置身体。
- CLI计划3独立seed20261301..3、最后一个验证，1s站稳/20ms小电机脉冲/4.8s教师前缀/4.8s短段轮换；.01噪声只加学生段。COM<1.2/非足>0逐1ms早停保留失败前缀，阈值只是采集安全界，不改变验收。
- 完整记录stage0/1/2、实际动作与标签/beta/time/appliedticks、1ms轨迹与每次恢复前后物理量；依照实际阶段选择trace记录，恢复是否有效由数据而非预设判断。
- 正确性检查：执行公式、prevactual、beta阶段、GRU全历史梯度、独立holdout、输入身份与篡改拒绝、MODELONLYwarmstart/fresh BCAdam/RNG/PPO/checkpoint重载。不在检查里跑长实际步态。

B/C. bash97真实采集三条完整1920决策退出0、无早停，stageprefix960/student200/teacherrecovery760，末尾不完整恢复剔除统计。BURST_ANALYSIS核验hash/时序/动作，学生段RMS覆盖.077–.087/p95.15–.205；高度误差恢复均值微小正值、角度无稳定改善，post滑移.50–.54，不是合格步行或完美恢复oracle。bash98训练1000lr.0003seed20261311完成exit0/reloadexact；bash102独立新进程1s站稳4.8s教师前缀然后9.6s纯神经，无后续教师查询，连续状态保持。实际COM最低.56005/非足43.718/滑移7.60985，失败。所有任务已收集、必要失败保留。

D/E. 最终成功视频尚未执行；本阶段只作受控数据改进。临时检查fixture自动清理，必要失败证据保留。原始完整验收未通过，goal继续而不转平台。
