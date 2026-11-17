# 身体侧 FlyGym 对标复训检查表（S2/S5）

运行目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T045152Z`。本次只研究人工搭建大脑项目的身体侧平地起步、直走和停稳。原三球足端冻住，保留 71 mesh、42 路有限 PD、5 ms 决策、1 ms 目标更新、12.5 μs 物理积分。平地主回合无附着、无外力、无实际根状态写入。时间戳目录内分别保存 FlyGym 同身教师、训练与新鲜冻结原型对照。仿真依赖沿用归档 Python 与 MuJoCo 3.9.0，多进程限制并将每进程 BLAS、OMP、MKL 设为单线程。

## 启动前

- 原有通过原型冻结，来自 `/run/media/sensen/Data2/cell_wound_prototype/experiments/body-motor-learning/20261116T000536Z/`。原动作指令和观察结构与输出 42 路保持。教师和学生的身份、物理输入、预算与成功门写在各自实验目录。
- 同身教师先通过闭环，再入库做有界监督残差。首轮模型零输出时须重放原型动作一致。教师行为需有可辨识、记录完整的 FlyGym 来源影响。不触碰现有视觉／下行权重。
- 正式配对使用标称 0.5 s 空指令、9.6 s 前进、1.2 s 停稳及相同冻结初态。以原有原生五项、六足换步及站稳综合门为准。留出体态、停步时机和轻扰另列。走动侧推沿用 lean_right、走动回合第 3.700 s、0.2 W×20 ms 的历史口径。保留 D1–D8 之外的工程通过范围及全部失败。
- 无 PPO／SAC。监督梯度和优化器以实际一次更新、零修正、限幅与存储复验为关口。教师与新权重物理闭环通过条件保留独立。

## 执行中

- 完成与失败各阶段追加实际命令、环境、日志、散列和退出码。

## 结束与验证

- 同初态同观测判定动作、身体安全、稳定站立及完整轨迹。轻扰与侧推保留每腿绝对切向速度及归一化滑移，区分推进分母下降的影响。
- 独立保存重放与同次视频验收，实际打开关键帧。通过与失败以冻结口径报告。

## 运行账目

同身环境使用 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python`，附 `MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`。教师侧的 FlyGym 直接仿真使用独立 `venv_body`。数据和软件均以各回合身份文件的 SHA-256 为准。

- 冻结对照由原 `prototype.py --walk-ms 9600 --start nominal --native`、`--start lean_right --light`、`--start moving_backward --walk-ms 10250` 和本轮 `audit_baseline_push.py` 生成，退出 0。用原 `native_score.py` 复算标称窗退出 0。四组相关轨迹和动作与档案逐字节相同，见 `BASELINE_VERIFICATION.json`。
- 教师源采集、相位缩放、原生复验中的失败与恢复、样本批准见本轮 `teacher/RUN_CHECKLIST.zh-CN.md` 与 `teacher/ADMISSION_INDEX.json`，包括原始退出码。独立 `compare_transfer.py` 读取完整来源文件后对三组增益退出 0，写出 `TRANSFER_AUDIT_G05.json`、`TRANSFER_AUDIT_G15.json`、`TRANSFER_AUDIT_G30.json`。试读未完成来源的开发结果曾因缺字段退出 1，后以独占来源记录复验。
- 训练与身体回放命令、优化器记录和装置检查由本轮 `student` 目录保存，所有权重更新绑定 `FIT_NOMINAL_G30/JOURNAL.jsonl`。从该权重加载回放，以同条件核对身体闭环及保留失败。
- `independent_nominal.py` 以同一冻结权重独立重放标称回合，退出 0。轻扰与真实侧推的首轮工具退出码 143，完整写出的轨迹另做两项 `physical.py --native` 独立重放。两项求解器和 shell 均退出 0，轨迹、动作、原生数组逐字节一致，见 `student/PULSE_REPEAT_COMPARISON.json`。
- `compare_body.py` 输出具名腿物理步态量，`plot_transfer.py` 从冻结、教师及候选的真实轨迹画去趋势形状变化，两者退出 0。图已实际打开。最初调用归档渲染器处理侧推因历史结果缺 `walk_ms` 退出 1，后使用本轮 `render_study.py` 补时段格式。标称教师、标称复训与侧推配对的三个视频全帧解码，各 566 帧，关键帧均实际打开。
- `inspect_recovery.py` 核查相同候选的配对回放并保存观测，退出 0。该观察包留出，用途仅为定位身体反射的采样时段。
