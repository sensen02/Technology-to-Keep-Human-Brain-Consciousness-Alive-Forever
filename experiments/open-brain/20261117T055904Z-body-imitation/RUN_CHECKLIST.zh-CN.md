# 身体首课实际运行（P1、S2–S5）

- 新目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain/20261117T055904Z-body-imitation/`。控制和法定工程阈值见 `PREREG.zh-CN.md`，回合种子见 `cases.py`，身体和快照见 `BODY.json`，训练数据/权重哈希见各模型 `INPUTS.json`、`JOURNAL.jsonl` 和 `SELECTION.txt`。身体无足部附着，.12 W×20 ms 推力只施加在声明的活动窗。
- 使用归档 Python 3.12.14、Torch 2.5.1 CPU、MuJoCo 3.9.0，OMP、BLAS、MKL 单线程、EGL。`drive.sh body_run.py freeze` 首次生成的快照和二进制可用，NumPy int32 使元数据写盘退出 1；修补后冻结退出 0，前后快照及编译模型两份 cmp 精确一致。初试把尚未启用控制范围的执行器的 [0,0] 误作硬门。原失败保留在 `TEACHER_INITIAL_BAD_LIMIT`；依据 `actuator_ctrllimited` 纠正后教师验证通过。
- 教师 `body_run.py teacher` 17 回合整组退出 0，先完成教师门再训练。`fit.py BC1`、三条开发和动作/脉冲审核退出 0，三个闭环行为门的失败按原门记录。每轮保存可读权重、Adam、随机状态及逐轮日志。损失为监督学习，PPO/GAE 检查不适用。`batch.sh correct DAG1` 六条同态纠正退出 0；`fit.py --continue-from ... --extra DAG1 --envelope` 两个分支与六个开发回合均退出 0。
- `batch.sh evaluate HELD`、`batch.sh teacher HELD_TEACHER`、`batch.sh evaluate REPLAY` 及 `audit.py --replay` 全部退出 0。输入、到达及身体复算详见 `HELD/INDEPENDENT.json`。最初时序复算假设主动停车后驱动持续零，未考虑两束衰减不同，检查退出 1。用实际脑脉冲、颈后到达与驱动时刻复算，确认 17 ms 回弹，`timing.py` 退出 0。五条正式学生、五条配对教师及五条学生复跑全部走完整连续回合。
- `render_evidence.py` 直接消费正式主动停步的同次 1 ms 保存姿态。85 帧视频全解码通过，30 fps、0.60 倍，已实际打开起步、走动和末窗姿态。没有追加物理求解或姿态矫饰。下一步用此冻结身体验收脑侧触碰/负载输入是否能选择起步和停车。脑侧不在本轮训练范围。
