# S2/S5 本轮运行记录：训练集冻结输入标准化

A. 父实验 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T094051Z`。本目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T094554Z`。假设是原始通道尺度影响学习及闭环稳定性；唯一模型改变为actor输入训练均值/标准差变换（std下限.05、无裁剪），critic保持原输入，不混入上一动作屏蔽。预登记PREREG.json先于运行。来源、已声明身体和原验收同父实验，不改XML/摩擦/增益/力矩/速率。

- 检查证据：NORMALIZATION_CHECKS.json 43项、STUDENT_CHECKS.json 78项通过，包括训练/验证隔离、统计冻结、FF/GRU一致、原始checkpoint拒绝、buffer/RNG/Adam重载、20ms连续交接；父教师已实际验证五项门且六腿完整事件。
- PPO实际梯度/GAE/KL不在本次改动范围，沿用已检查实现；这轮只BC，不把BC Adam状态带入PPO。actor均值和.05探索std声明保留。
- 训练数据 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T094051Z/SMOOTH_DATASET`，train0/1共3840帧、validation2。落脚停止尾窗不训练。
- 命令入口为本目录env/bin/python -B、本目录student_wave.py；OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1。FF1000步、GRU300步、lr.001，种子沿用CLI默认（保存在每个IDENTITY.config.cli）。身份记录覆盖源/数据/初态/编译物理参数。

B. root任务：bash-83 FF训练已退出0并收集、保存权重及全BC Adam/RNG/独立PPO空状态，final保存后重载一致；bash-84 GRU训练已退出0并收集；bash-85 FF、bash-87 GRU独立新进程真实接管评估均退出0并收集，两者都失败。无输入身份修改。后续进度/退出状态以同目录RESULT.json和会话收集记录为准，不把此段当后台调度器。

C. FF验证MSE2.76888e-5；首个接管动作在示范状态RMSE.001383，但不据此宣布行为成功。独立评估协议：1s实际站稳＋4.8s教师步行／GRU感觉预热＋9.6s纯student，连续保持物理/执行器/previousaction/hidden；神经阶段不调用教师。评估实际状态/力/滑移/支撑/推进在EVAL_*_NORM/RESULT.json逐项记录；未出结果部分仍为未评估。

D. 最终完整视频未执行；本次仍为诊断候选，只有神经实际门槛通过才进入原生完整验收与最终同次视频。先前教师视频不是此模型自主视频。

E. 临时检查目录自动清理；失败对照与必要checkpoint保留。当前目标未完成，不转其他平台。待全部评估后汇总结果并更新主状态、tracker及桌面。
