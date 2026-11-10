# S2/S5 相同神经结构、潜变量模仿损失对照

新目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T112848Z`，父 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T111240Z`。父目标位置反馈FF同预算训练MSE.0010、真实9.6s稳站但不走，当前检验输出tanh饱和导致的模仿学习条件问题，不同时改网络/物理。

A. PREREG固定：FFv2 raw238同mask/可训练抑制目标位置支路/H/critic，原数据train0/1 val2，scratch1000lr.001/std.05seed20261110；只将训练目标改为atanh(clip(Y,±.999)) latent均值MSE。实际部署仍tanh后的ANN42变化率，不改执行器，记录对原始Y的动作误差和clip失真而非只看latent损失。

- 标签变换是训练分布权重变化，不声称与原boundedMSE等价。Gaussian固定std下latentNLL差常数/尺度与雅可比项的梯度契约须核对。
- 检查atanh端点有限、梯度方向/FD/神经gain、饱和时latent梯度与action梯度差、critic/std不变、验证独立、检查点BCAdam/RNG/PPOfresh及现有loader完整身份。无长物理训练混入检查。
- 同身体/附着关/有限PD/力矩/2rad/s/12.5us物理/1ms积分/5ms动作，未改真实姿态或加力。actor不见相位/腿序/计时，监督仅来源教师动作。

B/C. 26latent检查和77student检查通过，bash119固定1000潜变量BC完成exit0/reloadexact，heldout原始actionMSE.001720；精确±1标签clip失真MSE7.44e-11单列。bash120新进程1sstand4.8steacher9.6spureNN完成exit0，body三门通过COM1.23857/nf0/足载1，但forward−.07510/slip10.4656/LF48其余0，失败。所有source/原始输入与编译参数身份保存，后段无teacher/hiddenreset，损失变化不等于实际行走。

D/E. 若没有全部实际行走门，最终成功视频不执行；失败完整保留并打开实际图。临时检查自动清理。完整goal仍未完成，持续同目标。
