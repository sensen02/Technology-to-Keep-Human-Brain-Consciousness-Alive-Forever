# S2/S5 慢神经GRU＋目标位置反馈：启动登记

新目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T112326Z`，父目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T111240Z`。父FF真实撤辅9.6秒足载≈1、非足0、COM1.242，但倒退/高滑移/无六腿协调，故本轮检验神经时间表示，不再同时改变身体。

A. 完整128维GRU隐状态慢积分tau.1秒，alpha=1-exp(-.005/.1)，只作为神经数值更新，不读取实际time或腿相位。感觉encoder与FF同样mask，输出由GRU残差和42可训练抑制性目标位置连接在actor内部产生。没有辅助角色损失或动作门控，原有限PD/42变化率/2rad/s/物理条件不变。

- rawX238与独立critic255，标签只教师动作；GRU从物理reset零hidden重放完整1920步，padding/reset语义验证，部署预热沿真实感觉历史连续，交接不重置。
- 预登记PREREG已固定：原SMOOTH_DATASETtrain0/1,val2、scratch300步lr.001/std.05同seed、无归一化或改PPO。新GRU模型身份含tau/alpha/mask/H/prior，拒旧rawGRU；FFv2兼容不变。
- 检查需覆盖stepsequence/resetpadding、alpha数学、全序列梯度、gains学习/符号、T一致扰动负反馈/无自带q项、原/新checkpoint拒载与RNG/BCAdam/PPOfresh重载、实际20ms交接无外力/重置。
- 完整71mesh、质量/重力/摩擦/有限PD/力矩上限/原生12.5微秒与1ms目标积分固定，不直接写真实根或关节轨迹。

B/C/D/E. 36slowGRU/81学生检查通过后bash117实际300BC完成exit0/reloadexact（heldoutMSE.0031937），bash123独立新进程1sstand4.8steacher9.6spureNN完成exit0；足载1/COM1.26036/nf0，但forward.0030/slip7.21296/完整周期0，未走成。命令/seed/source/data/compiled身份在IDENTITY，FFv2兼容逐位检查保留。没有后段teacher查询/hiddenreset，不用body三门代替行走。最终成功视频未执行，临时检查清理、负结果保留，goal继续。
