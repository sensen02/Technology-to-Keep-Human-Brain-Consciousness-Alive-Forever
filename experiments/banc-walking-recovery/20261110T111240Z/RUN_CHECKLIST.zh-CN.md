# S2/S5 目标位置神经抑制反馈：运行登记

新目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T111240Z`，父目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T105518Z`。假设：target−q误差抑制分支自带+q项，单独KT收缩不能保证全身稳定；本轮只将同一可训练分支改接target位置126:168，身份v2区别旧模型。

A. 源码、身份、预算、门槛见PREREG.json：原FF2x128/42可训练增益，物理MLP同样mask，criticraw/原logstd，scratchBC1000步lr.001/std.05/同seed，同SMOOTH_DATASET train0/1 val2。完整身体、无附着、质量/重力/摩擦/有限PD/2rad/s/力矩/12.5us物理/1ms目标积分不变；部署只有ANN输出，未加外控制器/相位/脚本根运动。

- 预检查应覆盖coherentT扰动两块/KT负对角，固定T实际q变化只有MLP项、零MLP则无q项，prevJac0、全42增益可训练/更新、critic原输入、旧v1拒载/新权重buffer/RNG/Adam重载、20ms交接不reset。
- 原始验收与工程门分别保留；未满足实际自主门不做最终成功视频。输入数据来源官方测量拟合形状，工程wave重定时已声明。

B/C. 46结构/77学生检查通过后bash110 scratch1000BC完成exit0/reloadexact；bash111新进程连续1sstand4.8steacher9.6spureNN完成exit0，COM1.24224/非足0/足载.99994三body门通过，但forward−.02879/slip13.7855、LF2/RM18不协调，未走成。独立全部输入身份verify通过、min足载.985。完整同次视频770帧解码/父代理已打开，明确NOTWALKING。相同前缀零输出129完整保持body但不移动，首次前台超时在INTERRUPTED单列、非完成结果不使用。身份旧新hash/命令source在各IDENTITY/CHANGE_LEDGER。

D/E. 失败轨迹、全身图像保留并实际打开。临时检查自动清理。目标仍原始完整自主行走，不转平台，不用计划或局部导数代替结果。
