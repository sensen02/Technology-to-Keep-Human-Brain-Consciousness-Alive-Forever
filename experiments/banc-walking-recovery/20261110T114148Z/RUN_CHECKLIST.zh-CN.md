# S2/S5 训练专用教师角色辅助监督：登记

新目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T114148Z`，部署模型与父 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T112326Z` 一致。父纯BC慢GRU已实际保持9.6s足部承重（COM1.26036/nf0/足载≈1），但推进.003且完整换步0；本实验仅检验隐藏时序表示的训练监督。

A. 只在训练侧加Linear128→6角色分类器，损失BC动作MSE＋.001CE。类别由已核验教师计划阶段与连续训练索引派生，teacher role不是实测接触真值。标签不进入actor/critic输入、hidden初始化/重置、动作门控。分类器不保存在部署actor state，不在评估计算，所有42命令仍由原神经actor输出。独立验证标签只报告，不参与优化。

- 同SlowGRUv1/tau.1/5ms/抑制性目标位置连接/原raw输入/身体执行器/300步lr.001std.05seed20261110，创建辅助头隔离RNG不改变原actor初权重。roleweight0必须与无辅助两步actor更新逐位相同。
- 检查完整序列hidden/mean一致、actor不受辅助头forward影响、角色边界/源order、CE梯度达到GRU、验证隔离、source/checkpoint/RNG/Adam/PPO鲜状态和原loader兼容，Core编译但无长真实仿真混入检查。
- 数据原SMOOTHtrain0/1 val2，无期末停止尾窗，所有物理条件原样。辅助标签只监督神经内部表示，不是部署时算法CPG或腿选择器。

B/C. 43role辅助检查和81学生检查通过后bash128实际300训练exit0/reloadexact，heldoutMSE.003263、角色准确率.9885；bash131独立9.6spureANN首次全6腿严格周期LF1其余2，forward.062618/COM1.25352/nf0/足载1，滑移1.35146仍失败。随后bash132独立同协议native1232000步602块、所有trace/decisions逻辑数组逐位相同、source0/2/3/外力0，原生forward.0626197/slip1.35273。原始D1/2/3/4/4b/5/6/8数值PASS、D7暂UNKNOWN和滤波PROPOSED不重写历史。native检查143exit0，视频142exit0，770帧全部解码，父已打开montage/交接/末帧，全身71mesh可见。视频清楚slipFAIL、不是全目标成功。OUTPUT_DEPENDENCY_CONTROL验证相同5800行身体前缀零输出控制只站不走、NN实际前进，承重自身不可归功网络。

D/E. 原始完整验收和最终成功交接视频仍未完成；保留失败、临时检查自动清理，不转平台，目标继续。
