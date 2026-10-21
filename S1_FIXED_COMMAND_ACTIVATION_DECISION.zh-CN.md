# 固定命令activation资格决策

手动研究非成品。去重旧统一阶跃后，新真实P1固定命令50ms独立90标量ODE已运行。100us Euler精确复现旧baseline所有活化（误差0），原生RHS28点差0，双DOP853参考公共网格差5.66e-15。活化超范围可由独立积分解释，不是需要用身体接触/神经变化才能解释。

预声明误差门max .01/RMS .001/范围有效/相邻更细差.005；100/50/25/12.5/6.25us均未获孤立activation资格，最细max .02738。不得以50us不超1就换入完整闭环，不放宽阈值，不静默act clamp/改tau/gain。report超1+1e-12计数753与旧严格>1计数965口径不同。

完整报告 `/home/sensen/Desktop/engine修复验收/S1-fixed-activation-20261014T225432/固定命令激活动力学资格报告.zh-CN.md`；activation.npz/report.json/activation.png同目录，图已实际打开。执行 `/run/media/sensen/Data2/cell_wound_prototype/run_fixed_command_activation.py`，协议 `/run/media/sensen/Data2/cell_wound_prototype/S1_FIXED_COMMAND_ACTIVATION_PROTOCOL.zh-CN.md`。无新身体或神经仿真，未改canonical/旧数据。

下一最小动作先预声明比较原生高阶激活积分路径/独立参考误差与成本（或受控更细网格）；不要先跑身体全闭环或无限细化。任何新数值/子步分支须声明身体采力时序、因果时钟、原参数不变并分支验收，不把好看动作当数值资格。当前自动目标完成交付与独立核验后停止，稳定站立/行走仍未达成。
