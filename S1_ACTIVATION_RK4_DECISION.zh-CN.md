# 激活积分候选：25us经典RK4，未接整身

手动数值研究非成品。固定真实命令独立激活方程，原参数不动。预先max.01/RMS.001/stage范围/相邻细差.005标准下，25us与12.5us经典RK4通过；25us为最粗候选，max误差.00476472/RMS8.20e-5/下一细差.00158844。100us虽端点不超1，stage到2且误差.298失败；50us误差.04059失败。6.25us精度满足但无更细相邻资格不放行。

25us RK48000次RHS向量调用/50ms，是原100us Euler的16倍，仅activation成本计数；Pythonwall不代表整身/GPU。MuJoCo整身RK4、身体收敛、神经时序均未由此验证，禁止直接写act子步进主runner。

完整交付 `/home/sensen/Desktop/engine修复验收/S1-activation-rk4-20261014T230203/RK4激活候选报告.zh-CN.md`、同目录report.json/rk4.npz/comparison.png；图已查看。实现 `/run/media/sensen/Data2/cell_wound_prototype/run_activation_rk4_comparison.py`，执行前协议 `/run/media/sensen/Data2/cell_wound_prototype/S1_ACTIVATION_RK4_PROTOCOL.zh-CN.md`。

下一最小动作仅只读核原生RK4 activation/force/control callback stage时序，然后预声明独立25us身体固定命令重放资格协议（原参数、显式分支、.5ms神经/5ms命令整数比20/200、无stage中间神经更新），经范围确认后再跑。不得用此候选宣布站立或自动姿态搜索；自动目标本次核验交付后停止。
