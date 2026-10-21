# 原生RK4身体固定命令结果入口

手动研究非成品。四case按协议完成：25us baseline/repeat/hookoff及12.5us refined；10000step/50004动力学评估/40003callback。native act与独立经典RK4最大差1.11e-16，参考误差25us .00476472/RMS8.20e-5，12.5us .000439966/RMS6.25e-6，所有stage/端点范围有效。repeat全部数组精确、hookoff状态精确、零外载/warning有raw。通过仅activation与时序，未身体整体收敛/新闭环/站立。

typed body差root平移.0001282模型长度、旋转7.79e-5rad、标量角.0001786rad；contact模式差7/2000样本，禁止图贴近即称收敛。hookoff report stage_checks统一true不代表有阶段日志，该臂无hook，证据是hook-on日志加端点等价；完整报告已警告。

报告 `/home/sensen/Desktop/engine修复验收/S1-native-rk4-body-20261014T233330/原生身体重放资格报告.zh-CN.md`，四NPZ/report/source ZIP/native-replay.png同目录，图已实际查看。实现 `/run/media/sensen/Data2/cell_wound_prototype/run_native_rk4_body_replay.py`，canonical/主runner/旧数据不改，无新神经。

下一最小动作不重跑四case：预声明同显式RK4分支神经/解码/身体时钟接口与短时因果协议，stage不可推进网络，.5ms/5ms/25us整比20/200。身体接触收敛与初态来源支撑仍单列，不以数值修复制造站立资格。自动目标本项交付核验后停止。
