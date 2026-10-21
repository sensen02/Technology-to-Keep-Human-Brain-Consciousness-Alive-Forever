# RK4六臂闭环结果：证据已完成，原进程exit1需明确

手动研究非成品。六臂50ms无push/touch/background已实际执行，normal/repeat精确，replay身体精确；sensorycut/silence命令0，motorcut15ms后0。正常新命令活化参考max.00476472/RMS8.20e-5。正常对三cut LF胫节最大角差1.325155/1.293591/1.325155rad，仅短时声明模型链路依赖，不是站立/生物/整体收敛。

两次exit1均报告序列化：第一次normal后ndarray错误，修复后第二次六臂+参考+图完成，最后guard callable错误。当前过滤callable已修但不第三次重跑，不能称干净运行PASS。第二目录recovered-report.json离线恢复，不改旧report RUNNING快照；guard最终资源/输入hash前后未落盘不可补造。

完整报告 `/home/sensen/Desktop/engine修复验收/S1-rk4-closure-20261014T234436/六臂闭环恢复报告.zh-CN.md`，原始六NPZ/参考/图/源码ZIP同目录；图已实际查看。首失败 `/home/sensen/Desktop/engine修复验收/S1-rk4-closure-20261014T234330/FAILURE.zh-CN.md` 保留。实现 `/run/media/sensen/Data2/cell_wound_prototype/run_rk4_neural_closure.py`，不改canonical/旧runner。

独立核验已完成，`/home/sensen/Desktop/engine修复验收/S1-rk4-closure-20261014T234436/independent-check.json`，六臂细事件重建rates/decoded命令误差0，重复重放/时钟/切断/零外载/参考误差/匹配角差吻合，明确clean_runtime_pass=false。停止接口审计；可交付入口的干净退出回归与站立初态资格分开，不扩大时长/gain/姿态搜索。
