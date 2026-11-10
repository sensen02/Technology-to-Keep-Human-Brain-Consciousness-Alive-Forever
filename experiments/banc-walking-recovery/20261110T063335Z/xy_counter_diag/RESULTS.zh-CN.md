# 单变量反事实：冻结参考XY，保留Z与躯干反馈

本任务仅新增冻结脚本 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T063335Z/xy_counterfactual.py` 及本产物目录，未修改既有教师、参考、物理或其他任务。

条件与原B一致：无附着，零动作先稳定1秒；warp10、vertical gain .25、height gain2，原5ms教师/1ms执行器/12.5微秒物理、同有限PD与限制。唯一干预为私有reference.sample返回的XY目标固定为settled rest、XY速度0；每次调用断言Z目标、Z速度与stance mask保持原source值。没有root/qpos/qvel写入或新外力。

最多要求.8秒诊断，实际594ms时COM<1早停，退出0，wall17.72秒。原B在587ms已同阈值早停，因此以下主比较只用共同587ms前缀。

| 指标 | 原B | XY冻结 |
|---|---:|---:|
| COM最低/窗末 | .998328 | 1.003802 |
| 足载均值/体重 | 1.000564 | 1.001835 |
| 非足峰值原生力 | 0 | 0 |
| LF最大足底高度 | .000709 | .123645 |
| 其余腿最高足底高度 | 均<0 | 均<0 |
| 目标速率边界比例 | .146264 | .079784 |
| 实际关节速度RMS rad/s | .894895 | .592952 |
| 目标净漂移范数 rad | 1.540427 | 1.496288 |
| qt−q跟踪误差RMS rad | .036075 | .035359 |
| 相对settled preload误差RMS rad | .019809 | .018781 |
| 实际ctrl扭矩饱和比例 | 0 | 0 |
| 端点重建raw PD饱和比例 | 0 | 0 |

XY冻结全594ms的COM最低.999607、LF足底最高.131874。它改善了LF抬升并降低速率边界冲突，但没有救稳整体身体。不能再将本次崩溃归因为扭矩饱和；数据也不支持“大跟踪误差导致完全追不上”的说法。实际关节与目标都显著漂移，更应检查支撑腿速度目标与积分式目标执行器的闭环语义。

附加仅诊断的线索：XY冻结.3–.5秒，实测free-root z速度均−1.474而body-COM上升速度指令均+.397；当前stance方程平移部分 `−vz_des−vroot_z` 均为+1.077，即关节足端相对躯干上收。完整足端还含角速度项，且root与COM速度不同，不能凭均值直接断言符号错误；但它表明“实测root twist+修正”与“绝对期望root twist”应分开，不能把虚拟root变量当实际可驱动自由度。

建议后续只隔离一个改动：stance用期望绝对root twist形成关节速度目标，swing继续补偿实测root；或采用父任务正在独立实现的目标位姿shadow方案。需保留静态承重preload、实际有限PD与力矩余量，不能通过写根部救稳。此次不追加实验或搜索。

数据：`XY_FROZEN_1MS.npz`、`RESULT.json`、`CONFIG.json`、`IDENTITY_BEFORE.json`/`IDENTITY_AFTER.json`、`OUTPUT_HASHES.json`。同进程逐数组读回和独立第二进程hash/shape/time/finite均通过退出0。`VERIFICATION.json`注明RESULT字段initial_com实际为free-root translation，勿当COM；比较用的COM列正确来自subtree_com。全部只是1ms诊断，不是原生完整行走验收。
