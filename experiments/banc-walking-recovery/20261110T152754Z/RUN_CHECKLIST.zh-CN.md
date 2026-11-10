# S2/S5 最终确认：全部完成

目录：`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152754Z`。

A. 启动前：PREREG.json先于候选结果写入，最早2500/3000全工程门＋六腿周期通过者用于独立原生确认，两个固定脉冲种子20261111/20261112。完整派生71mesh、49qpos/48qvel/42有限PD电机、无附着、12.5µs物理步、1ms目标积分、5ms决策、±2rad/s/力矩/质量/重力/摩擦均保持，无实际根写入或额外力。角色head仅训练，部署无相位、腿序、教师或计时输入。共享student/actor/physics/teacher等与冻结记录器同源逐位一致；独立静态复核确认脉冲后完整1s settle确保sharedhandover previous=0与实际一致，撤辅无reset。运行前强化baseline trace/decision哈希、deterministicstudent/exacttransition及teacherfiles/STUDENT_CHECKS显式身份闭包。

B. 实际运行：父托管bash172执行confirm_candidate.py，根据已提交结果选2500，独立完整原生记录→冻结wrapper原始验收→seed20261111真实20ms电机脉冲回合→seed20261112回合→同次原生轨迹视频。所有子命令exit0，根job172已完整收集exit0。运行源码未改；SOURCE_LEDGER.json六个固定source/prereg哈希终检一致。多余文件事件等待器bash166取消并收集，无训练影响。

C. 独立结果：原生1232000步/602块/19769188接触事件，80000settle+384000教师+768000ANN，15400端点qpos/qvel/COM/实际命令/目标/力矩逐位匹配，快评估和独立原生逻辑轨迹逐位相同。2880实际神经行，960priming/1920ANN，hidden链/所有模型输出/实际施加动作逐位一致。原生滑移.49411203866010717≤.5、前向.0713190529277483、COMmin1.2561640816296098、非足严格0、实际额外体力/广义力/附着全部0、LF1其余各2周期。

原D1/D2/D3/D4/D4b/D5/D6/D8数值PASS；D7旧UNKNOWN不改写，预先声明≥.1W median4>=3提供已报告更低正阈值的保守充分证据。D6连续因果1ms原生速度差分signedp95=2.464745405722336e-10W，绝对/无滤波/.25/.5/1/2ms/外力记账都保留。实际源边界接近1/5.8/15.4且保留全部间隔；末尾尖峰回归证明不漏末步。旧PROPOSED/UNKNOWN历史不篡改，外部审查后单独FINAL_ACCEPTANCE收口。

两个扰动回合分别全五工程门＋六腿周期PASS，滑移.49269334091673184/.494878086356009；独立检查器核对15420行来源、固定种子42维实际20ms脉冲、1000零命令settle、候选SHA、完整9.6秒自主窗、无重置/额外力和全部身份输出哈希。非原生扰动回合不冒充所有原生D1—D8认证，不扩展任意扰动稳健性。

D. 视频：VIDEO_NATIVE_ROLE_2500/ACTUAL_ANN_HANDOVER.mp4来自同一原生运行，770/770帧解码、30fps/20ms取帧/1.667倍慢放，15.4物理秒，完整1s稳定+4.8教师+9.6ANN。渲染0积分、保存qpos/qvel精确重建、编译物理与源不变、71不透明mesh。父实际打开六阶段蒙太奇和独立5.801s接管帧/15.381s末帧，全身/连续接管/相对地面位移可见；VIDEO.json存真实签注。面板1ms汇总slip.4927888，正式native.4941120，两者身份明确。视频SHA256=b2a190e1a5cb9ca7adea0a1e0c40003a00485cbd850a9dba521f2401af193ba3，桌面副本交付。

E. 最终合取：bash174运行close_acceptance.py退出0并已收集，FINAL_ACCEPTANCE.json=PASS，0物理步。逐项核验审计输入与固定依赖哈希、两扰动所有门、当前神经依赖证据、视频解码/来源/目视签注/源码不变。当前依赖范围为真实同前缀零输出对照＋当前保存hidden感觉切断命令变化；旧300实际感觉切断只作谱系历史，不谎称当前2500物理断感觉。

清理：正式样本全部核验后，桌面旧300失败视频/图及本轮2000失败视频副本被替代并已实际删除；项目全部必要原失败证据和可恢复checkpoint保留。临时fixture自动清理，CONFIRMATION_STATUS.tmp原子替换无遗留。所有相关root和child jobs已收集，无运行任务。

最终报告：`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152754Z/ROUND5_PROGRESS.zh-CN.md`。本次已授权的替代ANN/派生完整身体/有限PD、9.6秒自主行走实施验证完成；不声称原连接组、冻结原身体、SI/生物有效性或任意时长稳健性。
