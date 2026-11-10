# 有界慢周期教师诊断

仅算法教师诊断，非网络部署/自主行走/冻结身体验收。四个周期60/200/400/800ms，各4周期；所有条件adhesion=False、stride=.03、lift=.08、tau=.02。每条件先300ms零动作settle，再以实际足部位置重设rest；这与旧reset即运行且附着开启的教师不同，不能直接归因历史差异。除周期外本次四条件相同。

脚本：`/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T052624Z/slow_teacher.py`。

运行：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T052624Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T052624Z/slow_teacher.py`。托管job bash-26正常完成，退出0，总wall约79.43秒（RESULTS写入前wall79.09秒）。无额外扫参或增益变化。

| 周期ms | 前向原生长度/s | 足载均值/W | 非足峰值原生力 | COM最低 | 滑移比 | 严格完成事件LF/LH/LM/RF/RH/RM |
|---:|---:|---:|---:|---:|---:|---|
|60|−0.0000377|1.00114|0|1.22327|90.65|0/0/0/0/0/0|
|200|−0.001370|1.00031|0|1.21912|234.19|0/0/0/0/0/0|
|400|0.009430|1.00015|0|1.21742|44.67|0/0/0/0/0/0|
|800|0.016794|1.00012|0|1.21612|10.64|3/0/1/1/0/0|

所有周期在当前分支support/nonfoot/COM三项通过，forward/slip未过。800ms的stride/period=.0375<.05速度门，这个门未过不等于机械不可行；也不能仅靠这个规划比值断言完整身体的实际速度上限。近零推进时滑移比会放大，须结合原始脚速解释。

严格事件：足载<=.05W启动；仅在<=.05W样本累计相对胸部x前移与净空，要求最大前移>.03且最大padz>.015；随后足载>=.1W且padz<=.004算完成。分别保存事件起止和cycle索引；只抖动、卸载但不抬足、前摆但不回载均不算完成。

慢周期确实降低限速压力：LF/RF原始dq/2的p95从60ms约1.008/.933降至800ms .385/.347，800ms几乎没有任何关节请求超过2rad/s（只有RH约0.0001比例）。800ms的LF/LM/RF出现真实按定义完成事件，说明周期对这个控制器具有实质影响。

但放慢尚未解决所有腿。800ms LH/RH/RM最大padz仍−.0211/−.0060/−.0099，未达到正净空；六腿并未全完成，不能称协调步行。60/200ms所有腿最大padz为负；400ms只有RF达到约.0102，仍低于.015门。

800ms整窗COM世界x位移+.05374，400ms+.01509；本次无匹配全长零动作对照，不能把全部推进严格归因于有效换步。证据支持“此前60ms过快是合理且有部分实测支持的控制器因素”，不支持“放慢即可走成”，更不支持身体不可能。当前四条件仿真时长随周期变化，固定4周期便于周期完成比较，但瞬态/持续时间混杂仍存在；不据此给出纯周期因果效应量。

保存：CONFIG_PREREG.json、INPUT_IDENTITY.json、RESULTS.json、OUTPUT_HASHES.json及每条件settle/period NPZ。记录1ms状态、原始dq/2、执行命令变化率、实际关节速度、目标误差、相对/世界足位置、padz/load/slip、COM/qpos/qvel；实际物理步12.5微秒，但本诊断未启用原生记录器，不能当原生峰值或原目标D1—D8认证。

验证：所有NPZ重新加载有限、时间差1ms、请求变化率<=2rad/s；输入hash未变，OUTPUT_HASHES内产物逐项SHA256核对通过。JSON完整状态complete=true。无视频交付，未宣称目视核验。
