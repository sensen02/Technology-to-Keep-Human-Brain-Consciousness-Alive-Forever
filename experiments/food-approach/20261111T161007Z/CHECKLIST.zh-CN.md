# 大角度教师首档运行检查表

## 启动前（已登记，尚未运行）

- 根目录：/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z
- 预注册：/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/PREREG.json
- S4/S5；新问题是9.6s实际左右各≥30°，保持原worldX推进门及所有已声明物理门。旧直行验收不变。
- 唯一教师变化：请求yaw ±.06；speed_scale1；一秒smoothstep。归一化yaw/.06为新命令契约，不改旧source。
- 控制来源：全程明确算法教师，非神经自主；完整派生71mesh/42有限PD/noadhesion/noextra/no root trajectory writes。
- 预算：两完整回合单进程顺序、单核单线程；1s settle+4.8s prime+9.6s diagnostic。失败证据保留。无ANN训练。
- 原生记录：使用冻结成熟NativeRecorder逐12.5us保存，分块2048行；完整1ms轨迹供同次渲染；5ms原始238感觉/42动作/物理命令/新归一化命令/教师phase标签分列。
- 身份：运行前capture完整导入/XML/资产/hash/编译数组/执行器；初始runtime另存；结束复核hash和runtime；独立进程重建Core核验。
- PPO/GAE/BC优化器项不适用：不训练。观测动作/相位泄漏/动作界/finite/同步/外力与周期均独立审核。
- 后台job/PID/命令/exit：启动后写JOBS.json；不另开物理或渲染进程。
- 视频：本阶段优先物理教师可行性，完整同次轨迹保留；未渲染，未目视视频。
- tracker由父代理负责，本子任务不写。

## 运行与结束

待执行，不能提前勾通过。运行完成后以RESULT、独立AUDIT与REPORT填充结论。
