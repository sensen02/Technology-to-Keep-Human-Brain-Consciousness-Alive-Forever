# P3 命令周期唯一干预：只读源码复核

本文件是源码审查，不是仿真结果。对应本轮1ms命令/5ms感觉对照。

/run/media/sensen/Data2/cell_wound_prototype/run_neural_muscle_closure.py 默认body_dt=100us，派生必须显式prepare_body与episode body_dt_s=12.5e-6；不改DT_COMMAND（它也是5ms感觉时钟），motor_dt_s=.001且motor_translator=None。

同文件_split_motor 336只创建一次decoder；352–365只在感觉5ms边界读取完成body状态；366–370写此前已完成block pending；371–373每block2个neural ticks，376–380步进身体；389–391完成block再consume，供下一block，397记录完成tick。/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/motor_output.py151–157逐sample更新持久EMA，无需新增滤波，tau5ms/100Hz归一/max1/gainNone保持。/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/motor_cadence_schedule.py前一block endpoint等于当前start，是允许边界不是未来事件。

旧5ms episode215–226先算本block神经再写旧pending，split先写旧pending再神经；没有读取d.ctrl的神经计算依赖时两者写值/物理边界等价，需要确定性测试而不是再跑baseline。split证据结构不同于legacy：仅返回末state、无RECORDS，分析不能暗用旧字段；要保存实际事件/消费trace/写入。

独立只读子代理审查未发现因果或重复consume问题；不执行import/仿真，调用0。动态对照后感觉数值可以分化，不将整个250ms差当固定神经输出纯延迟效应。工程调度测试不证明站立/生理正确。原动物参考和解剖/质量力标定限制保持。
