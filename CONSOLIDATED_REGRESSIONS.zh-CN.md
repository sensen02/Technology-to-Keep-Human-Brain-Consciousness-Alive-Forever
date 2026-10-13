# 闭环汇总回归

手工研究原型，不是成品；测试通过不等于站立成功。

新增固定入口 `/run/media/sensen/Data2/cell_wound_prototype/run_closure_regressions.py`，显式运行20个轻量测试模块（不是整个仓库自动发现），保存测试源码哈希、数量、失败/错误、时间及输出。共150项通过，约2.5秒。

覆盖时钟、原子包、连接方向、扰动推论、运动清单与滤波、感觉输入与逐腿映射、初态报告、历史证据、短窗/200ms已保存数据及源码归档安全。

首次合并运行暴露1项测试污染：test_neural_command_audit把整进程sys.modules中的其他测试导入误认作本审计加载仿真。已改为独立解释器执行audit并检查禁止模块，保留约束且消除顺序依赖。修复文件 `/run/media/sensen/Data2/cell_wound_prototype/test_neural_command_audit.py`。

此入口不重新加载身体或运行神经episode；实际长短窗重放和原生身体步长测试有独立受守卫运行证据。不能把轻量回归替代真实仿真。

结果：`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/consolidated_regressions.json`
