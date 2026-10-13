# 运动输出清单边界修复

**手工研究原型，不是成品。** 当前正常清单没有因此证明错误；本修复针对可复现的错误输入静默改变路由，不调整滤波、增益或肌肉命令。

运动输出构造器过去直接int转换n_neurons/n_actuators和local_indices，浮点/布尔输入可能被当成另一个真实神经元索引。actuator_index比较同样接受bool或整数值浮点。明确拒绝非法身份比静默选择错误细胞更安全。

保留非负整数计数、合法空集合、跨输出共享运动池、声明的部分池均值归一化。只拒绝非法输入，不猜测缺失连接，不给未覆盖肌肉补命令。

## 验证

新9项运动清单测试修复前暴露30个失败子项及7个意外OverflowError，修复后通过。真实12000神经元/90输出清单下，4个确定性区间（含切断）的完整命令/记录摘要前后相同。

还发现感觉索引与调用者NumPy数组共享存储：修改原数组把应进入细胞0的0.3nA输入改送细胞1。新增测试实际复现后修复为持有已校验索引副本；运动池也采用同样的所有权隔离。没有设置新神经路由，只防止外部后续修改绕过校验。

修复文件：
- `/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/motor_output.py`
- `/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/sensory_channels.py`

测试：
- `/run/media/sensen/Data2/cell_wound_prototype/test_motor_manifest_validation.py`
- `/run/media/sensen/Data2/cell_wound_prototype/test_sensory_channel_validation.py`

父任务48项合成/既有回归通过，完整7组真实闭环与前序全部22个数组逐位一致，实际源码快照完整性核验通过。默认有效输入的行为未改变。

最新运行包：`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/runs/add80e12-4b1b-45fc-90be-2336ddf7757c`

另外2项实际observe_quantities函数的合成测量契约测试通过：位置/速度不同地址、有符号角差、逐接触力范数忽略力矩、无接触零输入及调用不改身体数组。测试文件：`/run/media/sensen/Data2/cell_wound_prototype/test_sensory_physics_observation.py`。这些是测量接口测试，不是组织应变模型或生物极性验证。


