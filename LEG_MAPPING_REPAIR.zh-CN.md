# 旧感觉API按腿映射修复

手工研究原型，不是成品。当前主闭环直接生成ChannelQuantities调用encode，不经过本报告的旧assign/quantities接口，因此这不是当前下落根因的证明。

旧assign(leg,...)忽略leg，把所有映射保存到同一个_rows；第二条腿的配置覆盖第一条。未配置时还有默认前11行兜底，缺失速度默认为0，参考长度固定>=66。这可能让调用者把错误关节信号送到正确感觉细胞，必须修复，但不能把修复混同于真实闭环性能改善。

新增合成回归测试先复现了跨腿覆盖等失败。计划改为每腿显式配置，未知/未配置腿拒绝；参考向量任意有限非空1D，行号精确整数且有界，重置参考清除映射；量/速度长度严格一致，不补0。保留显式跨腿共用同一组行的合成实验，但调用者必须为各腿分别声明。

唯一旧验收调用者此前只assign LF却对6腿取量，将改为显式6腿共同映射同一组合成行，保持其原实验意图且不冒充解剖映射。

实现完成，16项按腿映射回归、17项已有编码器输入验证、4项选择性输入测试合计37项通过。主encode路径没有改变；连同3项初态范围报告测试，父任务共复跑40项全部通过。完整7组真实闭环再次运行，全部22个NPZ数组与修复前63ab的前序f10c基线逐位一致，定向事件计数吻合；新增范围警告明确打印。新包为`/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/runs/63ab7236-bfda-4cc7-8866-b83e64a7e6fb`。这证明默认合法路径保持不变，不是运动表现改善。

修复文件：`/run/media/sensen/Data2/cell_wound_prototype/engine/embodied/sensory_channels.py`

显式合成调用者：`/run/media/sensen/Data2/cell_wound_prototype/run_neural_rebuild_13_channel_frontend_acceptance.py`

测试：`/run/media/sensen/Data2/cell_wound_prototype/test_sensory_leg_assignment.py`

