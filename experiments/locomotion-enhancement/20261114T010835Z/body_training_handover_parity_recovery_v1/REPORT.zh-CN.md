# 身体交接与训练源保存数据对齐诊断

只读取实际保存数组，0NN／0Core／0物理。全部95九阶段已完成后才允许运行；本报告不改任何行为门、不证明泛化或policy容量。

|case|首输入/命令/保存状态一致|首输出实测差≤1e−12|是否受控可期待精确|源old256记录|原行为PASS|
|---|---|---|---|---|---|
|stop|True|True|EXPECTED_EXACT_GIVEN_MATCHED_SAVED_WARM_HISTORY_AND_CALIBRATION_PRIME_PROOFS|NOT_RECORDED|False|
|left_pivot|True|True|EXPECTED_EXACT_GIVEN_MATCHED_SAVED_WARM_HISTORY_AND_CALIBRATION_PRIME_PROOFS|NOT_RECORDED|False|
|right_pivot|True|True|EXPECTED_EXACT_GIVEN_MATCHED_SAVED_WARM_HISTORY_AND_CALIBRATION_PRIME_PROOFS|NOT_RECORDED|False|
|walk|False|False|NOT_CONTROLLED_FOR_COMMAND|RECORDED_OLD256|False|

时间：首preaction5800ms，trace5799是前一毫秒末端；第一个动作实际积分trace5800..5804，首posttarget取5804。教师source样本start960／warm0，完整NNsource start1160／warm200。walk训练命令按真实raw_turn恢复，不假设正好0。
缺失教师old256为NOT_RECORDED，本分析未运行神经补算。源教师首动作是训练标签，不是新模型预测；只以EVALUATION和CONDITIONAL600的新模型保存输出作首NN参考。
若首接口一致，只支持初始对齐，不支持后续真实闭环或行为成功；若不同，应先核warm/input/command/state差异，不能据此判policy容量。
具体首差列／预热qpos/qvel/target全前缀／physicalcompiledSHA／normalizer字段核对见RESULT。
