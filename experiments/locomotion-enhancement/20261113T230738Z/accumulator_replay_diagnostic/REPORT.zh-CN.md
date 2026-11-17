# 增量执行器保存数组重放与条件性目标误差

本次0NN、0Core、0物理、0训练。先以实际应用动作通过13条target与感觉编码强门，再读取已保存fcbd评估动作做9条源固定X条件性target积分。三条辅助混合轨迹不作未混合教师累计监督。

|来源|实际target重放最大误差|targeterror／position float32精确|
|---|---:|---|
|stop|0|True|
|left_pivot|0|True|
|right_pivot|0|True|
|walk|0|True|
|V_left_pulse|0|True|
|V_right_pulse|0|True|
|V_left_to_right|0|True|
|V_right_to_left|0|True|
|old_walk_control|0|True|
|actual_fcbd_walk|0|True|
|actual_fcbd_stop|0|True|
|actual_fcbd_left_pivot|0|True|
|actual_fcbd_right_pivot|0|True|

|9条真实监督源|动作RMSE|条件target RMS|峰值rad|最终RMS rad|
|---|---:|---:|---:|---:|
|stop|0.028357999|0.02820446|0.11018609|0.040335096|
|left_pivot|0.067061118|0.098789783|0.59955419|0.12807891|
|right_pivot|0.061619782|0.050587024|0.20279203|0.079367268|
|walk|0.01489491|0.027065618|0.10524542|0.031942582|
|V_left_pulse|0.01760763|0.024692205|0.11939984|0.04102965|
|V_right_pulse|0.017802398|0.03625233|0.14004836|0.051736313|
|V_left_to_right|0.020863203|0.025533631|0.14383461|0.040273656|
|V_right_to_left|0.020026008|0.038480174|0.16678071|0.053165885|
|old_walk_control|0.015109827|0.027731836|0.11615773|0.03467138|

逐通道有符号动作偏差、目标peak/final及rate/angle cap计数见STAT。输入normalizer由真实冻结XML重建并核实际IDENTITY的compiledjnt_range/key_qpos哈希，bounds/rate直接取实际IDENTITY，未从float32感觉倒推。
阶段B只积分保存在真实源X上的网络预测动作，没有将预测target反馈重新送入NN；qpos/载荷/oldhidden等仍来自源条件。目标误差不能证明新身体位移或停步，也不能判唯一根因。三条assisted真实target积了混合动作，未被拿作fullteacher累计监督。
已有fcbd四项身体FAIL保持，离线损失下降不等神经自主运动通过。任何可微累计loss、target-feedback训练或第四模型须另预登记，本诊断不创建候选或更新权重。
