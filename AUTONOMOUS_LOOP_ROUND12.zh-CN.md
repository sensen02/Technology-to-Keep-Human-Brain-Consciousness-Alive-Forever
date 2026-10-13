# 第十二轮：实现可选区间固定系数解析电压更新

目标未完成。飞行暂缓，手工仿真原型。没有添加行为控制。

## 实际改动

engine/neural_cond.py新增voltage_method='frozen_exact'；默认仍为backward_euler，原计算路径逐项保留。新方法在单步固定总电导、输入、适应电流时用Vinf+(V-Vinf)*exp(-G*dt/C)更新，用expm1减少小量消减。电导/适应衰减与阈值仍分步处理，因此不是完整连续网络的精确解或生物标定。

TierConfig新增同名选项，NeuralTier初始化与reset均传递；诊断入口新增--voltage-method。未改权重、阈值、感觉增益、肌肉映射或物理参数。不是根据站立优化算法。

## 验证

新tools_test_voltage_method.py六项测试通过：固定系数解析解到12位小数，默认BE逐位一致、BE离散式、非法值拒绝、零输入/所有权、不应期。

实际0.3秒四分支：frozen_exact完整147740脉冲、终高0.72214mm；感觉切断0；读出切断147345且命令0；传播切断823仅感觉脉冲且命令0。240区间命令来源检查通过，图已核看。身体仍塌陷。

再次运行默认BE完整四组，整个cases数据（全部命令、脉冲、身体步）与第十轮逐值一致，证明新功能未悄悄改变默认结果。

本轮修改修正了固定系数子问题的积分误差，但没有解决缺失感觉/肌肉支配及生理参数标定。不能把数值精度进展说成行为自主完成。

源码改动：/run/media/sensen/Data2/cell_wound_prototype/engine/neural_cond.py、engine/embodied/adapters.py、diagnostics/diag_autonomous_sensorimotor.py；新测试 /run/media/sensen/Data2/cell_wound_prototype/tools_test_voltage_method.py。
结果 /run/media/sensen/Data2/cell_wound_prototype/outputs/autonomous_solver_round12/ 和 /run/media/sensen/Data2/cell_wound_prototype/outputs/autonomous_solver_round12_default/ 。运行后只补充neural_cond模块文档头，因此历史运行哈希对应当时版本，不回写历史哈希。接口文档已再生成。

桌面 /home/sensen/Desktop/cell_wound_prototype/autonomous_round12/ 。
