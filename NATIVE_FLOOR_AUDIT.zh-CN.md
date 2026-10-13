# 冻结原生正控制下限的严格验收边界

**手工研究原型，不是成品。** XML保持冻结，不通过修改act或肌肉参数让零命令对照更好看。

原生90肌肉控制下界为0.0001。神经命令为0时，MuJoCo内部控制限幅仍可能生成微小激活。本轮使用5ms实际零ctrl积分，以及已保存正常/输出静默轨迹在相同qpos/qvel上的scratch act=0分解，量化残余激活和主动分量。scratch分解不驱动真实轨迹，不代表新增控制器。

必须区分：控制命令为零、肌肉激活为零、主动力为零、被动力为零四件事。此项阻碍严格神经唯一驱动验收，但不自动证明它是身体跌落的主要原因。

实际50个0.1ms身体步完成，4项测试通过并由父任务复跑，图已实际查看。全部90个ctrl读回始终严格为0，但每步90个act均为正。5ms状态与保存输出静默组qpos/qvel/act逐位相同。

原生初始act_dot为2/s，mju_muscleDynamics在限幅后的0.0001输入计算得到相同结果。0.1ms欧拉步产生首步act=0.0002，随后趋近0.0001，5ms为0.000100143659362；此为冻结积分/激活动力学的实际响应，不是额外注入神经驱动，也未报数值警告。

保存静默轨迹5–50ms的同状态主动分量：最大单肌肉绝对力0.04260387模型力，肌肉总力最大33.96334653。每样本主动/总肌肉力向量L2比为0.1266%–0.4918%。非零但很小，不能据此认为它解释主要运动或跌落。角广义主动分量范数峰0.00512376模型力矩；自由根平移的内部肌肉广义力为0，不能将肌肉力和当外部支撑。

两份scratch的qfrc_passive差严格为0，act=0的肌肉力偏置仍可能不为0。没有构造修改下限的模型，也没有将scratch act=0覆盖真实状态。

结论：冻结模型下，严格零神经命令不能获得严格零活化/主动肌肉分量。这是精确验收障碍，但不是新发现的站立失败主因；不能通过删除下限或把残余称为被动力来隐藏它。

## 产物

- `/run/media/sensen/Data2/cell_wound_prototype/run_native_floor_audit.py`
- `/run/media/sensen/Data2/cell_wound_prototype/test_native_floor_audit.py`
- `/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/native_floor_audit.json`
- `/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/native_floor_audit.npz`
- `/run/media/sensen/Data2/cell_wound_prototype/outputs/neural_muscle_closure/native_floor_audit.png`

