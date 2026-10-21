# S2 单执行器活化时钟细化（执行前声明）

固定目标编译index0，不按结果挑肌肉；原生floor和zero_floor分别，zero control与ctrl0.1分别。初态只读复用已验S2包projected_seated_qpos，act/qvel初始0；每case fresh模型/data，不积分中覆盖状态。全身体自由动力学，不固定/纠正姿态，故力差含几何变化。

仅改变派生编译model.opt.timestep：100、50、25、12.5微秒，10ms共100/200/400/800步。原canonical XML及参数不改，积分器不换；ctrl固定。记录映射actadr目标活化、act_dot、force、qpos/qvel、warning与末端时间。比较共同100us端点粗细差，峰值/终值。没有生物/收敛PASS阈值，不宣称全系统收敛，最后细网格只是参考非真值。记录编译dynprm以解释步长比例，不能改tau迎合结果。父独立核首步Euler act=dt*initial_act_dot和共同端点算差，图实际查看。模型输入/runner/protocol/pose来源SHA前后一致。
