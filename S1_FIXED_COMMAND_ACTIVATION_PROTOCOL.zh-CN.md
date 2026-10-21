# 固定真实命令的独立激活动力学资格协议

手动数值研究非成品，S1/S2，不是站立。已有run_activation_timestep_audit.py完成全90统一ctrl=1的5ms、100/50/25us三步长，故本轮不重跑身体、不重复其阶跃实验。新增仅针对冻结P1 normal10x90真实命令的50ms向量化独立标量ODE。

公式按已有原生crosscheck与安装mju_muscleDynamics核验：s=.5+1.5*clip(a,0,1)，d=clip(u,0,1)-a，tau=(tau_act*s if d>0 else tau_deact/s)，adot=d/max(tau,1e-15)。局部clip只计算时间尺度，积分状态不clip。native元数据prm=.0001/.0004/0，Euler；所有90参数相同且actadr顺序须与raw匹配。

固定P1 applied_controls，每5ms切换，初act0。Euler dt=100/50/25/12.5/6.25us，精确整数网格处理切换，不步跨命令边界。全90独立ODE，无身体/接触/外力/姿态。dt100us逐端点与新baseline500x90 act比较，报告array_equal及max误差，误差>1e-12停止解释。原公式与mju_muscleDynamics至少30点含a>1比对，绝对误差<=1e-9。

参考每个命令窗口独立DOP853 rtol1e-11 atol1e-13 max_step2us，再以1e-12/1e-14/maxstep1us重复；公共100us网格参考最大差<=1e-8。Euler每格均比较对应密集参考，误差最大值和RMS分别报告，所有dt输出公共网格全90raw。资格：范围[-1e-12,1+1e-12]、全时全通道最大绝对误差<=.01、RMS<=.001；候选还需与下一更细步长公共网格差<=.005。最细dt没有更细相邻验证不单独宣布通过。参考不合格则整项FAIL，不事后改阈值。

保存原输入/源码/依赖身份、NPZ/JSON/图，图实际查看；不改canonical/tau/actlimited/ctrlrange，不clip轨迹，安装mju只函数调用不load/step模型。2分钟限时，约90x8000 Euler更新+分段参考（向量化独立方程），无新网络/身体仿真。独立代理重算Euler/范围/参考误差；结果仅孤立activation资格，不外推身体/神经闭环收敛，更不选择能站姿态。
