# 左大角度转向末端足节静态可行性

本轮父进程managed bash24 exit0已收集，仅在private MjData进行FK/Jacobian/几何距离与碰撞检查，0积分/0力预测。原模型参数、Core实际data与所有输入身份前后不变。独立新进程audit exit0，记录完整性与数值几何检查均通过。

真实物理left_wider回合已做到30.08248°、slip.40690855、原五门与六腿周期通过；当前唯一严格失败是RH末端RHTarsus4节段接触floor。Tarsus1—5焊接，无独立末端电机，因此只探查现有RH七电机关节在保持原足垫3D任务时是否还有改变末端姿态的余量。

固定8个1ms保存时点，每点分别检查真实qpos、以及只替换RH7角为motor target的假想姿态，后者明确不是教师q_des。16种姿态全部保留，无选择隐藏。

实际8姿态：4个初始已到安全间隙，另4个在最多20次private运动学小步内到达；全部原关节界限/root7/其他35角保持。有效接触includemargin实测.002native，候选安全间隙为.004native。峰值11.973秒真实姿态中，末端最低点-.000657603085→+.004185081818，12个运动学小步、最大累计关节变动.05750485051rad（约3.295°），原pad任务残差约1.07e-10；这不是实际60ms运动或动态追踪证明。Jpad rank3、nullspace4，gZnorm.0505316，解析Jac与中心差分一致。

motor-target替代姿态8项未达到安全间隙；其中末端深穿地，是有限PD/前馈目标的几何代入，不能当作真实姿态或原IK q_des，也不能据此声称实际身体不可行。

结论仅支持一个下一候选：先原teacher primary IK，再在private q_des上对RH加入有界末端clearance次任务，保持pad目标/参考/身体/执行器/碰撞分类；每decision信赖域.005rad，原速率与力矩限制不变，预热前缀严格不改。必须另立预登记并跑完整物理，不能把静态安全当动态安全。特别要同时记录真实h与q_des影子h；若影子已safe但真实仍擦地，应保留为追踪误差，不在运行中更换方案。

结果：/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/left_tarsal_clearance_probe/RESULT_STATIC_PROBE_20261111T174831Z_377047_1794419311801068289.json

独立审核：/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T161007Z/left_tarsal_clearance_probe/AUDIT_STATIC_PROBE_20261111T174902Z_377065_1794419342194613810.json
