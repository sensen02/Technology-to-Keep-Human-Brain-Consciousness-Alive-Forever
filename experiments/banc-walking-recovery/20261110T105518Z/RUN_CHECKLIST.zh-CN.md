# S2/S5 可训练抑制性误差连接：启动登记

新目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T105518Z`，父控制接口 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T104706Z`。本实验是替代ANN结构先验，明确不是原连接组或生物真实性宣称。

A. 新FF网络的感觉MLP不直接接收目标记忆相关通道及上一动作；另有42条神经连接接收各自target−q误差，其权重为可训练负softplus，按实测教师局部偏导初始化10H。输出在actor内部相加后tanh，仍输出原42维变化率，所有可训练权重都参与学习。部署不增加ANN外固定纠偏控制器，不调用教师、不使用参考/相位/计时。critic保持raw，物理有限PD不改。

- 固定数据 `/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T094051Z/SMOOTH_DATASET`，相同train0/1,val2，原始BC目标/Adam lr.001/FF1000步/std.05，初次从头训练作明确结构对照，不混成对标签损失/归一化。模型身份包含连接符号/输入mask/H初始化；旧rawFF不能误载。
- 预检查：T两观测块一致扰动时输出对T负对角、offdiag0、上一动作梯度0；物理感觉梯度与抑制权重梯度非零，权重确实更新，critic不受mask；权重/H/RNG/Adam完整保存重载与输入契约拒载；实际20ms交接不reset。
- 这些数学性质不能证明完整接触身体稳定，也不承诺全部局部特征值收缩，softplus增益可能太大；必须真实纯神经阶段验证。
- 身体完整71mesh、重力/摩擦/有限PD/力矩/2rad/s/12.5us物理/1ms积分全部保持，不写实际根姿态或加辅助力。

B/C. 32结构/77学生检查通过后实际启动主训练bash104，1000步lr.001完成退出0、权重重载一致，独立IDENTITY.verify通过。所有42增益更新。bash105新进程同次1s站稳＋4.8s实际teacher＋9.6s纯student（没有后续teacher动作查询）完成exit0并收集，交接全部状态连续；自主足载.1235、COM最低.54284、非足24.5323native、滑移5.0095、forward.11452，失败完整保存。只有前向单项通过，不选作达标模型。ROUND3_ALL_RESULTS/ROUND3_ACTUAL_STATES实际打开。目标T负反馈与实际q正反馈代数耦合在JOINT_TARGET_COUPLING_AUDIT单列，冻结身体子系统性质不能冒充全物理稳定。

D/E. 最终原始验收与成功交接视频仍未完成；合格教师视频单列，不当神经成果。临时检查fixture自动清理，失败对照保留。目标持续行走，不转其他平台。
