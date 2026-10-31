# P3：标准步行物理预热 → 神经接管实施约束

状态：实施中，尚未完成算法辅助物理步行或接管。核心必需的交接设计；不得当行为通过报告。

## 用户要求
先修正交叉前足，找到可追溯的真实标准步行；完整身体在真实物理模拟中先由算法辅助走数步，再撤除辅助启动神经接管。全部阶段需在同一身体、同一状态时间线上展示。

## 采用路线
1. 步行参考优先使用已安装的真实自由步行处理数据：`/run/media/sensen/Data2/cell_wound_prototype/venv_body/lib/python3.12/site-packages/flygym_demo/complex_terrain/assets/single_steps_untethered.pkl`；由`/run/media/sensen/Data2/cell_wound_prototype/venv_body/lib/python3.12/site-packages/flygym_demo/complex_terrain/preprogrammed.py`的PreprogrammedSteps读取，注意v2右腿roll/yaw符号转换。不是未经处理原始测量，处理、步周期与单位据元数据记录。
2. 参考身体与当前90肌肉身体的七个关节并非一一同构：参考含Tarsus1关节，当前含独立Trochanter yaw。禁止直接复制42角。先在参考身体做正运动学得到身体坐标系足端轨迹，再在当前身体逆运动学重定向。
3. 保留当前完整90肌肉身体和可信运动池。允许预热阶段向42腿关节施加显式算法辅助力矩，但自由根6广义力必须始终为零；不脚本平移根，不在运行中覆盖qpos/qvel，不固定身体。原肌肉被动动力学如实保留。
4. 标准步行参考是步态依据；相位同步、时间缩放、轨迹缩放与关节反馈控制均作为算法辅助的工程设定标明，不冒充连接体神经运动。

## 本轮已验证的输入（尚非步行）
- 标准参考：`/home/sensen/Desktop/engine修复验收/s5-standard-gait-reference/20261028T040000Z/STANDARD_GAIT_FK.npz`，33个闭合相位、6足位置；处理后的周期0.135秒（7.407Hz），来源45点×0.003秒。父独立核对5项哈希、有限值、精确周期闭合、左右前足分侧通过。
- 参考点是第五跗节body原点，不是爪尖；当前静态原型以Tarsus5网格选定顶点作足尖。动态重定向前必须统一点定义，不得直接拼接两种足点坐标。
- 候选静态姿态：`/home/sensen/Desktop/engine修复验收/s5-noncrossing-static-prototype/20261028T041000Z/STATIC_STANCE.npz`。42关节范围未改、六目标误差低于2.7e-6；优化器达到100次函数评估上限，不能称形式收敛。只验证几何候选，0身体积分、0神经步；不替代真实承重或行走验证。
- 新候选姿态不同于032姿态。旧模型的42springref和90条lengthrange仍对应旧姿态，下一派生模型须一次性重新对齐并记录假设，不得直接将候选qpos塞给旧标定后称已综合修好。

## 先过姿态与步行门
- 初态：左右足保持各自身体侧，至少LF足端y>0、RF足端y<0；足端分侧不足以证明腿段不相交，必须检查前腿链空间距离并实际打开前视/俯视全身图。
- 姿态应在声明关节范围内，六足可接触，头胸腹均不接触地面。不能把IK优化成功当几何或动力学成功。
- 辅助步行阶段须记录真实触地/离地、足地反力、身体位移与姿态；目标先验证数个完整步周期。只有腿摆动或身体倾倒滑移不得称走几步。
- 算法参数调试允许，但每次新派生配置与真实调用、失败均记账；不得把没有走成的预热直接接管后剪视频当交付。

## 必须重写连续交接入口，不能直接复用旧episode初始化
`/run/media/sensen/Data2/cell_wound_prototype/run_neural_muscle_closure.py`的episode在第155–160行新建MjData并清零qvel/act/ctrl，所以不能在辅助走完后直接调用它：那会丢失预热速度和肌肉状态。

新入口应：
- 同一MjModel和MjData连续积分；保存交接前后完整qpos/qvel/act/ctrl/time及必要积分器状态。
- 在明确t_handoff边界把42辅助腿力矩写为0，并保证此后每步保持0；xfrc_applied与辅助根力始终为0。
- 保存边界撤除前、撤除后（未积分）状态：qpos/qvel/act/time必须逐值相同，控制来源可按声明切换，不把ctrl也强行要求相同。
- 按用户“再启动神经元接管”的顺序，初版在交接边界初始化历史静息连接体网络并开始真实感觉编码；明确这是从静息启动、无神经预激活。若以后改为预热阶段让神经网络观察感觉但屏蔽输出，应作为另一明确方案，不混称。
- 神经侧复用现有ConductanceNetwork、ChannelSensoryEncoder、MotorOutput和可信池load_runtime_pool；保持完成窗解码、下一窗保持的因果关系。
- 感觉参考与物理当前状态分开记录，不把转入神经阶段当作重新归零足角和负载。

## 交付与验证
同次全身视频明确标注ALGORITHM-ASSISTED WALK / HANDOVER / NEURAL ONLY与物理时间；禁止裁掉失败交接。轨迹同时保存辅助力矩与神经命令，使接管后辅助为0可独立核验。

本设计文件与已声明的静态IK原型均为核心/重要辅助工程材料，不是已实现预热。尚无符合本流程的新行为视频。用户旧修复、canonical与历史封存不动；本轮无授权删除用途不明旧资产。
