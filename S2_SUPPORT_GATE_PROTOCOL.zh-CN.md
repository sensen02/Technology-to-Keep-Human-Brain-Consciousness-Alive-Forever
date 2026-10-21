# A门固定姿态支撑资格协议 v1（计算前声明）

手动研究原型，非成品。问题：既有投影就座姿态在qvel=0时，肌肉活化箱约束与理想单侧接触/激活限位约束下是否存在全nv广义力平衡？不是稳定性、可控性、真实软接触平衡或神经成功。

复用 `/run/media/sensen/Data2/cell_wound_prototype/outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz` 中projected_seated_qpos，不再投影、就座、settle或搜索；独立canonical load核验XML，qvel=0、ctrl=0。只time0 forward，无step。原生与zero-floor激活稳态候选区间分别取ctrlrange/actrange交集，下限zero仅诊断假设，不改模型；须单状态muscle且无force/joint-actuator force clipping、actearly、gravity compensation、外载/等式/frictionloss，否则拒绝。

先用act=0得到被动力和肌肉基线，每个actuator激活1得到全广义力列；验证moment.T@force，三组固定向量（全.37、线性0..1、交替0/1）核验仿射性，atol1e-9/rtol1e-10为模型工程恒等式非生物容差。记录actuator参数/ctrl/act范围。设置act仅用于静态力映射采样，非真实初态或时间演化，更不证明能从零招募达到。

接触只限实际生成且efc_address>=0的condim3接触；记录位置/frame/dist/friction/J。用mj_jac在接触点构造body2-body1平移Jacobian，四条轴向摩擦射线n±mu1*t1/n±mu2*t2（diamond，MuJoCo pyramidal3允许形状）及四条矩形角射线n±mu1*t1±mu2*t2（outer）分别求解。outer含diamond/椭圆锥，inner不可行不能推outer/真实椭圆不可行。支持不设力上限，所有生成有效接触允许任意非负反力，活动限位efc_J允许任意非负力；这是宽松理想接触/限位平衡，忽略柔顺力与穿透的兼容性、对力的实现能力，不是实际MuJoCo求解器可实现状态。

等式 A*x = qfrc_bias-qfrc_passive-qfrc_actuator(act0)，覆盖根6+关节全部自由度，肌肉活化有箱界、contact/limit非负。逐行以max(1,abs(target),maxabs(A行))缩放（数值预处理，混合力/力矩不相加作生物指标）；scipy HiGHS单LP10s/maxiter10000/primal-dual tolerance1e-9，四个LP（native/zero × diamond/outer），不搜索参数。status0须独立原矩阵重建、活化/非负边界和缩放残差<=1e-7，才记RELAXED_WITNESS；status2记NUMERICALLY_INFEASIBLE_FOR_STATED_RELAXATION，不称严格数学全局证明；其他记INDETERMINATE。若outer不可行仅限固定姿态/活动接触集/活化箱，不能说所有姿态永不站立。若可行，仅解除一个静态必要条件障碍，不等于稳定支撑。

保存原始qpos、力映射、J、接触与限位、缩放矩阵、每解/残差/参数、环境/源码/姿态/XML哈希及图。原始采样先保存，再LP；异常写FAIL保留，输入源摘要前后核对。最多两个模型端点基准/每执行器90映射+三混合与还原+每分支witness复测，总forward预算110，无积分，无神经网络，无安装/服务；2分钟外部超时。记录实际进入forward计数。求解后可把候选act放回同一qpos forward复测肌肉项，但真实constraint不等于LP反力，单独报告实际qacc/force不作动态验证。

既有184案例只回答单肌刺激响应，旧diagnostics/diag_static_support_feasibility.py有历史LP但含settle/姿态处理/缺仿射及floor验收，不重跑或沿用结论。本次明确新增是精确既有姿态+floor区分+仿射核验+内外接触模型分层；不重做全量回归。输出桌面新目录；图须实际打开；独立离线重建后决定下一步。若可行后续优先检查松弛witness与真实柔顺约束兼容性，不直接跳行为门；若不可行先列平衡方向/模型限制，不扩大BANC或调gain。
