# S2/P2 最终状态采集合同 v1（未来采集适配层，非已集成runner）

## 范围

手动研究原型，不是成品。本轮只实施纯NumPy结构校验和合成测试，无MuJoCo import/model/forward/step，不修改旧冻结runner或旧证据。适用固定MuJoCo3.9.0、NEWTON2、noslip0的primal观测；不扩展PGS/noslip/CG或其他版本。此合同不会自动补齐历史档案缺失字段。

## 同端点来源

未来调用者须在一次明确完成的mj_forward之后、下一次状态写入/动力学调用之前，对所有字段作独立copy。记录MuJoCo版本、solver、noslip_iterations、islands_enabled、nisland、nefc、时间/端点身份、源hash、实际运行时mjNSOLVER/mjNISLAND。这个调用顺序与字段真实性不能由纯数组校验器证明；调用者声明与真实集成验收分开。

保存原始efc_state_global_raw，绝不以推导状态覆盖。另输出带来源标签的final_state：

- nefc=0：empty，不要求岛映射。
- nefc>0且islands_enabled且nisland>0：必须保存island_iefcadr、island_nefc、map_iefc2efc、map_efc2iefc、iefc_state；映射须为完整互逆排列，各岛行区间完整无重叠覆盖nefc。final_state[global]=iefc_state[map_efc2iefc[global]]，来源island_primal_observed。
- 其他情况：monolithic_primal_observed，final_state独立copy自全局state；不混入无效/休眠的island字段，不凭nisland单独假定走island。disabled-island支持显式路径声明。

state数组须一维整数合法枚举0..4（SATISFIED/QUADRATIC/LINEARNEG/LINEARPOS/CONE）；不把cost区域名当physical contact identity或跨端点状态跟踪。原始与最终标签差异报告，但不是自动失败；结构错则拒绝。另保存efc_type/id/J/aref/D/R/force/qacc，用于未来独立分段算术；不得仅以state==1对所有solver/cone套标量公式。当前工具只校验结构，不检验本构或KKT。

## 分岛求解统计

复制solver原始平坦存储及solver_niter。MuJoCo3.9.0槽地址为island*mjNSOLVER+iteration。要求提供所有可能使用槽的capacity：island路径用nisland个槽、monolithic用槽0。nisland不得超过mjNISLAND；每槽保存min(niter,mjNSOLVER)条，并记录niter_total、niter_saved、truncated。不能把第一岛统计当全部岛；超过iteration记录容量时不伪造条目，不把truncated说完整。浮点统计须finite；niter非负整数；输入尺寸不匹配拒绝。统计列顺序固定 improvement,gradient,lineslope,nactive,nchange,neval,nupdate；真实wrapper的布局转换/字段顺序须未来适配集成验证。numeric payload预算不等于RSS上限。

## 校验与边界

工具仅输出拷贝，不修改输入；算法必需字段缺失、不完整映射、重复/越界/非互逆、重叠/空岛/未覆盖区间、错dtype、不支持的solver版本语义、非finite/负niter必须拒绝。路径只按islands_enabled/nisland推导，没有单独capture_path字段；额外键不参与校验，端点时间/身份/来源hash仍是调用者归档要求，不在本工具结构验证范围内。empty路径不检查统计（即使提供了统计）；非空路径要求整个统计存储finite，并校验实际保存的活跃记录nactive/nchange/neval/nupdate为非负整数值，未用槽不作计数语义检查。合成测试须含重排双岛、全局陈旧标签、disabled monolithic、空约束、独立统计槽、容量截断、缺失元数据与非法字段。真实来源身份/多岛运行尚未验收；保持旧82来源SHA不变。

源码依据（已归档版本）：
- https://github.com/google-deepmind/mujoco/blob/3.9.0/src/engine/engine_solver.c#L38-L64 （saveStats）
- https://github.com/google-deepmind/mujoco/blob/3.9.0/src/engine/engine_solver.c#L1020-L1051 （island pointers）
- https://github.com/google-deepmind/mujoco/blob/3.9.0/src/engine/engine_forward.c#L993-L1031 （实际路径与回写）

下一步仅在本适配层审阅通过后设计一次性真实字段/布局检查；未经新协议不追加身体轨迹，更不把本合同通过当物理、生物、收敛或站立行走通过。
