# S2/P2 原始接触观测预注册（实现/仿真前）

仅 NEW 两案例：native_floor、target actuator 0、pulse ctrl=0.1 [0,1ms) 后0、同历史 projected_seated_qpos、10ms、RK4 dt25/12.5us；无搜索，无生产 XML/tau/pose 修改。每完整步记录提交 ctrl/time，初 qpos/qvel/act/ctrl；与原 run_s2_contact_clock.py 相同每步 mj_step 后 mj_forward，每共同25us端点取观测（不改变原观测语义）。不新加每步姿态覆盖。

端点逐 contact 保存 geom1/2、pos/dist/frame、mj_contactForce local6、efc_address/dim/exclude/friction；保存按排序geom第二geom受力的世界pair合力（frame.T × local前三；反序负号）、ncon、nefc、solver_niter、efc_force/state/type/id。contact和efc用数值扁平数组+端点offset，不pickle，硬上限每端点contact512、efc4096；逐案例写盘并释放，不驻留两案例动态状态。efc地址仅端点内引用，跨时钟/时间不默认行index匹配。

离线仅比较预定[1.5,2.3]ms、[7.7,8.7]ms及pair(0,26)/(0,25)：完整端点表、逐pair点数、距离/法向和局部力范数分布(min/median/max)、世界合力差峰、峰±2端点 ncon/点数/距离/约束state直方图及相邻直方图变化。state直方图变化非同一接触身份转移。约束type/id/state整体及pair关联行保存，pyramidal dim映射使用编译cone，excluded/-1无有效行。关联不许命名物理原因，因果仍unknown。

独立工具不导入新runner/分析工具，从保存frame/local重建世界pair及总和验算；这只是原始接触算术重放，不进行新动力学。容差rtol1e-12/atol1e-14仅算术一致性，不是力/支撑收敛阈值。测试非对称旋转、反序符号、变长含空efc序列与allow_pickle=False读取。

新唯一桌面目录中文报告+plot必须实际read_image。记录源码/规范XML/pose前后SHA、venv_body环境、CLI/测试实际退出码。保留历史dirtytree；不reset/服务/硬件/装依赖。手动研究非成品；不证明生物标定、身体收敛、站立行走。父必须review后才最终接受。
