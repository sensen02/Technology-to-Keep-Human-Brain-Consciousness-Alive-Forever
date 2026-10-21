# S2/P2 stationarity实际采集执行界

本轮用户“继续”实施上一轮协议的未来采集，不改变历史协议已完成的纯合成范围。手动研究原型非成品。仅canonical和既定artificial双球各fresh一次mj_forward，禁止step/积分、参数/pose/gain搜索；初态复用已冻结forward_adapter。旧脚本/协议/hash保持不变，新独立runner。

已事前读取安装MuJoCo3.9.0 docstring：mj_fullM(m,dst,M)，输入为稀疏惯性M，不是自动假定旧qM；mj_mulM(m,d,res,vec)。采集明确使用d.M，保存原始M及m.M_rownnz/rowadr/colind，独立可重建稀疏下三角/对称矩阵；若字段/布局不合停止，不换姿态/反复forward。

每模型创建后、MjData和dense M分配前检查nv<=256；一次forward后nefc<=4096/nisland<=mjNISLAND、dense J；只读复用旧capture函数获取完整岛内标签/映射/分岛统计。保存M_dense、M_sparse、存储metadata、qacc/qacc_smooth/qfrc_smooth/qfrc_constraint、库内mulM(qacc)、J/f及所有单位标签。mj_fullM/mj_mulM前后检查state及M/J/force/qacc/aref/smooth/constraint/岛标签等所有捕获动力字段字节不变。

计算使用纯工具tools_s2_stationarity；DOF label依据free前3/slide为平移，其余free/ball/hinge为旋转。M/J和所有numeric fields写盘前finite校验、最终完整payload（含初末状态/统计/M）<=64MiB，非RSS界；不对真实残差设PASS阈值。原82来源/2旧NPZ、4adapter来源/2adapterNPZ及新协议/工具/runner来源前后SHA核对。

输出独立桌面NPZ/report，每DOF残差/项/分母完整保存及分单位统计。图按模型和单位分开，父实际查看。独立review仅离线重算scalar dot、稀疏重建、group stats、库内M乘法差、hash/state/mapping/stat slots，不再执行forward。不做KKT/生物/收敛/静态站立验收。
