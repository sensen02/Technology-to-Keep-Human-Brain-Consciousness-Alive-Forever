# S2/P2 同端点stationarity最小观测协议 v1

## 本轮范围

手动研究原型，不是成品。本轮仅协议、纯数组工具和合成测试，不import MuJoCo、不加载模型或forward/step、不追加真实轨迹，不修改旧runner/冻结来源。历史包缺同端点M或qfrc_smooth，禁止事后拿另一次forward的质量矩阵拼接旧force/qacc并宣称同端点验收。

## 官方数学与独立量

MuJoCo3.9.0 PrimalUpdateGradient：grad=M*qacc-qfrc_smooth-qfrc_constraint。
来源 https://github.com/google-deepmind/mujoco/blob/3.9.0/src/engine/engine_solver.c#L1201-L1210 ，本地已归档完整源码 `/home/sensen/Desktop/engine修复验收/S2-contact-state-audit-20261014-v2/engine_solver.c.fetch.txt`。smooth已经是合并的非约束广义力，不再次扣bias或加actuator/gravity，否则double count。

独立输出：
- mapped=J.T*efc_force；mapping_error=qfrc_constraint-mapped。
- g_direct=M*qacc-qfrc_smooth-qfrc_constraint。
- g_rows=M*qacc-qfrc_smooth-mapped。
- 恒等交叉核对 g_rows-g_direct=mapping_error，不能将此算术自洽当物理真值。
- smooth_error=M*qacc_smooth-qfrc_smooth，用于分离质量矩阵/非约束求解与最终联合残差。
- 每DOF逐项相对分母 max(abs(Ma),abs(smooth),abs(mapped))，分母0标undefined。smooth和mapping分别用自身对应项max，不能以其他方程分母归一化。

真实残差全部REPORT_ONLY，不复用解码1e-10当solver阈值，不预设真实solver一定为0；有限终止、内部增量Ma与外部重算舍入、接触正则化与数值尺度分开解释。此量是primal stationarity观测，不是完整KKT、接触互补、冲量/功平衡或静态站立；qacc可非零，不能称静态力平衡已达成。

## 未来采集（本轮不执行）

只在新协议授权的fresh一次forward后、下一次写状态前同端点copy qacc/qacc_smooth/qfrc_smooth/qfrc_constraint、J/force/type/state/final岛映射、DOF类型、模型选项/solver统计；通过官方mj_fullM接口展开惯性矩阵，记录实际3.9 wrapper signature/layout、输入质量存储形状及所有字段来源。不得假定传统qM树状打包与当前M稀疏布局相同；接口尚未实测，未来先读signature，再按实际接口展开，失败停止不猜。

同时保存MuJoCo mj_mulM(qacc)作为库内对照（若公开接口支持），与独立dense矩阵乘法逐DOF比较；调用只能读数据，前后检查time/qpos/qvel/act/ctrl及已捕获动力字段不变。质量展开不等于新动力求解。模型nv<=256预分配检查、nefc<=4096、numeric最终payload<=64MiB（包括before/after/统计），非RSS保证；finite与完整性校验必须在写档前执行。首次真实采集只计划canonical和单列人工2岛模型各一次forward，不调solver/pose/gain，未来需明确执行许可再运行。

## 纯数组子合同

本轮工具只收dense完整M与J，不解稀疏存储。要求M为finite实对称正定方阵、J finite(nefc,nv)，向量形状和finite正确；对称校验只作矩阵输入工程校验，rtol=1e-12/atol=1e-14，不是solver通过阈值。不自行对称化或正则化。numpy cholesky检查SPD；必须报告结构校验与stationarity验收不同。

DOF类别必须完整：translation/model_force或rotation/model_torque；按类别分别报告每方程absmax/RMS/relative/undefined，无混合物理总范数。模型力/力矩不冒充SI；质量单位/生物绝对标定仍未建立。该label来自未来model joint类型映射，纯数组工具无法证明调用者label真实。输出独立copy不改输入，空约束J(0,nv)可用，nv必须正。

合成测试必须包括非对角SPD、混合单位、非零g、mapping独立错误、零分母、空约束、坏布局/NaN/非SPD/非对称、单位标签错、输出隔离。逐scalar dot与工具矩阵算法独立比较。桌面样例明确SYNTHETIC，不制作伪仿真图。独立审阅后才安排真实采集；不可把本轮工具测试通过计为真实stationarity已测。
