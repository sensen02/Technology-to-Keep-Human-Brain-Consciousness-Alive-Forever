# S2/P2 同端点软约束残差预注册（仅来源/可用性阶段）

## 授权与身份
本协议在任何新轨迹之前登记。本轮只核官方MuJoCo 3.9.0源码，并用 `/run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python` 调用原入口 `engine.canonical_physics.load_model()`、fresh `MjData`、一次 `mj_forward`。只在初forward之前将历史 `/run/media/sensen/Data2/cell_wound_prototype/outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/initial_pose.npz` 的 `projected_seated_qpos` 赋给qpos；不设新姿态、不积分、不 `mj_step`，不改生产/helper。手动研究不是成品；不证明S2/P2收敛、行为、生物标定或物理原因。未来执行观测仍需另轮明确有界授权。

## 官方确切依据
- [engine_solver.c](https://raw.githubusercontent.com/google-deepmind/mujoco/3.9.0/src/engine/engine_solver.c)：`mj_solPrimal` 明确 `Jaref = J*qacc - aref`；`PrimalUpdateConstraint`调用`mj_constraintUpdate_impl`并映射`qfrc_constraint=J^T*f`；`PrimalUpdateGradient`为`M*qacc-qfrc_smooth-qfrc_constraint`。Gauss cost为 `0.5*(a-a_smooth)^T*M*(a-a_smooth)`（利用`M*a_smooth=qfrc_smooth`）。停止条件为 scaled improvement 或 scaled gradient 小于 opt.tolerance，不是逐行1e-10。
- [engine_core_constraint.c](https://raw.githubusercontent.com/google-deepmind/mujoco/3.9.0/src/engine/engine_core_constraint.c)：`mj_constraintUpdate_impl`先逐行`force[i]=-D[i]*jar[i]`，再按类型/区域覆盖；QUADRATIC保留该式，cost为`0.5*D[i]*jar[i]^2`；非椭圆单边行jar>=0置SATISFIED、force=0，jar<0为QUADRATIC。摩擦损失上下界为±R*floss，线性区force为∓floss；椭圆中区为CONE且另式，禁止套标量式。`mj_makeImpedance`给出D=1/R；`mj_referenceConstraint`给出aref=-B*vel-K*I*(pos-margin)，connect/weld还减Jdot*v。`mj_projectConstraint`只为dual构造AR。
- [engine_forward.c](https://raw.githubusercontent.com/google-deepmind/mujoco/3.9.0/src/engine/engine_forward.c)：`mj_fwdConstraint`给出 **b=J*qacc_smooth-aref**，不是J*qacc-aref；noslip>0最后再dualFinish会改变力/加速度，不能假定仍满足primal本构式。`mj_forward`不调用积分；`mj_step`才调用积分器。
- [mjtype.h](https://raw.githubusercontent.com/google-deepmind/mujoco/3.9.0/include/mujoco/mjtype.h)：solver PGS=0(dual)、CG=1(primal)、NEWTON=2(primal)；QUADRATIC=1；PYRAMIDAL=0；Jac AUTO=2，nv<60使用dense。

## 最小独立计算与命名
对同一次完整forward后的端点，独立逐行dot计算 `r_i=sum_j J_ij*qacc_j-aref_i`。**r不应被要求为零**：它是软约束参考加速度偏差，不是硬不穿透/KKT违规量。
最小有意义本构一致性残差：对state==QUADRATIC的行，`e_i=f_i+D_i*r_i`；同时记录该行type/id/state、D/R、r与f；只在primal CG/NEWTON且noslip_iterations==0时解释为该源码分支一致性。QUADRATIC式是软目标导数，不是完整最优性证明。state非QUADRATIC分别计数，不能隐去或按零残差通过。
当前pyramidal/frictionless/limit单边行可附`state==SATISFIED => f=0,r>=0`、`state==QUADRATIC => r<0,f>=0`；数值边界near-zero单列REPORT_ONLY，避免用同一state反推状态转换。摩擦损失需floss及R；椭圆CONE需整contact块、dim/friction/mu，当前最小方案不验此分支。
另一个纯映射恒等式：`e_map=qfrc_constraint-J^T*f`。若补质量矩阵，梯度/力平衡残差 `g=M*qacc-qfrc_smooth-J^T*f`，同 `M*(qacc-qacc_smooth)-J^T*f`（还需核smooth solve），这是软目标stationarity诊断，非完整KKT。
Dual最优性需 `w=AR*f+b`、AR=J*M^-1*J^T+diag(R)、force admissibility与各类互补/摩擦cone/active set；不得把w无条件要求零，更不能把只核e称完整KKT。当前AR未构造，不切solver获取它，不分配nefc²巨矩阵。b可用但不是最终r。

## dense / sparse / 同端点语义
运行时调用mj_isSparse，不按字段是否存在或AUTO enum猜布局。
- dense：要求len(J)==nefc*nv，按C行主序reshape(nefc,nv)；rowadr/rownnz/colind虽Python有shape，**无语义，不读取为稀疏布局**。
- sparse：要求rownnz、rowadr各nefc；每行0<=adr、0<=nnz、adr+nnz<=len(J)且<=len(colind)，列0<=col<nv，检查重复/排序与nJ、sum nnz。dot使用J[adr:adr+nnz]及相同colind，独立oracle可小规模dense；空约束不造通过率。
端点所有数组须同一次forward后拷贝，无额外forward混采，无RK4内部阶段混用；行号/contact id只在该端点解释，绝不跨clock或时刻匹配物理点。未来若复用原10ms两时钟案例，只补观测、先另轮授权；本轮不执行。

## 预声明指标与容差
本轮字段探针不计算真实动态律残差。未来记录逐行e/r/f、QUADRATIC覆盖数、各state/type直方图、绝对max/RMS、分type及分DOF统计。相对e使用`abs(e)/max(abs(f),abs(D*r))`，分母零标undefined；不混合model force/torque单位求物理总范数，不擅转SI。
**真实 solver e、r、g、active-set/边界全部REPORT_ONLY**。1e-10仅可作显式标注的合成float64算术/布局oracle工程检查，或既有解码工程阈值，不能挪作solver收敛失败阈值。内部Jaref递增更新与独立重算存在舍入，opt.tolerance=1e-8是缩放目标/梯度停止条件，不能推出e<=1e-10。之后若要用途相关数值验收，先另行定单位/尺度/界而非看结果调整。

## 未来最小归档字段、界与失败政策
基础：time、qpos/qvel/act/ctrl、qacc/qacc_smooth、qfrc_smooth/qfrc_constraint、qacc_warmstart、warning与solver_niter/stats、ne/nf/nl/nefc/nJ/ncon、所有实际solver/cone/Jac/flags/noslip/iteration/line-search/tolerance/timestep/impratio配置、source/environment身份。
行：J（dense原flat或sparse值+rownnz/rowadr/colind）、aref、D、R、force、state、type、id、b、vel、pos、margin、KBIP、frictionloss。contact：geom、pos/dist/frame、dim/exclude/efc_address/friction、local6，**solref/solreffriction/solimp/includemargin/mu**；用于软约束参数、参考安全clamp及机制解释，不能仅凭f,r归因。compiled geom/pair solref/solimp/friction/margin/gap/priority/solmix与override选项也须存。力平衡若启用另存实际M压缩值及模型M布局/DOF单位，不能凭缺字段猜。
未来每端点硬界ncon<=512、nefc<=4096、nv<=256、J元素<=1048576；两case各400端点，逐case写盘，不常驻多巨矩阵、不存AR/Hessian。numeric ragged offsets + allow_pickle=False，局部rowadr在端点内；字节预算累计<=256MiB/案例，超界明确停止/失败不截断。shape/索引/非finite/source hash变化/warning均显式失败，无可用AR标not_constructed而非零；不满足分支标unsupported而非成功。
编译身份前后SHA覆盖原canonical XML、resolved mesh/texture/include输入、canonical loader及依赖loader、历史pose、诊断脚本；不得改历史hash迎合现在。当前XML无file texture/include，71mesh resolved在本地cache，无下载。报告不存巨矩阵。

## 本轮新增有界执行授权（用户继续指示）
仅NEW runner/tests/offline verifier：nativefloor pulse target0固定两case，共同历史projected pose、10ms、RK4 dt25/12.5us、共同25us400端点。沿用原runner的每步mj_step后mj_forward，所有端点字段同一次forward后拷贝；不改生产/tau/pose/solver。NEWTON primal、noslip0强制。复用既有availability探针的本地mesh解析/77来源集合但不执行探针；额外归档新诊断源码hash。逐case预算包含数值数组与offset，<=256MiB；compiled/contact参数支持解释，不构造AR/M、不声称stationarity/完整KKT/mass balance/因果。真实残差全部REPORT_ONLY。dense稀疏metadata存零占位并明确无语义，真实sparse只读有效row范围/局部列映射，支持padding。独立冻结归档验证用重建矩阵@而非逐scalar循环，检验归档与源码hash不变。

## 本轮交付与停止点
桌面机器可用性 `/home/sensen/Desktop/engine修复验收/S2-dynamic-availability-20261014/availability.json`；一次性可审查探针 `/home/sensen/Desktop/engine修复验收/S2-dynamic-availability-20261014/availability_probe.py`；中文 `/home/sensen/Desktop/engine修复验收/S2-dynamic-availability-20261014/来源与字段可用性报告.zh-CN.md`。协议/报告须父审阅后再决定下一有界动作；无自动轨迹续跑。
