# S2/P2 接触状态标签离线语义审计

范围：只读 `/home/sensen/Desktop/engine修复验收/S2-dynamic-residual-20261014T211552Z-7fbf1130` 两个冻结NPZ及manifest，不加载模型、不forward/step、不改参数或旧证据。

官方固定版本 MuJoCo 3.9.0：
- https://raw.githubusercontent.com/google-deepmind/mujoco/3.9.0/src/engine/engine_core_constraint.c
- https://raw.githubusercontent.com/google-deepmind/mujoco/3.9.0/src/engine/engine_solver.c
- https://raw.githubusercontent.com/google-deepmind/mujoco/3.9.0/src/engine/engine_forward.c

本轮探索性初查已经发现每case 424个全局state1/r>=0/f=0行，不能把后续核查称为盲预注册验证。

核查方法：限定实际dense、NEWTON2、noslip0、pyramidal0；每端点独立重建r=J*qacc-aref。逐type6行按官方非负约束分段公式f_expected=max(0,-D*r)，状态按r<0为QUADRATIC1否则SATISFIED0。报告分段force差、保存state与推导state交叉表、带符号分类、全部不一致行CSV、最大旧e行的端点内contact/geom/edge关联。行不跨端点/clock匹配；r=0归SATISFIED。force误差仅REPORT_ONLY，无真实solver PASS阈值。D使用保存的最终efc_D，不重新发明阻抗。

对照：保存全局state1下旧e、按端点r<0分类后的e、全部非负接触分段误差分别报告，不能悄悄覆盖旧统计。R*D一致性仅算术观测。用独立逐标量算法复算，冻结包所有文件及82来源哈希前后核验，新输出放桌面独立目录。单元测试覆盖边界0、符号、陈旧标签、非法布局/非有限数值；不导入MuJoCo。保存源码摘录/哈希与引用，注明官方版本源码不是所装wheel二进制的构建证明。

源码语义需要分别审查warmstart全局更新、PrimalPointers岛内state、迭代更新、forward回写清单；若原包未保存nisland/map/iefc_state，不能声称直接观察到了岛内状态。安全限值与pyramidal R改写单列，不把状态标签错误解释成接触物理原因。新图查看，独立核查后更新tracker。手动研究原型，不是成品；无KKT/质量矩阵、收敛、生物或站立行走验收。
