# S2 实际MuJoCo RK4派生诊断（执行前声明）

固定执行器0/同已验投影就座pose，初qvel0；pulse初act0、ctrl0.1持续1ms后0至10ms；decay仅目标初act0.1、ctrl0至10ms，其余act0。两floor分支独立编译，RK4派生integrator枚举1，步长100/50/25us，共12cases。不是生产默认改变，不在积分中覆盖姿态/act，不刺激搜索。

每步前按整数索引设置固定控制，不跨事件混用；step后forward记录目标act/force、全qpos/qvel、ncon、最低contact.dist（无接触null报告，NPZ用单独mask+0）、全actuator force及moment、qfrc_actuator、警告。完整每case端点存NPZ，瞬时力分解rtol/atol1e-10工程身份核，不算生物力。

标量DOP853既有reference比较target act，<=0.001仅沿用人工标量工程目标，不证明全身体收敛。比较同case共同端点qpos/mixed unit最大分量差及接触数，最后细网格非真值，不设站立/力阈值。来源before/afterSHA含canonical/pose/helper/reference/runner/protocol，不假称全mesh快照。命令退出0只代表运行完成，数值筛查FAIL必须保留。实际看图/独立npz验核，不根据表现调tau/pose或切生产。
