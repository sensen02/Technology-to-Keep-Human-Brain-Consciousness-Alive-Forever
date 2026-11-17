# 旧336长时匹配平地对照（S2/S5，ONLYCODE）

独立目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/long_flat_matched_control/`。匹配来源 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/rough_terrain_long_window_38p4/` 冻结源码与XML，唯一场景变量为删除六个静态terrain_block_00..05节点。XML_DIFF_PROOF记录每个删除节点、原位置与完整属性，逆插入精确恢复原XML树；保留terrain_plane、棋盘材质和所有其他节点、属性、资产绝对路径。XML序列化格式可能不同，科学节点不变。旧源与结果只hash引用，不覆盖、不重跑。

旧motor789f+视觉/嗅觉/人工脑+来源DN2/AN1完整336不变；不使用当前new600motor352或当前训练结果。原1s稳定+4.8s预热+38.4s自主，总44.2s/3536000native/3072000post/8840决策/2210真实RGB及44200ms排他相机上界不变。完整71opaque/42PD、质量、摩擦、力限、无adhesion/额外根力、noreset和所有数值门不变。观察版本仍为TERRAIN_RELATIVE_PAD_CLEARANCE_238_V1，真实ray/contact因平地环境而改变。

动机：旧粗地形长窗末段趋静，而短9.6s平地通过无法证明38.4s长期稳定。若平地也衰减，支持一般旧336闭环长期问题；若平地仍持续推进，支持场景关联。眼图从初始就可能不同，此对照不能分离视觉与机械因果，也不强求脑状态beforecontact逐位一致。教师预热物理是否相同待实际比较，不预写PASS。

ONLYCODE命令（标准库、源码、JSON/XML；不加载数组/科学包/Core/NN/物理/渲染）：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/long_flat_matched_control/sourcecheck.py
PYTHONDONTWRITEBYTECODE=1 python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/long_flat_matched_control/preflight.py
```

FIRST READY冻结后命令只验FROZEN。CODECHECK是零Core源检查，不能冒充实际compiled parity。实际父任务须等待当前95 new600motor四body+四NN审计全部结束并释放资源，父再决定是否执行以下guard。guard允许科学导入、仅构造一个新平地Core，并直接编译旧rough MjModel（不构造旧Core/不重跑旧physics），0积分/0NN/0数组payload；按名字映射比较common body/joint/dof/actuator/mesh/geom物理与extrinsic，排除确实删除的6geom和必要地址布局。几何96→90、nq49/nv48/nu42及71opaque/18foot/plane registry需实际通过。失败保存真实报告，不能绕过。

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/long_flat_matched_control/parent_zero_step_guard.py --parent-resource-released
```

父guard通过且决定运行后，仅一次managed background任务执行以下orchestrate，串行one trial+fresh336 audit+SUMMARY；最大3536000native，无自动retry，不重跑旧rough19/38、gray或plane9.6，不占用当前95重资源：

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/long_flat_matched_control/orchestrate.py --parent-resource-released
```

对照case=`plane_long38p4_matched_control`，kind=`SOURCE_NECK_LONG_FLAT_MATCHED_CONTROL_V1`。结果预期 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/long_flat_matched_control/trials/plane_long38p4_matched_control/RESULT.json`，同目录VERIFY.json。汇总 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/long_flat_matched_control/SUMMARY.json`。四个预登记9.6s窗报告root/COM位移、动作RMS、1ms target变化、motor hidden变化、原strict cycle完成事件与最后脑命令；末9.6s单列。事件按完成时刻落窗，可能起于前窗，不假装每窗独立重置审核。不引入新‘停止’阈值，父据完整记录比较。

roughpass/all_pass及障碍coverage在本平地对照为null/NOT_APPLICABLE_FLAT_CONTROL；原五门（含slip≤.5）及原行为/六腿门保留参考，不把无障碍接触伪写NOTTESTED或FAIL，不把工程诊断门强制变为所有用户目标的前置。functional_stable_body不等于完整神经颈部或生物有效性。未授权或执行render，本scope不提供有障碍假设的旧render入口。父编译guard、trial、审计与所有新行为结果目前均NOT_RUN，用户未要求本子占重资源。
