# 最小保存数组积分诊断：ONLYCODE，0NN／0Core／0物理

本工具只做代数执行器目标重放和读取已保存fcbd预测。不会加载权重、调用NN／教师、创建Core／执行mj_forward或积分，不产生新候选／训练预算。全部旧科学源与数据冻结。

A先验门覆盖13条：原8真实监督轨迹、221508 old_walk_control，以及fcbd四项实际身体轨迹。每条从物理5.8s之前的trace.target[5799]开始；取其实际每5ms应用action，rate=clip(action*2,±2)，再按实际max_rate裁，连续五次target=clip(target+.001*rate,真实bounds)。逐1ms比较trace.target[5800:]，最大差≤1e−12且实际action hold逐位一致。对应preaction X target-error42:84与target-position126:168按真实double qpos/target/q0/halfspan、clip[-10,10]再float32→float64重建，要求逐位一致；任一失败先写VERIFY、停止B，不掩误差。

限幅与normalizer从实际元数据取得：普通12来源用IDENTITY.actuator实际bounds/max_rate；冻结完整XML angle=radian、42hinges+一free、keyframe49给q0和范围。body按自有joint再递归children拓扑分配qpos地址，actuator声明指定42顺序。每来源实际IDENTITY.compiled_arrays jnt_range43×2与key_qpos1×49的dtype/shape/SHA必须与该重建完全一致。子端仅用标准库struct打包这些XML标量，已对全部13身份的两项SHA完全核对，无需Core，也没有读取数组值。

原8中的plane_control采用派生地形IDENTITY schema，没有常规actuator数字列表。唯一特例明确标source-derived contract：actualderivedXML SHA与adapter声明核验、terrain_core.py:82-91只改XML loader、不传actuator参数；IncrementalActuator.__init__的max_rate默认2与bounds默认native由冻结源码AST核对，两source SHA必须在该实际plane身份中。其compiled jnt_range/key_qpos仍同上严格实值hash门，输出provenance不冒充原IDENTITY显式字段，不根据感觉／经验峰值猜边界。

仅A全部13通过，B读取已actual审核的fcbd EVALUATION_ARRAYS.new_action[1920,12,42]，按实际bindings顺序选原8+oldcontrol共9条；actual_target必须逐位等于对应真实已应用action。对预测action按同一真实初始target/caps积分，报告有符号动作平均偏差与RMS、target全时RMS／峰值／末值及42通道逐项、rate/angle cap计数。三条assisted真实target积分了.75teacher+.25student，不能当unmixedteacher累计监督，故不进这9条条件性比较。

B的预测来自固定真实源X上的已存NN前向，没有把预测target反馈回NN，也没有改变q／载荷／隐藏历史；只是条件性目标寄存器积分，不是新身体轨迹或停步成功，不能据累计偏差宣布唯一根因。fcbd四真实身体target只在A核其实际动作执行一致性，不拿旧teacher同clock target当新状态纠正目标。

身份范围为直接输入约96项：新代码／预登记、实际PARENT_CALIBRATION_REVIEW、saved0/300/600证明、EVALUATION/58项独立审核、bindings/EVAL数组、各RESULT/VERIFY/IDENTITY/INITIAL与真实trace/samples、完整XML及plane派生合约源码。没有把CURRENT状态或全部3558祖先反复校验塞进60秒范围。数据字节SHA前后核验；报告函数所有递归LOAD_GLOBAL/NAME按实际namespace解析、纯文本干跑通过，不再遗漏json。

父独占CPU后执行一次：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T230738Z/accumulator_replay_diagnostic/diagnose.py --execute-parent-saved-accumulator-diagnostic
```

全流程60秒／512MiB，输出STAT.json、REPORT.zh-CN.md、VERIFY.json。A失败只保存不通过证据、不执行B；资源超限不自动延期。真正target／编码复现结果与条件误差等待父数值执行，本子只JSON／NPY头／源／标准库hash检查，0数组值、0NN、0Core、0物理。
