# Target-feedback顺序修复模型：纯NN身体组件 ONLYCODE（S2/S5）

新目录：`/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_motor_body_component`。

范围 **BodyANNcomponent_NOTnewbrainneckdeployment**。KIND为TARGET_FEEDBACK_ORDER_REPAIR_MOTOR_COMPONENT_BODY_V1。它是身体组件测试工具，不是人工脑—真实颈部成品，不把拟合/存储/条件重放通过当行为通过。新final600实际SHA未知，未绑定或实际运行。所有旧科学源、数据和失败保留，本子不改90修复目录或主tracker。

CAL明确为 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_order_repair_v2`，不能指向失败的env_recovery_v1。已实际读该目录common/model/sources/prepare/train/run/preflight/adapter，以及adapter引用的原target_feedback_calibration verify_saved/evaluate/model。公开MODELFORMAT仍INDEPENDENT_MOTOR_TARGET_FEEDBACK_V1；state[128,64,64,96]=352、obs238+cmd2、412encoder/GRU96/head192/8新张量/old789f冻结、step_details/prime_old/cache接口不变。模型production操作完整继承原412actor；这次修复调整训练scan的冻结旧神经分量运算次序，不能误称新增actor控制律。

实际父只读diagnostic证明两个失败序列原冻结三分量+原运算次序修复后六输出exact0；原save300失败是条件浮点次序资格失败，不据此宣称body policy已坏。原envscope300实际SHA b562fcc3e39c2aa56381a87b17f1e80124f203fa1a73e3cdf8888778de119f14，作为resume父，不作为已通过最终模型。新scope恢复原8Adam计数/矩/RNG及originalinitialanchor，300→600，只新增300更新，保存300/450/600。原0PASS祖先与原failed300VERIFY false均保留，不能把旧失败改PASS、要求不存在新0或冒称新训600。

单层adapter直接读取 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/independent_motor_body_component` 原始common/preflight/runner/audit/run AST，不经230738 adapter再嵌套。固定原7core SHA先写OLD_SOURCE_SNAPSHOT，转换仅ROOT/CAL/组件FORMAT/新MOTORFORMAT字符串/旧合格教师绝对路径/冻结FILES身份。原数值literal multiset完全不变，逆身份替换后完整AST相等。source3为任务明确候选身份，在common原AST编译后覆盖同一namespace SOURCE_CANDIDATE=3，函数全都用该新globals，loop/common.score/独立audit/Trace.summary筛选一致，warm2/settle0。sourceID变化单独声明，不伪称沿旧source6。

四例固定walk[1,0]、stop[0,0]、left_pivot[0,1]、right_pivot[0,-1]。每例1秒settle200决策无NNstep；4.8秒旧教师实际动作，960次prime_old[1,0]仅旧256/new96h0；5.8秒真实状态/目标/缓冲/previousaction/352连续撤辅不reset；9.6秒完整newmodel公共step_details输出tanh(mean)直接有限PD，source3，之后教师调用0。没有辅助beta/mix/source7或teacher clock/ideal q_des/root输入。

完整71不透明mesh/free-root/42有限PD/native12.5us/noadhesion/no额外力/noactualrootwrite。unsafe仍跨warm/candidate连续100ms COM<.8或NF>.5W，短窗与异常证据保留，不重试或筛例。每例max1232000原生步，四例4928000。preflight单Core0积分另列；主编排固定4phys+4freshNNsavedsourceaudit+summary共9阶段，无增加物理。

门保持原实现与原研究reference：walk原5门含worldXforward≥.05/slip≤.5，再六腿原walkingcycles、1ms/nativeNF0及完整窗；stop COMxy最大漂移≤.1体长、过渡.5秒后身体XY速度RMS≤.01体长/秒、absYaw≤5deg、support≥.9/COM≥1.2/NF0/完整窗，周期NA；pivot同向yaw≥45deg、maxCOMxyDrift≤.5体长、六腿load-unload-clearance-reload、安全完整窗。旧forward/slip各case单列；用户此前滑移可松建议在本固定研究门不暗改，未来若重订reference另立。

新prepare仅std读取实际JSON和整文件SHA，不读PT/NPZ payload。须新cal三mainstage实际exit0、TRAIN updates=total600/new300/already300、IDENTITY resume_step300/parent b562/originalanchor+AdamRNGrestored，INPUT_BINDINGS.resume_lineage精确300→600与SAVE_STEPS[300450600]，新preflight READY_RESUME300_TO600/source/data同一。严格checkpoint final600在实际数值加载时验证8Adam计数600/矩/RNG/config/原anchor及旧789f身份；metadata层保留freshVERIFY canonical8Adamstates。

真实verify_saved由新adapter单层搬原entry，每300/450/600写VERIFY_{step}.json：20actualX行（12训练+8DEV）及训练12 conditional_fullproduction/finite检查，保存CONDITIONAL_{step}.npz条件六输出artifact。prepare逐manifest核actualX IDs/长度/old256exact/对应checks和conditional12errors≤1e−12，数组文件SHA全部绑定。evaluate经新adapter将保存审核门从0/300/600改为300/450/600；输出EVALUATION conditional_rows12、actualX_rows12、DEV8，独立EVALUATION_VERIFY和STAGE。不得要求不存在独立DEVVERIFY/DEV_EVALUATION、新0或原failed300变true。所有源、数据、proof与artifact前后hash再核后exclusive写新INPUT_BINDINGS，未知SHA拒绝，已绑定不覆盖。

预检沿旧先strictnew600 NN后physical，1Core0step/352state/四command finite42/8Adam600/old789f digest/71opaque42finitePD/原targetbounds。load_motor沿旧common/model sentinel finally restore；新IO内部恢复model/common/sources/adapter原别名。fresh身体审核只在真实保存X上公共production重放settle/warm/candidate完整352，mean/residual/compatoldmean=mean−residual/hidden/action精确；兼容oldmean并非直接旧均值bitexact声明。独立原metrics/cycles/nativecounts/identity/torque/source审核不放宽。trainingconditional scan不在body路径调用。

本子只做标准库AST/JSON/sourcehash/guardedimports/functionglobals，无Torch/NumPy/NPZpayload/PTload/model/NN/Core/积分/训练/渲染。代码freeze前读取真实新修复完整source规范；actual最终资格尚无时不预测结果。所有执行留父管理，使用既定125712环境。

准确命令（ACTUAL必须替换父新scope final600实际审核SHA，不能执行占位符）：

```bash
/usr/bin/python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_motor_body_component/static_check.py

/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_motor_body_component/prepare.py --bind-completed-final600 --sha256 ACTUAL

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_motor_body_component/preflight.py --execute-parent-preflight

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_motor_body_component/run.py --execute-parent-fourcases
```

父末条单一managed CPU1运行并collect全部9stage；若保存/审核schema错误仅独立新recovery读已存结果，不重物理。完整artificialbrainNN→realDN2queue→NNdecoder2/full432感觉回归仍未部署且不可跳过；身体组件非成品，未知生理/意识/电极不作成功声称。
