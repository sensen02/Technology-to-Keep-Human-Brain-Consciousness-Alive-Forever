# Local-mean corrective 352纯ANN身体组件（S2/S5，ONLYCODE）

目录：`/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T230738Z/local_mean_motor_body_component`。

组件KIND为 LOCAL_MEAN_CORRECTIVE_MOTOR_COMPONENT_BODY_V1，范围 **BodyANNcomponent_NOTnewbrainneckdeployment**。新的校准尚在父固定600更新执行中，最终SHA未知；本组件行为UNKNOWN。不把221508教师辅助工程目标通过、标签mask合格或离线拟合视作纯神经行为通过。原007d与45458失败和原789f通过模型及全部科学文件保留；本子不改cal输入或主tracker。

实际新校准目录：`/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T230738Z/local_mean_corrective_calibration`。已读common/model/sources/run/train/verify_saved/evaluate/audit_eval实际源码。新FORMAT INDEPENDENT_MOTOR_LOCAL_MEAN_CORRECTIVE_V1，strict common.load_checkpoint(path,digest,final=True)→motor,payload；公共step_details/step/prime_old/cached_features/initial_state和obs238+cmd2、state[128,64,64,96]共352保持一致。模型内部encoder输入412=physical112+oldafter256+task2+同次直接oldmean42，GRU96/head192/8训练张量，old789f冻结。原target-position输出prior通过旧mean真实因果路径进入可训练encoder；这是局部神经输出耦合，未增加教师时钟/理想q_des/root或颈端口。body入口只调公共生产step，不抄370或412数值计算。

单层adapter直接读原 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/independent_motor_body_component` common/preflight/runner/audit/run原始入口AST，不通过195245的adapter再克隆。OLD_SOURCE_SNAPSHOT.json先固定原7core哈希；每次编译核其SHA。AST仅ROOT/CAL/组件FORMAT/旧qualified教师绝对路径/冻结文件清单与motor FORMAT字符串变化。转换前后数值literal multiset完全一致，逆身份替换要求完整AST相等。

用户明确本次纯ANN控制源为3；原161918源标识常量为6。adapter在common原AST编译后将同一namespace SOURCE_CANDIDATE身份元数据覆盖为3，所有函数__globals__保留该同一新字典。loop记录、common.score、独立audit筛选、Trace.summary与结果全用同一source3，避免旧6筛空。此sourceID身份变化单独声明；不是行为数值/门槛变化。warm2、settle0保持。没有教师辅助beta/mix/source7或oldcontrol实验。

四例名称/顺序固定walk[1,0]、stop[0,0]、left_pivot[0,1]、right_pivot[0,-1]。每例1秒settle前200决策不NNstep；4.8秒原教师实际动作，960次prime_old[1,0]仅推进旧256/new96恒0；5.8秒真实状态/target/buffer/previousaction/完整352连续撤辅不reset；9.6秒每5ms一次完整新step_details，actual action=tanh(newmean)，source3。此后没有教师调用。原完整71opaque/free-root42有限PD/native12.5us/noadhesion/无实际root写入或外力保持；unsafe原100ms连续COM<.8或NF>.5W，短窗保留不重试。每例max1232000原生步，总max4928000。

验收门保持旧PREREG原实现：walk原五门(worldXforward≥.05/slip≤.5/support≥.9/COM≥1.2/原非足门)再加六腿原walkingcycles、原生NF0与完整自主窗。stop最大COMxy漂移≤.1体长，过渡.5秒后身体XY速度RMS≤.01体长/秒，绝对yaw≤5°，support≥.9/COM≥1.2/NF0/完整窗，cyclesNA。pivot同向yaw≥45°、最大COMxy漂移≤.5体长、六腿load-unload-clearance-reload、安全完整窗；legacy forward/slip各例另列。原指标函数与独立audit完全复用，不能继承辅助数据采集工程PASS。

prepare仅标准库metadata和整文件hash检查；unknownSHA拒绝，不覆盖已有绑定。必须实际新cal三阶段全部exit0/TRAIN600/PT实际SHA/STAGE_VERIFICATION/EVALUATION_VERIFY/真实arraysSHA/源数据整图核，再独占写新INPUT_BINDINGS。

**实际新cal证明布局与旧fastpath不同**：verify_saved.py将每保存0/300/600的12TRAIN+8DEV合并在 TRAIN/VERIFY_000000.json、VERIFY_000300.json、VERIFY_000600.json，每条有production_cache和412directoldmeanfeature检查，rows.training与labels_available区分、DEV_teacher_labels_available=false。evaluate.py将8DEV嵌入EVALUATION.json['DEV']，不会产独立DEV_VERIFY或DEV_EVALUATION文件。父已确认按真实布局严格核验；新prepare核20行逐id/长度/old256exact/两checks及最终8DEV无标签、12训练validcounts，不伪造不存在文件。binding标proof_layout=combined_VERIFY12TRAIN8DEV_and_EVALUATION_DEV。qualified masked labels只证明训练来源，不证明自主body。

父preflight沿原1Core0积分：严格新final600 NN先load再physical；真实352/finite42输出/原789f digest/8Adam600/71opaque42PD/原bounds/七bodymesh范围；真实local412接口由新model公开step内部处理。strict loader校验payload.identity/frozenold/RNG/config/source及8参数形状；common/model别名使用旧sentinel/finally恢复。无额外数值preflight在本子运行。

之后固定4phys+4freshNN savedsourceaudits+summary共9主阶段，无重试或筛例，preflight另列。原saved NN审核freshstrictcheckpoint后依次重放settle无NN/warmprime/currenttask完整352生产step，mean/residual/hidden/action及保存兼容oldmean=mean-residual逐位一致。该oldmean字段不是声称直接旧mean输出bitexact；direct_oldmean42因果特征在模型内部按生产实现计算，并由新cal每保存20序列直接特征审核证明。不存在抄旧370编码尺寸的body计算。独立原metric/cycle/identity/actuatortorque/实际force/nativecounts审核与原门保持。

本子执行仅标准库AST/JSON/sourcehash/受保护stdlibimports/globals，0数值库/NPZpayload/PTload/NN/Core/积分/训练/渲染。冻结后父使用既定实际环境，准确命令如下。ACTUAL必须替换为父审核完成的新final600实际64hex SHA；未知占位符不执行。

```bash
/usr/bin/python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T230738Z/local_mean_motor_body_component/static_check.py

/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T230738Z/local_mean_motor_body_component/prepare.py --bind-completed-final600 --sha256 ACTUAL

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T230738Z/local_mean_motor_body_component/preflight.py --execute-parent-preflight

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T230738Z/local_mean_motor_body_component/run.py --execute-parent-fourcases
```

末条由父单一managed CPU1运行并collect全部9stage。完整人工脑→真实DN2来源队列→NNdecoder2→full432感觉回归仍须另行实现验证，当前body组件诊断不能作为新脑颈成品或生物/电极结论。
