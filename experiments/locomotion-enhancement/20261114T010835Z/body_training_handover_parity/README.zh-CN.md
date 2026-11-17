# 身体—训练源交接保存数据诊断（S2/S5，ONLYCODE）

目录：`/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/body_training_handover_parity`。
本工具不训练、不重放NN、不构造Core或物理，父仅一次NumPy读取已经保存的数据，≤90秒、RSS≤1500MiB。子只读源码/JSON/ZIP NPYheader，不读数组payload、不导入NumPy或Torch。全source冻结后不再修改；若执行有schema问题另新scope恢复，不重跑身体。

目标是区分首NN交接与训练源接口是否一致，不因四case拟合改善推断真实修复，也不盲加epochs或改loss。现有42项freshNN保存审核只证明body日志与公共模型一致；本诊断补充body handover vs训练raw source与已保存新模型conditional首输出的直接比较。

实际模型为 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/target_feedback_order_repair_v4/TRAIN/checkpoint_000600.pt`，SHA `41fa1e2232f5143fec0a3ebb5d506c819af0ba084b3a05041e829fd1477b906b`。身体root仍目录名target_feedback_motor_body_v3，actualCAL为v4。本诊断不加载PT，引用actualbody binding与cal源/数据/证明SHA。

父95尚运行时仅允许子读已完成walk/stop的headers与旧source metadata，不读未完成左右产物。本工具的实际parent入口qualify要求body JOBS完整9stage全exit0、四RESULT及各42保存审核通过、SUMMARY与cal600的proof全部固定；无条件结果缺失不能分析或伪填。行为失败与完整性审核通过分开；parity mismatch是需要保留的诊断结果，不会通过筛例或重试隐藏。

实际训练映射按cal.INPUT_BINDINGS.entries id取walk/stop/left_pivot/right_pivot，不按case顺序猜训练序号。原teacher2880行从1秒开始，warm_start0/start960；fullNN3080行含200settle，warm_start200/start1160。两者首NN物理preaction均5800ms，实际trace[5799]为此前末端，首动作累计5次更新trace[5800..5804]，posttarget取trace[5804]。训练源walk命令由实际data.py重建[1,raw_turn]，可能有旧脑微小turn，不盲设[1,0]；教师命令[0,entry.turn]。

逐case比较：完整sharedwarm trace0..5799的qpos/qvel/target/ctrl/action，首差endpoint毫秒；实际初始机械状态和5.8s preaction；960warm X238与首NN X238全部238列差；actualbody task与训练真实task；若源hidden256保存则核warm beforeafter/firstbefore，教师源未存则NOT_RECORDED，禁止神经补算或假零。body new96在全部warm与首NN before必须实际zero。

新body首mean/action/new96after对比cal EVALUATION_ARRAYS的conditional_mean/action，以及TRAIN/CONDITIONAL_000600.npz中对应id+训练index的mean/action/new96第一行；body pretarget5799/posttarget5804对比conditional pre/post和EVALUATION conditionalpost。原source首教师action或旧NNaction只是训练监督标签，明确不能冒充新NN预测。源label与newaction差值只作描述。

actual normalization使用已绑定normalizer_mapping.qpos_indices/halfspan42/q0_42，对源和body真实pretarget/actualq重算float32再float64的42:84 targeterror与126:168 targetoffset编码并比较；不发明qvel映射。identity仅读取保存compiled_arrays物理字段SHA、actuator/physics/adhesion及共享physics/XML sourceSHA；geom_group/rgba或场景名称等其他差异单列，不自动当物理原因。源完整336脑眼颈神经状态不补造，当前bodycomponent absence为已声明scope。

若command不同，明确标NOT_CONTROLLED_FOR_COMMAND，首输出差只作测量，不判初始化/集成bug，parity_as_interface_test=null。仅warmhistory、初X/task/preaction/已记录old256/new96zero匹配且calibration prime生产proof成立时，才把首输出称为可期待精确的受控对照；教师old256直接记录仍标NOT_RECORDED。首接口一致只支持初始接口一致，不能证明后续闭环/动力学泛化或行为PASS。若不同，先看具体warm/input/task/target/state首差再定位，不能据此判容量不足。旧walk/stop/pivot的slip、速度、漂移、NF、安全门全不修改。

输出RESULT.json逐case详细比较、REPORT.zh-CN.md与VERIFY.json。VERIFY核的是保存源完整性、四case覆盖、资源和0NN/Core/物理，不把parity一致本身当必须通过的门。父实际执行命令（等95全部完成后，既定125712环境，单一managedCPU1）：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/body_training_handover_parity/analyze.py --execute-parent-saved-handover-analysis
```

标准库检查命令：

```bash
/usr/bin/python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/body_training_handover_parity/static_check.py
```

不更新主tracker，父统一记账。完整artificialbrainNN→realDN2queue→NNdecoder2/full432尚未部署，不把组件诊断当成品或生理、电极、意识结论。
