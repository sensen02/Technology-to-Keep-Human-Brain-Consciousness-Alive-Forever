# 中性walk匹配来源：单case保存数据附录（ONLYCODE）

目录：`/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/body_neutral_walk_handover_appendix`。

仅补原训练index8 old_walk_control，与当前bodywalk同中性命令[1,0]。原四source主报告的walk采用[1,raw_turn]而被标NOT_CONTROLLED_FOR_COMMAND，不等于中性walk没有训练覆盖。本附录不改原主算法、恢复脚本、cal/body、模型或tracker，不做新物理或NN重放。

source准确目录 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T221508Z/teacher_assisted_collection/trials/old_walk_control`。它虽在collection目录却是teacher=false/teacher_assisted=false的纯旧789f正控制，source8、oldcontrol_weight1/student_weight0/teacher_weight0、5.8后教师0，007d完整352只shadow。实际source59项NN保存审全true，原五门及六腿周期PASS/slip.4927888。监督target是直接oldmean.tanh，不能当new41fa模型预测。

只复用冻结 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/body_training_handover_parity/analyze.py` 的原genericcase_report()。appendix显式先加载冻结recovery_v1 metadata普通模块，再sentinel暂借metadata alias导入originalanalyze普通模块，finally恢复alias，不调原main、不clone函数/contextlib。所有数学、容差、索引在原函数原globals中保持。构造source index8/conditionwalk/conditional_tag old_walk_control8/bodypaths actualbodywalk；只调用一次case_report，不复制四casepipeline。

模型预测参考仅actualv4新41fa600的EVALUATION conditionalmean/action[0,8]及CONDITIONAL600 old_walk_control8六输出。source352前256是相同冻结789f旧链，切片作warm/firstbefore对照有实际59审核依据；source后96属于007dshadow，不能与新41faafter96当同模型。原generic函数只比较共同旧256和新600 conditional new96，新bodybefore96实际zero。源和body都warm200:1160、首NN1160/pretrace5799/firstupdates5800..5804/post5804，sourceINITIAL和WARM_BOUNDARIES有40×42buffer和target；不假造缺录字段。

metadata qualifier先复用冻结恢复版的完整95九阶段/四42审核/calnew600/sourcehash证明，再独立核index8真实flags、59审核、artifacts、NPYheaders、条件输出tag和SHA。FIRSTREADY前实际STDqualify到return，禁NumPy/Torch/payload，保存STD_METADATA_QUALIFICATION.json。父后续才NumPy只读保存数组，≤90秒/RSS1500MiB/CPU1，0NN/Core/物理/训练。输出RESULT.json/REPORT.zh-CN.md/VERIFY.json，difference是结果不隐藏或改行为门。

父132资源释放后单次命令：

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/body_neutral_walk_handover_appendix/appendix.py --execute-parent-neutral-walk-appendix
```

标准库静态命令：

```bash
/usr/bin/python3 -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261114T010835Z/body_neutral_walk_handover_appendix/static_check.py
```

匹配只证明初始接口，不证明后续闭环或步态成功。95全部behaviorFAIL保持。scope BodyANNcomponent_NOTnewbrainneckdeployment；完整人工brainNN→realDN2queue→decoder2/full432仍未部署。
