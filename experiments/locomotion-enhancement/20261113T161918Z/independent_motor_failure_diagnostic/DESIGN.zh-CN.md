# 阶段A：保存数组＋固定观察神经诊断（ONLYCODE）

只诊断已经完成的独立352维身体motor四个失败回合，不创建新候选、不训练、不运行物理、不查询教师动态。所有校准／组件／教师科学文件保持冻结。实际权重固定SHA454581a41034f92e253a2cb31215b5ddd1e09f8795ae54fb89053fd3187d1398。

已只读核对四RESULT/VERIFY与全部JSON身份、ZIP NPY字段头：四回合各neural3080×352、X3080×238、trace15400×49／targets15400×42、geometry14400。4×1,232,000原生步全部已结束，9阶段exit0，完整性审核true，但四项task_pass_candidate都false。仅统计程序子代理没有导入NumPy/Torch/MuJoCo或读取数组值。

程序先严格读取actual600及全部完成证据，父显式flag后才载入Torch。校准data.prepare(m)已经源码确认只计算冻结old789f缓存，没有要求训练后新head仍为零。临时sources别名绑定校准目录后finally恢复，避免把caldata路径误指诊断目录；缓存仅oldmean/after256/真实physical112/task2，目标标签不进actor。无需任何Core/教师FK/优化器。

必须首先对四case全部3080决策完整baseline重放逐字段精确一致：200settle不step，960warm每tick primeold/new96zero，1920自主currenttask，核mean/rawmean/oldmean=mean-residual、residual、352beforeafter、policy_action。任何一项不匹配就保存VERIFY失败、停止所有训练拟合／域／干预分析。旧actual producer的额外actuator/源标签完整性已有独立审核，本阶段不重物理。

每case固定四窗口ANN0–.05、.05–.5、.5–2、2–9.6秒。raw_stats量化实际oldmean、新残差、均值／action、352状态、目标／ctrl、实际足载／速度／COM／1ms非足接触。执行器触边/触cap使用实际IDENTITY边界、力矩及速率参数推导，字段不称“已记录clipping”。qvel48没有明确执行器DOF映射时仅全局统计，不臆把任意channel当某腿；六足记录有明确LEG_ORDER，42动作没有另造腿映射。native峰只接受原每子步摘要，无法给最早native峰时间；事件时间只来自真实1ms记录。

匹配前缀分开机械状态与神经预热：stop/L/R取各自已通过完整工程教师的trace前5800ms，walk取实际plane神经行走；共同qpos/qvel/ctrl/target/action等字段逐项精确性、最大差、首次差时间完整保留，不预设一定相同。NNwarm X[200:1160]与teacher X[:960]或NN X[200:1160]对齐。教师没有old256归档，不制造其神经状态；HANDOVER仅比较明确可用hash字段交集，buffer／target schema不明则UNKNOWN。

训练域为8条原实际训练缓存的逐feature min/max/mean/std，分别报告physical112、oldafterhidden256、task2。std0精确列单列，z只在非恒定正std列计算，不用epsilon伪造大z；new96 |h|>.99饱和比例单独记。省略全量nearest-neighbor矩阵。首次domain阈值固定为组超范围比例>.05或max|z|>5，新96饱和比例>.01；首次动作变化用max|Δaction|>.1。与1ms NF>0、COM<1.2、足总载<.9W／逐腿<.05W的时序并列，不以先后或域外相关宣称唯一因果。

四条固定观察神经流：

1. baseline：实际保存task与new96连续历史。
2. residual_zero：沿相同old256和task，使用保存oldmean的tanh；new96历史保持baseline，仅输出残差移除。保存oldmean本来是原mean-residual定义，不冒称另外求解新身体。
3. new96_frozen_zero：每次决策将输入new96清零、旧256继续正常，允许该次GRU产生afternewh并记录；下次又从new96输入0。不是“afterstate全零”，不改真实状态。
4. speed_flip：v→1−v，turn完全不变，同X／旧256预热，new96连续；旧投影只用turn，故old基底状态原则上不受速度翻转，程序报告其差而不推测实际yaw。

所有干预只量化动作／神经状态范数，不生成物理轨迹、成功预测或合格新权重。已训练head的零h96读出6*tanh(head.bias)也只作静态readout诊断；h96初值零并不保证学习后输出残差首步为零。

训练源误差与学生新轨迹动作差分开报告。同clock的teacher源动作不是学生实际状态上的teacher query，不能当纠正标签；阶段B需要另外授权完整qpos/qvel/执行器历史恢复和FK/支撑可行性检查。本阶段不做阶段B或DAgger。

父在相关视频证据完成并独占CPU后，仅运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/independent_motor_failure_diagnostic/diagnose.py --execute-parent-fixed-observation-diagnostic
```

无需绑定新权重，不修改科学数据。300秒墙钟／1800MiB RSS上限，按128tick检查，无重试／延期。输出STAT.json、REPORT.zh-CN.md、VERIFY.json；source/hash/全模型权重摘要和Python/NumPy/Torch RNG前后核验。STAT保留逐窗口描述与全部干预，中文报告只呈关键证据和界限，不把完整性PASS写成behaviorPASS。

本目录metadata.py与stats.py顶层仅标准库；diagnose.py也仅在父flag通过后数值执行。代码和接口已ONLYCODE审核，真正神经重放／统计与资源是否达标仍需父实际运行。本阶段仍是身体组件，不是完整脑—真实颈部—其他新任务部署。
