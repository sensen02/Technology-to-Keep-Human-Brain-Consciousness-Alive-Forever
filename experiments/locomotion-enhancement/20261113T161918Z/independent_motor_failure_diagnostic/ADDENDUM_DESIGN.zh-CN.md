# bound6表示下界补充：独立运行，原阶段A源码不改

新增ADDENDUM_PREREG.json与representability_probe.py，原diagnose.py／metadata.py／stats.py／PREREG／CODECHECK及所有科学来源保持冻结。探针只在原阶段A实际STAT/VERIFY完成且四baseline重放精确后运行。

对真实8条训练缓存oldmean和实际目标Y，定义区间lo=tanh(oldmean−6)、hi=tanh(oldmean+6)，最优独立残差动作是把Y夹到[lo,hi]，逐点绝对误差即Y到区间的距离；均方给该约束放松问题的MSE下界。6是新mean残差幅度，不是执行器力矩。因为有限head下6*tanh(head)只位于开区间，使用闭区间给的是保守infimum／闭包下界，不保证有限共享ANN或GRU可达到。

按每序列、全42及每腿7个实际源码执行器通道报告：严格／1e-12容差区间外比例、MSE/RMSE下界、95%和最大绝对误差下界、仅不可达坐标的95%误差、任一坐标区间外的时间比例，并和已有final600训练动作误差作描述性比较。分腿映射通过完整冻结XML中actuator的joint_LF/LH/LM/RF/RH/RM前缀与声明顺序解析；实际四IDENTITY的XML SHA及42原生力矩上限逐项完全相符已在子端仅用标准库文本核对，不凭42维形状猜腿。

父运行时严格装载已审核454581...600，caldata.prepare(m)只复算冻结old789f真实source缓存，不更新新控制器、没有优化器。oldmean／Y还须与原EVALUATION_ARRAYS逐位一致；实际learned动作从已审核该保存评估artifact读取，无额外新控制器forward。补充自行保存representability_cache.npz，避免改阶段A输出协议。

对四条学生实际X，仅用原保存oldmean（mean−residual分解）给动作区间宽度和符号覆盖描述。没有其正确teacher目标，所以不计算新状态真实MSE下界，也不拿旧teacher相同时刻动作伪造标签。

下界大支持“此幅度6在这些训练X/task/history上逐点就无法拟合目标”；下界近零仅排除大的逐点幅度瓶颈，不排除共享参数容量、历史表征、观测歧义、欠拟合或闭环暴露问题，不能单凭此诊断选定根因。未来任何bound／conditioning改动都需独立候选，探针不改输出、不改权重、不物理、不查询教师、不新训练。

父等原阶段A完成，再同一独占CPU环境托管：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python -B /run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/independent_motor_failure_diagnostic/representability_probe.py --execute-parent-representability-probe
```

上限180秒／1800MiB，无重试或延期。实际输出REPRESENTABILITY.json、REPRESENTABILITY.zh-CN.md、REPRESENTABILITY_VERIFY.json和representability_cache.npz。源／权重digest与RNG前后保持，所有旧四组件FAIL与neck尚未部署状态不改变。
