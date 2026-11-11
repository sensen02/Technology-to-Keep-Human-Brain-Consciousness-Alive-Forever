# 续轮3最小动作（备忘，不是新试验预登记）

先读本目录CURRENT_STATE.json和evaluation_recovery/FINAL_SUMMARY.json实际结果。goal revision3/active/armed，续轮2全部作业结束收集，无后台训练。原行走基线、右转通过模型和全部失败保持；左转没有通过，食物行为/神经减速仍未实现。

## 本轮已排除的继续方式

- 同一positive神经读出从2000精确续10000，拟合8.72e-5→8.24e-5，实际leftslip.514841585→.515626812，无达标或继续收益证据。不要无界追加同一head步数。
- 原两轨迹慢残差1500右转PASS.495011，leftFAIL.538039；增加纠正数据的同预算1500left.54496/right.56329更差。不能直接把失败轨迹当passingdemo或反复加样本不检验实际作用。
- 新readout在负/零命令严格零、原parent194878冻结，实际中性/右完整身体和神经数组精确保留。左变化受限于固定64维旧残差hidden的线性读出与bound.25；不能仅看离线MSE推断它足够。

## 建议下一次具体受控候选

在新的时间戳目录先做只读诊断：以实际左转failedrun为准，计算RH/RM负载加权滑移分子、地面速度、动作误差时段以及与教师的状态/命令差异。全窗口和原分母保持，不删失败段。确定原64维parenthidden的读出是否表达不足、残差幅度是否饱和、错误集中在承重还是卸载转换；这些都只能给候选依据，不能断言单因果。

一个小而快的候选是将额外正向神经读出从Linear64→42换成（冻结旧线性输出＋可训练小MLP残差），新MLP可读取实际允许的112物理＋旧hidden64/128/命令，保持relu(actualturn)门，因此negative/zero始终不变。这样仍保留已通过右转，试验只修左转响应。若采用必须新PREREG：明确定义父检查点、额外参数/范围、唯一固定预算和最终物理检查点、原source/data/优化器政策；不要同时改门槛、身体或部署感觉。可在相同range下比较新输入/非线性，区分容量和范围，但不强制做庞大多模型搜索。新支路初始化0使原函数精确，临时unitstep后验证neutral/right全部均值/动作/192hidden不变。

优先父模型应在试验前命名，建议既有head2000（实际left.514841585略优于head10000.515626812），但这属于新的候选设计而非改写上轮final10000失败。可选择固定parent模型后冻结其全部参数，只训练新MLP，使用原合格leftteacher的真实238观测/命令/标签、全连续父记忆cache，fresh新optimizer（不冒称旧Adam接续）。实际包括已passedneutral/right和left全部回合，full1s+4.8+9.6/all5/all6/NF0/yaw.01不变。只有最终固定检查点测试，若仍fail保留而不增幅/多点挑选。

## 重要接口与路径

- qualifiedrightfrozenmotor1500：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/train_two_only_control/TRAIN/checkpoint_001500.pt`，SHAc6b395f29676ef933e2f55ca5a92b7c5b55183a2d432226487b39e916b51acdb。
- head2000：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T010218Z/train/TRAIN/checkpoint_002000.pt`，SHAeeced49c5bdd50a7e5bce2cfd63d56d9929b3bf2d2cb00c1985019d91321e6df；train/model_io.py ReadoutMotor load_checked返回tuple128+64；元数据路径错误由PARENT_METADATA_CORRECTION.json外部说明，实际identity与全部194878父权重正确。
- head10000：本目录train/TRAIN/checkpoint_010000.pt，SHA3bac5af7aa5058a617383eb9deef4a848778d04ae47f66600ddd53788703344e；train/model_io.py同原ReadoutMotor对象，新config准确区分immediatehead2000和frozenmotor1500。
- actualfinal10000：evaluation_recovery/EVAL_neutral,left,right/{RESULT,trace,neural_decisions,geometry,IDENTITY}，VERIFY_*.json；原evaluation/仅是Core前缺ROOTPREREG导致的入口失败，不能错当模型failedphysicaltrajectory。
- actualhead2000：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T010218Z/evaluation/`。
- leftteacher合格数据：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/teacher/left_full/SAMPLES_5MS.npz`，先验RESULT/IDENTITY/样本hash/all5/all6不漂移。

所有部署ANN只读真实感觉和神经命令；禁止phase/time/teacher/食品坐标进入actor，实际腿42有限PD/完整71mesh/native12.5us/无附着外力根脚本。神经减速尚未训练成功，食品behavior的速度通道必须锁定未训练范围；实现双向转向后再做命令变化/新初态/原生记录与食品留出/切断重放对照，才能宣称第一轮趋近完成。goal不可提前complete。
