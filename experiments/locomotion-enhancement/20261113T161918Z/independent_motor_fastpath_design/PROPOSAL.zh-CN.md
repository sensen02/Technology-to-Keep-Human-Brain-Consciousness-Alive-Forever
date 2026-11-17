# Fast NN path：仅形状／迁移设计，未实施候选或训练

本文件独立于已经执行的阶段A诊断与已冻结校准／组件目录；不加入其运行manifest、不修改任何原源码／权重／数据。本轮没有Torch／数组／模型／物理／新训练，也没有指定候选更新预算。路线须等bound6实际表示下界和父对首步证据裁定后选择。

## 为什么值得检查直接神经路径

父175阶段A已实际完成：四个完整352维神经基线逐位重放，源／权重／RNG不变，0物理；三个教师任务首个X／command／oldafter256与训练来源一致，new96输入从零起。父报告首动作相对实际teacher的RMSE：stop约.03661、左pivot约.19974（最大差约.73199）、右pivot约.06891；左首步动作相对warm末变化约.20244。后续域阈值时间并不自动解释首步已存在的误差。这里引用父的实际阶段A结果，准确数值权威仍在 `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/independent_motor_failure_diagnostic/STAT.json`，本文件不重新计算数组。

现source `/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/independent_motor_calibration/model.py:145-154` 是features370→encoder96/tanh→GRUCell96→100ms leaky状态h→head42。alpha=−expm1(−.005/.1)≈.0487705755，因此从h0=0的首决策状态增量受这个泄漏因子限制；但已训练head.bias本来可以非零，所以这不等于“首步必无输出”或“迟缓已证明唯一根因”。只有固定观察／目标误差与表示下界证据能帮助选择路线。

## 建议形状

保持输入features的原顺序：112实际physical、old789f本次afterstate256、独立任务2，共370。仍先算：

```text
encoded = tanh(encoder(features370))               # B×96
candidate = GRUCell(encoded, old_new_h96)           # B×96
h96 = (1-alpha)*old_new_h96 + alpha*candidate       # B×96
residual = 6*tanh(head(concat[h96, encoded]))       # B×42
newmean = frozen_oldmean + residual
action = tanh(newmean)
```

唯一新结构是head输入由96变192；current物理／旧隐藏／任务经学习encoder可以直接到head，不必全部经过new96的100ms状态滤波。它仍是ANN路径，不是外置CPG／规则门／速度乘零／case gate。没有phase、clock、teacher swingleg或世界坐标进入actor。

状态仍为[128,64,64,96]=352；encoded是本次前馈激活，不是新递归记忆。未来与现感觉／脑组合的432维身份仍只是假定接口设计，尚未部署。encoder2+GRU4+head2仍是8个训练张量，但head.weight从42×96增为42×192，**增加4032个参数**，不能说参数量／容量完全不变。

保留alpha、bound6、old789f全部冻结张量及旧投影[relu(turn)/3,turn]、身体71mesh/42有限PD与命令API。公共生产接口继续 `m.step(obs238, command2, state4)→mean42,state4`。新架构须新format、严格loader和实际保存资格，不能让旧600checkpoint因同state维数而被旧loader误认成同模型。

## 从实际454581…600的无表现初始化

若父裁定实施，迁移只按以下明示对应，不猜数值：

- old789f全部参数、buffer、source identity与原256状态定义逐位复制并保持冻结。
- encoder370→96与GRU96所有参数完整复制；alpha、残差bound6不变。
- 新head.weight[:, :96]复制旧head.weight；[:,96:]全部置0；新head.bias复制旧bias。

在完全相同观察序列、命令、960步oldwarm和newh0下，encoded与两级递归状态应逐位一致；head输出在数学上等价于原head。因为GEMM列数/求和核改变，即使后96列为0，也不能无证据要求mean/action逐位相同。迁移前检查预登记可采用mean/residual/action最大差≤1e−12，old256与new96状态逐位相同；若超出先定位实现，不带着差异训练。

这一检查要覆盖真实8训练序列、四已记录身体观察及首步／完整1920历史，显式匹配publicstep和缓存scan；检查源码/权重、dtype、source绑定、optimizer尚未启动。固定观察初始化等价不等于新物理轨迹永远逐位一致，更不能被当成新运动能力。

因为新增直接路径初权重为0，迁移时它不改变原输出；其梯度可在首步直接依赖encoded任务/感觉，使后续学习具备当前特征到动作的快通路。是否实际改善teacher目标拟合、减少首步跳变或闭环稳定，仍要有限训练和独立身体条件实证，不能由结构图推出。

## 可选损失联动：必须明确并非单因素机制证明

若父选择这一个候选，可另立预登记考虑首.1s交接样本的额外离线监督权重，以及把5个v=1来源的残差保持先验从.01提高到.1。首.1s掩码只在训练损失中使用真实序列相对索引，不能作为部署NN输入或运行门控。权重应归一化并保持stop／pivot完整1920历史监督；不能只挑首步而放弃支撑与周期。

Fast head结构、首.1s监督权重与v1保持先验若同时改变，就属于明示的联合修复，不可据改善宣称已经隔离“只有100ms滤波”或“只有walk权重”的因果。更强v1正则可能保护原行走，也可能妨碍stop/pivot需要的抵消；应据同次各任务指标判断，不能靠低平均MSE覆盖权衡。

## 先等幅度表示下界

新增直接路径不扩大6*tanh(head)的残差范围。若真实teacher轨迹上的代数区间下界已大，该结构仍受同一幅度限制，不能期待fast path绕过数学不可达区间。若下界接近0，仅说明没有大的逐点幅度瓶颈，不能排除共享参数容量、观察别名、历史或优化问题。

因此先读取独立 `representability_probe.py` 的实际结果，再决定是实施这一个有限ANN候选，还是先授权阶段B：在学生实际状态上查询已资格教师、验证IK与支撑可行性后聚合纠正标签。阶段B不等于使用旧teacher同时间动作；本文件不实现教师查询或DAgger。

## 验证边界

所有旧完整行走通过、旧teacher成功、首版新motor四项失败、粗地形slip1.339失败及源SHA均保留。Fast path如实施也首先属于身体组件研究，必须独立保存重放与四项身体门，随后才可做脑NN→真实ID DN2延迟队列→ANN decoder2接入和完整感觉回归。没有经颈部署前，不称新三部分模型已学会停止／原地转向。

本提案没有创建新模型代码、没有更新次数或训练预算、没有改变现有运行输入。下一步由父在实际表示下界与首步证据后裁定。
