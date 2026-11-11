# 续轮4最小动作备忘（不是新实验预登记）

先读CURRENT_STATE.json、FINAL_SUMMARY.json及FINAL_REPORT.md。goal保持active/armed revision4，round3结束时所有作业收集、无后台物理/优化。最新四RH神经增益候选全部滑移FAIL，selected=null；不要增加第五个alpha或复用它们宣称双向资格。

## 已有可用基线与失败边界

- 真正已验收的是原直行、以及慢残差原2-only1500的固定右转工程回合。相同right通过模型的neutral完整轨迹保持原通过基线。固定rightslip=.49501104818453806，不代表newinitialstate/command-switch/fullnative/foodqualified。
- head2000（eeced）在上述parent上只加正向神经读出，right/neutral实际精确保留，leftslip=.514841585FAIL。固定续10000（3bac）left=.515626812FAIL、拟合略改善但物理未改善。
- 更丰富的正向MLP306→64→42虽然teacher拟合降至3.81e-5，actualleftslip=.568578724更差；拒绝该候选，不再只加BC步数。
- 实际left失败的RH载荷加权足速比teacher高约.08，分母几乎相同，约.961分子来自>=.1W承重时段且整窗都有。旧正向线性head没有饱和，不是简单界限裁剪；四统一RH输出连接增益0/.5/1.5/2都比原alpha1差，无法靠该统一缩放修复。有限失败不证明身体不可行。

## 推荐下一项受控学习问题

若继续原正向线性head（冻结197608旧模型，只调2730已有head）可以验证“均匀actionMSE没有优先控制承重RH误差”的假设，而不是再变网络/范围或增加无界预算。

需新时间戳、新PREREG，在优化前固定：

1. 父模型采用已namedactualhead2000 eeced，不按新结果挑旧checkpoint。严格载入实际head2Adam/counter2000/RNG/curve，原frozen194878和128+64状态不变；只允许正向神经head参数更新，negativezero功能精确保留。
2. 数据仍ONLY原qualifiedleft_full实际2880，首960priming不计loss。把同一次teacher trace里“决策前”的RH载荷>=.1W与关节排序RH7验证后作为训练loss权重：例如该时段RH7误差权重4、其他样本/通道1，精确归一化为加权平均。具体系数/预算须在新PREREG固定；负载仅训练评分，不输入actor/initialhidden/部署动作，不按source/phase控制神经元。不是状态机或外置矫正。
3. 可固定2000→8000追加6000、lr.001（若换lr与loss同时变化须明确工程候选而非单因果），保存5000/8000仅完整性，只实际最终8000。不要看中途MSE选择物理检查点。保持oldpositiveReLU*.25/head64linear/所有范围/身体/最终门槛不变。记录0/始端metric和actualRESUME_STATE，正确区分immediateparenthead2000与frozenmotor1500身份，保留旧metadata更正外部清单。
4. lossmask对应真实teacher决策前端点索引：原trace含1000settle+14400walking，决策k时间为1s+k*.005；k0用settle结束row999，其余row999+5*k，不可用决策后载荷当标签。样本时间可逐行核对，.1W阈值用实际冻结weight，载荷只训练mask。全2880父dualhistory因果，不截断/重置。
5. 先只读计算加权/原loss下RH与其他通道梯度、原实际headteacherfit误差承重份额，若RH承重误差已极小或梯度无法改善，明确否定该假设，不为启动实验强行作结论。可以临时unitupdate核对梯度实现，但不先搜多个权重再只登记最优。
6. 前后neutral/right全2880mean/action/192hidden位一致，全部frozenparent参数无gradient，真实缓存/生产输出和全部head梯度<=1e-12；保存moments/RNG/curve/source完整独立核验。最终actualN/L/R原all5/all6/NF0/yaw.01/full1+4.8+9.6不变；rightfullphys/neural精确保留，只有新leftactualPASS才进入range/holdout。

这个试验不是已验证答案。若仍FAIL，应保留并考虑通过物理目标直接优化ANN少量参数/源参考修正等新的问题，不能继续重复全局MSE或扩展同一增益扫描。

## 路径

head2000：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T010218Z/train/TRAIN/checkpoint_002000.pt` SHAeeced49c5bdd50a7e5bce2cfd63d56d9929b3bf2d2cb00c1985019d91321e6df；其train/model_io.py ReadoutMotor和data.py提供因果cache。
原teacher：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/teacher/left_full/{RESULT.json,SAMPLES_5MS.npz,TRACE_1MS.npz}`。
RH诊断：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T020958Z/diagnostics/DIAGNOSIS.json`。
原可用actual审计：010218Z/evaluation、013852Z/evaluation_recovery、235830Z/two_only_evaluation；路径均在同一experiments/food-approach根。

已准备的futuresteering_holdouts在020958Z/steering_holdouts，但严格绑定已失败非线性5000，不能自动给不同checkpoint使用。未来取得全3条件PASS时需明确新loader/资格amendment并实际六case验证。当前没有取得双向接口，神经减速/foodbehavior/到达/切断重放/正式native证据还未完成，goal不可complete。
