# 若本轮原两轨迹对照仍未通过：下一最小候选备忘

这只是条件性备忘，不是新的实验预登记或运行授权记录。先收齐 `two_only_evaluation/FINAL_SUMMARY.json` 和 `evaluation/FINAL_SUMMARY.json` 实际完整结果；若已有同一个检查点中性＋左右全部通过，优先开展命令变化/留出验证与食物学习，不启动下面的替代试验。

## 已有证据与尚不能作的推断

旧500慢残差在matched1s ramp下能保持前进、全六腿周期、NF0和方向，但slip左.55249/右.51403失败。纠正标签续训1500同样其余门通过，slip.54496/.56329失败。即时MLP大幅更差，但数据、训练、幅度界限不匹配，不能把慢记忆当已证明唯一稳定因素。

旧慢残差bound=.5、gate=tanh(sum(command²))。旧`RESIDUAL_RANGE_DIAGNOSIS.json`显示约2.1%教师动作在当时点态可达区间外，平均MSE下界不大，最大误差可达约.11，集中在后腿块。不能仅凭平均下界小就证明它对max逐腿滑移无关，也不能反过来证明它就是滑移原因。

## 一个可快速核验的神经读出层试验

如果仍无通过模型，可在新的时间戳目录提前登记以下单一工程候选：

1. 固定父模型为已经完成actualmatchedcontrols的旧500，SHA `040b26b7ff0e48329c6f6f8f3461f5ced783f108db3d48c264cdf78c5e35e7e3`，不按中途损失挑父模型。冻结其全部194878参数（原base151636＋旧slow残差43242）。两组显式128+64记忆、旧输出与反馈保持。
2. 新增一个神经读出 `Linear(64,42)`，输入是本决策已经由真实观察更新的旧残差hidden64；权重和bias初始化0。新pre-tanh mean = 旧mean + `old_gate * .25 * tanh(new_readout(current_residual_hidden))`。部署仍只输入实际238＋神经命令2，forward结构性0，只训练转向。不是姿态纠正器、查表、时钟或物理外力。
3. 新参数仅2730个，原慢记忆和全部旧权重不变。初始全函数必须与旧父模型在neutral/左右真实保存流上逐位一致；更新后neutral仍精确。直接目标位置负Jacobian保持，不能通过外置反馈改变物理。
4. 先测可达区间/误差下界及冻结hidden对新标签的拟合可能性，但不能把离线拟合当运动证据。该候选同时扩展神经读出容量和残差幅度，总界限从.5到最多.75，若成功不能单独归因某一个因素；身体执行器/速度/力矩上限均不变。
5. 只用原两条合格转向教师全2880观察的真实因果父hidden/mean缓存，排除前960 priming损失。原父模型冻结，缓存不包含未来；新head逐轨迹等权bounded-action MSE。新head用明确fresh Adam，而非声称继承旧8组Adam精确续训；旧优化器仅作冻结父谱系证据。可考虑提前固定2000更新/lr.003、保存1000/2000但只物理评估最终2000，具体预算需新PREREG。
6. 验证缓存输出/新head全部梯度与真正生产step序列<=1e-12、旧全部权重/两组hidden未被训练更新、临时unitstep后neutral2880精确、模型/新optimizer2状态/RNG/曲线保存核验。旧源文件一个字节不改。
7. 最终仍完整1s稳定＋4.8真实教师priming＋9.6纯ANN、中性/左右1sramp，原五门/all6/NF0/方向.01不变，独立15400端点/2880新checkpoint+192hidden链/力/身份/交接审计。通过仅固定条件工程资格，后续仍须命令变化/留出/同次native证据与食品闭环。

上述候选利用已有慢记忆、避免再训练完整循环网络，计算预算较小，值得在原模型纯续训不足时做一次明确验证。但没有证明冻结hidden足够、新读出能降低实际滑移或保证稳定，不能提前称为修复完成。

父模型/实现路径：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/residual_candidate/residual_motor.py`
`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T191449Z/residual_candidate/TRAIN_RESIDUAL_500/checkpoint_000500.pt`

本轮actual模型接口与审计路径：
`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/evaluation/`
`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/two_only_evaluation/`

神经减速尚未训练成功，不能把本转向试验当全部目标完成。食物behavior的forward必须保持0，直到另行验证速度调节；实际食物到达与感觉依赖性尚需完成。
