# 续轮3非线性正向神经读出：中性/右保持，左滑移变差

最终checkpoint：`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T020958Z/train/TRAIN/checkpoint_005000.pt`
SHA：`8e341270d82d923dcf19ec5b7286fc5797806688b3f27567f75ea3336f7216bd`

ROOTPREREG在训练和结果前固定；冻结已有head2000的197608父参数及128+64记忆，仅训练22378个新MLP神经参数。输入为允许物理112、当前basehidden128、residualhidden64及命令2，总306维，64tanh隐藏后42zerooutput；追加latent为relu(actualturn)*.10*tanh(newMLP)。forward0，负/零新支路0，无时钟/phase/source/food坐标，部署没有教师/额外力/根脚本。

子bash3固定5000训练exit0，更新62.58s/总70.50s，原合格leftteacher2880/排除960prime；MSE8.72e-5降至3.81e-5。28项预检和2500/5000各19项实际保存状态检查通过，4组Adam/newparams和完整curve/RNG、parent全部权重冻结、neutral/right保存流精确。

## 实际物理结果

parentOWNbash7三个完整实际回合及独立审计exit0收集，`evaluation/FINAL_SUMMARY.json`COMPLETE、saved_trajectory_verificationTRUE、all_conditions_eligibleFALSE。

| 条件 | 最大滑移，门槛<=.5 | yaw rad | 结论 | 独立检查 |
|---|---:|---:|---|---:|
| neutral | .49278883265 | -.00667761 | 原五门/all6/NF0通过 |126|
| left | .56857872397 | +.06067299 | 仅滑移失败，其余四门/all6/NF0/方向通过 |102|
| right | .49501104818 | -.07642541 | 原五门/all6/NF0/方向通过，完整身体/神经/192state/geometry复现合格c6b父右 |145|

原完整71mesh/42有限PD/native12.5us/无附着和额外力/无根写入、全1秒稳定+4.8秒教师+9.6秒ANN、交接连续和完整评分窗口均保持。43项最终模型准备核验和所有source_paths存在/hash检查在Core前完成；首次准备报告np.bool序列化错误只发生于无Core预检，修正后真实重跑所有43项，没有以改hash代替运行。

左转比线性父.514841585更差，即使teacher拟合误差更低也不能部署；本候选拒绝用于食物行为。不能从有限失败推断身体不可行或特定单因果。

## 下一项有限实际物理校准

只读诊断显示原左RH载荷加权足速比合格教师高约.08，而推进分母几乎相同，约.961分子来自>=.1W承重端点，且非仅启动。旧线性读出未饱和，不适合继续只加BC步数。

已在`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T024558Z/PREREG.json`提前登记：以旧线性head2000 eeced为父，固定仅四个候选将RH7读出神经weight/bias行乘0/.5/1.5/2，验证真实关节名映射和所有其他权重相同后，在完整左转实际回合按原五门/all6/NF0/yaw.01验收。没有新增动作规则或非神经控制器；有限ANN连接参数校准在既有owner授权范围。所有候选在物理前写入hash，按预登记顺序首个通过且全四结果公开，若无通过不追加第五值；通过后还需同候选实际N/R确认和后续资格。此报告不提前声称其结果。

食品行为网络、神经减速、命令变化/初态留出、正式native审计及食品到达/感觉依赖对照仍待完成。持续goal active/armed revision4，不complete。所有旧失败/哈希保持。
