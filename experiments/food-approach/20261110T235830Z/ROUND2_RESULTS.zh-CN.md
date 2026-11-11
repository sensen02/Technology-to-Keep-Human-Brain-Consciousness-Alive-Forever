# 续轮2慢残差匹配试验结果

原身体、已验收直行检查点和历史证据未修改。三组模型均在结果出现前由`ROUND2_PLAN.json`、`MATCHED_PARENT_CONTROL_AMENDMENT.json`及`TWO_TRAJECTORY_CONTROL_AMENDMENT.json`列明，只用最终指定检查点，不按中途损失挑选。

| 模型 | 左转滑移 | 右转滑移 | 中性 | 完整接口 |
|---|---:|---:|---|---|
| 冻结旧500父模型，matched1s ramp | .5524919161 | .5140340623 | 已验证PASS | FAIL |
| 原2教师＋2纠正标签，续训1500 | .5449570661 | .5632890732 | PASS | FAIL |
| 仅原2教师，同预算续训1500 | .5380394078 | .4950110482 | PASS | 左FAIL、右PASS |

全部转向的前进、足载、非足触地、质心、六腿周期和方向均通过；表中失败只在原滑移<=.5门。右转的PASS限定于完整9.6秒固定命令工程回合，不代表正式全native认证、任意切换/扰动稳健性、原连接组或食物行为。

训练和审计：

- 原2+纠正2父500→1500续训期间中断，原236退出状态未知。有效1000检查点SHA47a84f5058654f76624f897fa1d90580b2733739df64e99c234a25468260cd59及其19项独立核验存在；新`train/recovery1000/`从该实际保存状态恢复model/8Adam/RNG/fullcurve，99条旧未保存日志的损失/梯度重新计算误差0。未声称从日志恢复权重，未伪造原退出码；新恢复bash-2退出0收集。耐久1100–1400仅用于恢复，不物理选择；最终1500SHA2ab8e38b9903e5a1acccd6f6163fc2bfa29f44d8bff1ff456f79684720091a32。实际评估bash-6三回合及独立审计退出0收集。
- 原2-only匹配对照训练bash-4退出0收集，1000追加约793.65s，1500SHAc6b395f29676ef933e2f55ca5a92b7c5b55183a2d432226487b39e916b51acdb，末端拟合反弹如实保留，未改选1400。预检22/完整双history7/1000与1500保存各20项独立检查PASS。实际评估bash-7退出0收集，三回合完整来源/神经输出/192隐藏链/原始前缀/力/compiled/全窗口门与周期审计通过。
- 旧500对照的两个完整回合和审计保存后239句柄失效，退出码未知；新bash-1仅从现有完整数组恢复汇总并重新审计（积分0），两项各92checksPASS，不重复物理。

可读证据：

`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/parent_control/FINAL_REPORT.md`
`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/evaluation/FINAL_SUMMARY.json`
`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261110T235830Z/two_only_evaluation/FINAL_SUMMARY.json`

右转同次视频：`visual_right_pass/ACTUAL_RIGHT_PASS/VIDEO.json`770帧全解码/71opaque/全轮廓在画面内、源/compiled/轨迹身份保持、渲染积分0。父实际打开六阶段蒙太奇。标题RIGHT TURN ENGINEERING PASS / LEFT NOT QUALIFIED / NO FOOD明确。桌面副本位于`/home/sensen/Desktop/engine修复验收/食物趋近实施-20261110/`，交付SHA见DELIVERY.json。

基于原2-only1500已通过右转而左仍失败，在新时间目录`/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T010218Z/`提前登记了正向转弯神经读出实验：冻结全部父194878参数及双记忆，只加zero-init2730参数读出，relu(actualturn)*.25*tanh(readout(currentresidualhidden64))。固定左教师2000updates/freshAdam.003，原中性/负命令函数精确保留；不是脚本步态或外力支持。新的容量和残差范围同时变化，不能作单因果解释。

新读出训练与离线最终保存审计已完成，但实际三项验收仍在进行，不能在本报告先声称左转通过。最新状态读010218Z/CURRENT_STATE.json。神经减速、行为网络、食物到达和感觉依赖性评估仍未完成，goal保持active/armed继续。
